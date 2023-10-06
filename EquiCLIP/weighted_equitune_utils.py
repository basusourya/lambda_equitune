import torch
import torch.nn.functional as F

from tqdm import tqdm
from exp_utils import group_transform_images, random_transformed_images

group_sizes = {"rot90": 4., "flip": 2., "": 1.}

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()  # dim [max_topk, batch_size]
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def get_equi0_output(output, target, topk=(1,), group_name=""):
    if group_name == "":
      return output
    elif group_name == "rot90":
      group_size = 4
      output_shape = output.shape
      output = output.reshape(group_size, output_shape[0] // group_size, output_shape[1])  # [group_size, batch_size, num_classes]
      output, _ = torch.max(output, dim=0, keepdim=False)  # [batch_size, num_classes]
      return output
    elif group_name == "flip":
      group_size = 2
      output_shape = output.shape
      output = output.reshape(group_size, output_shape[0] // group_size, output_shape[1])  # [group_size, batch_size, num_classes]
      output, _ = torch.max(output, dim=0, keepdim=False)  # [batch_size, num_classes]
      return output
    else:
      raise NotImplementedError


def get_equitune_output(output, target, topk=(1,), group_name=""):
    if group_name == "":
        pred = output.topk(max(topk), 1, True, True)[1].t()  # dim [max_topk, batch_size]
        return output
    elif group_name=="rot90":
        group_size = 4
        output_shape = output.shape
        output = output.reshape(group_size, output_shape[0] // group_size, output_shape[1])  # [group_size, batch_size, num_classes]
        output = torch.mean(output, dim=0, keepdim=False)  # [batch_size, num_classes]
        return output
    elif group_name == "flip":
        group_size = 2
        output_shape = output.shape
        output = output.reshape(group_size, output_shape[0] // group_size, output_shape[1])  # [group_size, batch_size, num_classes]
        output = torch.mean(output, dim=0, keepdim=False)  # [batch_size, num_classes]
        return output
    else:
        raise NotImplementedError


def weighted_equitune_clip(args, model, weight_net, optimizer, criterion, zeroshot_weights, loader, data_transformations="", group_name="",
                           num_iterations=100, iter_print_freq=10, device="cuda:0", model_=None):
    import time
    torch.autograd.set_detect_anomaly(True)
    since = time.time()
    top1, top5, n = 0., 0., 0.
    training_iterator = cycle(iter(loader))
    # for i, (images, target) in enumerate(tqdm(loader)):
    import time
    st_time = time.time()
    for i in range(num_iterations):
        if (i+1)%iter_print_freq == 0:
            print(f"iteration number: {i+1}")
            curr_time = time.time()
            print(f"time elapsed per iter: {(curr_time - st_time) / (i + 1)}")
        (images, target) = next(training_iterator)
        images = images.to(device)  # dim [batch_size, c_in, H, H]
        images = random_transformed_images(images, data_transformations=data_transformations)  # randomly transform data

        # images = torch.rot90(images, k=1, dims=(-2, -1))
        group_images = group_transform_images(images,
                                              group_name=group_name)  # dim [group_size, batch_size, c_in, H, H]
        group_images_shape = group_images.shape

        # dim [group_size * batch_size, c_in, H, H]
        group_images = group_images.reshape(group_images_shape[0] * group_images_shape[1], group_images_shape[2],
                                            group_images_shape[3], group_images_shape[3])
        target = target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # predict
        image_features = model.encode_image(group_images)  # dim [group_size * batch_size, feat_size=512]
        if not model_ is None:
            image_features_ = model_.encode_image(group_images)  # dim [group_size * batch_size, feat_size=512]

        # print(f"image_features.shape: {image_features.shape}")
        image_features_norm = image_features.clone().norm(dim=-1, keepdim=True)
        image_features = image_features / image_features_norm

        if not model_ is None:
            image_features_norm_ = image_features_.clone().norm(dim=-1, keepdim=True)
            image_features_ = image_features_ / image_features_norm_



        # weighted image features
        # use .half since the model is in fp16
        # normalize group weights proportional to size of group_size
        group_weights = weight_net(image_features_.float()).half()  # dim [group_size * batch_size, feat_size]
        # group_weights = group_weights.reshape(group_images_shape[0], -1, 1)
        # group_weights = F.softmax(group_weights, dim=0)
        # weight_sum = torch.sum(group_weights, dim=0, keepdim=True)
        # print(f"weight_sum: {weight_sum}")
        # print(f"group weights: {group_weights.permute(1, 0, 2)}")
        # group_size = group_sizes[args.group_name]
        # group_weights = group_size * (group_weights / weight_sum)
        # group_weights = group_weights.reshape(-1, 1)
        image_features = torch.einsum('ij, ik -> ij', image_features.clone(), group_weights)


        # zeroshot weights correspond to text features for all possible classes
        # logits = 100. * image_features @ zeroshot_weights  # dim [group_size * batch_size, num_classes=1000]

        # IMPORTANT NOTE: higher logit factors automatically biases the model towards the one with higher scores, hence,
        # acts like (un)equituning naturally even without lambda
        logits = args.logit_factor * image_features @ zeroshot_weights  # dim [group_size * batch_size, num_classes=1000]


        logits = torch.nn.functional.softmax(logits, dim=-1)
        # print(f"logits.shape: {logits.shape}")

        # measure accuracy
        if args.method == "equitune":
            output = get_equitune_output(logits, target, topk=(1,), group_name=group_name)  # dim [batch_size, num_classes=1000]
        elif args.method == "equizero":
            equitune_output = get_equitune_output(logits, target, topk=(1,), group_name=group_name)
            equi0_output = get_equi0_output(logits, target, topk=(1,), group_name=group_name)
            output = equitune_output + (equi0_output - equitune_output).detach()
        else:
            output = get_equi0_output(logits, target, topk=(1,), group_name="")

        ## backprop
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


    return model