import torch

from tqdm import tqdm
from exp_utils import group_transform_images, random_transformed_images



def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()  # dim [max_topk, batch_size]
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def equi0_accuracy(output, target, topk=(1,), group_name=""):
    if group_name == "":
      pred_indices = output.topk(max(topk), 1, True, True)[1].t()  # dim [max_topk, batch_size]
      correct = pred_indices.eq(target.view(1, -1).expand_as(pred_indices))  # dim [max_topk, batch_size]
      # return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
    elif group_name == "rot90":
      group_size = 4
      output_shape = output.shape
      output = output.reshape(group_size, output_shape[0] // group_size, output_shape[1])  # [group_size, batch_size, num_classes]
      # pred_values, pred_indices = output.topk(max(topk), 2, True, True)[0],\
      #                             output.topk(max(topk), 2, True, True)[1]  # dim [group_size, batch_size, max_topk]
      # correct = pred_indices[0].t().eq(target.view(1, -1).expand_as(pred_indices[0].t()))  # dim [max_topk, group_size * batch_size]
      # return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
      output, _ = torch.max(output, dim=0, keepdim=False)  # [batch_size, num_classes]
      pred_values, pred_indices = output.topk(max(topk), 1, True, True)[0].t(), \
                                  output.topk(max(topk), 1, True, True)[1].t()  # dim [max_topk, group_size, batch_size]
      correct = pred_indices.eq(target.view(1, -1).expand_as(pred_indices))  # dim [max_topk, group_size * batch_size]
      # return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
    elif group_name == "flip":
      group_size = 2
      output_shape = output.shape
      output = output.reshape(group_size, output_shape[0] // group_size, output_shape[1])  # [group_size, batch_size, num_classes]
      # pred_values, pred_indices = output.topk(max(topk), 2, True, True)[0],\
      #                             output.topk(max(topk), 2, True, True)[1]  # dim [group_size, batch_size, max_topk]
      # correct = pred_indices[0].t().eq(target.view(1, -1).expand_as(pred_indices[0].t()))  # dim [max_topk, group_size * batch_size]
      # return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
      output, _ = torch.max(output, dim=0, keepdim=False)  # [batch_size, num_classes]
      pred_values, pred_indices = output.topk(max(topk), 1, True, True)[0].t(), \
                                  output.topk(max(topk), 1, True, True)[1].t()  # dim [max_topk, group_size, batch_size]
      correct = pred_indices.eq(target.view(1, -1).expand_as(pred_indices))  # dim [max_topk, group_size * batch_size]
      # return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
    else:
      raise NotImplementedError

    return pred_indices


def equitune_accuracy(output, target, topk=(1,), group_name=""):
    if group_name == "":
        pred_indices = output.topk(max(topk), 1, True, True)[1].t()  # dim [max_topk, batch_size]
        correct = pred_indices.eq(target.view(1, -1).expand_as(pred_indices))  # dim [max_topk, batch_size]
        # return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
    elif group_name=="rot90":
        group_size = 4
        output_shape = output.shape
        output = output.reshape(group_size, output_shape[0] // group_size, output_shape[1])  # [group_size, batch_size, num_classes]
        output = torch.mean(output, dim=0, keepdim=False)  # [batch_size, num_classes]
        pred_values, pred_indices = output.topk(max(topk), 1, True, True)[0].t(),\
                                  output.topk(max(topk), 1, True, True)[1].t()  # dim [max_topk, group_size, batch_size]
        correct = pred_indices.eq(target.view(1, -1).expand_as(pred_indices))  # dim [max_topk, group_size * batch_size]
        # return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
    elif group_name == "flip":
        group_size = 2
        output_shape = output.shape
        output = output.reshape(group_size, output_shape[0] // group_size, output_shape[1])  # [group_size, batch_size, num_classes]
        output = torch.mean(output, dim=0, keepdim=False)  # [batch_size, num_classes]
        pred_values, pred_indices = output.topk(max(topk), 1, True, True)[0].t(),\
                                  output.topk(max(topk), 1, True, True)[1].t()  # dim [max_topk, batch_size]
        correct = pred_indices.eq(target.view(1, -1).expand_as(pred_indices))  # dim [max_topk, batch_size]
        # return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
    else:
        raise NotImplementedError

    return pred_indices


def eval_compare_clip_ImagenetV2(args, model, zeroshot_weights, loader, data_transformations="", group_name=""):
    # comparison between vanilla, equitune, and equizero for ImagenetV2
    import time
    since = time.time()
    with torch.no_grad():
        top1_v_y_eq_n_eo_n = 0.
        top1_v_y_eq_n_eo_y = 0.
        top1_v_y_eq_y_eo_n = 0.
        top1_v_y_eq_y_eo_y = 0.

        top1_v_n_eq_n_eo_n = 0.
        top1_v_n_eq_n_eo_y = 0.
        top1_v_n_eq_y_eo_n = 0.
        top1_v_n_eq_y_eo_y = 0.

        top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(tqdm(loader)):
            images = images.cuda()  # dim [batch_size, c_in, H, H]
            batch_size = images.shape[0]
            num_classes = zeroshot_weights.shape[-1]
            images = random_transformed_images(images, data_transformations=data_transformations)  # randomly transform data

            # images = torch.rot90(images, k=1, dims=(-2, -1))
            group_images = group_transform_images(images,
                                                  group_name=group_name)  # dim [group_size, batch_size, c_in, H, H]
            group_images_shape = group_images.shape

            # dim [group_size * batch_size, c_in, H, H]
            group_images = group_images.reshape(group_images_shape[0] * group_images_shape[1], group_images_shape[2],
                                                group_images_shape[3], group_images_shape[3])
            # print(f"images.shape: {images.shape}")
            target = target.cuda()
            # print(f"target.shape: {target.shape}")

            # predict
            image_features = model.encode_image(group_images)  # dim [group_size * batch_size, feat_size=512]
            # print(f"image_features.shape: {image_features.shape}")
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # zeroshot weights correspond to text features for all possible classes
            # logits = 100. * image_features @ zeroshot_weights  # dim [group_size * batch_size, num_classes=1000]
            logits = args.logit_factor * image_features @ zeroshot_weights  # dim [group_size * batch_size, num_classes=1000]
            # logits = -torch.nn.functional.softmax(-logits, dim=-1)
            logits = torch.nn.functional.softmax(logits, dim=-1)
            # print(f"logits.shape: {logits.shape}")

            v_pred = equi0_accuracy(logits.reshape(-1, batch_size, num_classes)[0], target, topk=(1,), group_name="")
            v_correct = v_pred.eq(target.view(1, -1).expand_as(v_pred))

            eq_pred = equitune_accuracy(logits, target, topk=(1,), group_name=group_name)
            eq_correct = eq_pred.eq(target.view(1, -1).expand_as(eq_pred))

            eq0_pred = equi0_accuracy(logits, target, topk=(1,), group_name=group_name)
            eq0_correct = eq0_pred.eq(target.view(1, -1).expand_as(eq0_pred))

            top1_v_y_eq_n_eo_n += torch.sum((v_correct) * (~eq_correct) * (~eq0_correct))
            top1_v_y_eq_n_eo_y += torch.sum((v_correct) * (~eq_correct) * (eq0_correct))
            top1_v_y_eq_y_eo_n += torch.sum((v_correct) * (eq_correct) * (~eq0_correct))
            top1_v_y_eq_y_eo_y += torch.sum((v_correct) * (eq_correct) * (eq0_correct))

            top1_v_n_eq_n_eo_n += torch.sum((~v_correct) * (~eq_correct) * (~eq0_correct))
            top1_v_n_eq_n_eo_y += torch.sum((~v_correct) * (~eq_correct) * (eq0_correct))
            top1_v_n_eq_y_eo_n += torch.sum((~v_correct) * (eq_correct) * (~eq0_correct))
            top1_v_n_eq_y_eo_y += torch.sum((~v_correct) * (eq_correct) * (eq0_correct))



            n += images.size(0)

    top1_v_y_eq_n_eo_n = (top1_v_y_eq_n_eo_n / n) * 100
    top1_v_y_eq_n_eo_y = (top1_v_y_eq_n_eo_y / n) * 100
    top1_v_y_eq_y_eo_n = (top1_v_y_eq_y_eo_n / n) * 100
    top1_v_y_eq_y_eo_y = (top1_v_y_eq_y_eo_y / n) * 100

    top1_v_n_eq_n_eo_n = (top1_v_n_eq_n_eo_n / n) * 100
    top1_v_n_eq_n_eo_y = (top1_v_n_eq_n_eo_y / n) * 100
    top1_v_n_eq_y_eo_n = (top1_v_n_eq_y_eo_n / n) * 100
    top1_v_n_eq_y_eo_y = (top1_v_n_eq_y_eo_y / n) * 100


    print(f"top1_v_y_eq_n_eo_n: {top1_v_y_eq_n_eo_n}")
    print(f"top1_v_y_eq_n_eo_y: {top1_v_y_eq_n_eo_y}")
    print(f"top1_v_y_eq_y_eo_n: {top1_v_y_eq_y_eo_n}")
    print(f"top1_v_y_eq_y_eo_y: {top1_v_y_eq_y_eo_y}")

    print(f"top1_v_n_eq_n_eo_n: {top1_v_n_eq_n_eo_n}")
    print(f"top1_v_n_eq_n_eo_y: {top1_v_n_eq_n_eo_y}")
    print(f"top1_v_n_eq_y_eo_n: {top1_v_n_eq_y_eo_n}")
    print(f"top1_v_n_eq_y_eo_y: {top1_v_n_eq_y_eo_y}")


    current_time = time.time()
    time_elapsed = current_time - since
    print(f"time elapsed: {time_elapsed}")


def eval_compare_clip_cifar_class(args, model, zeroshot_weights, loader, data_transformations="", group_name=""):
    # comparison between vanilla, equitune, and equizero for ImagenetV2
    import time
    since = time.time()
    with torch.no_grad():
        top1_v_y_eq_n_eo_n = 0.
        top1_v_y_eq_n_eo_y = 0.
        top1_v_y_eq_y_eo_n = 0.
        top1_v_y_eq_y_eo_y = 0.

        top1_v_n_eq_n_eo_n = 0.
        top1_v_n_eq_n_eo_y = 0.
        top1_v_n_eq_y_eo_n = 0.
        top1_v_n_eq_y_eo_y = 0.

        top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(tqdm(loader)):
            images = images.cuda()  # dim [batch_size, c_in, H, H]
            batch_size = images.shape[0]
            num_classes = zeroshot_weights.shape[-1]
            images = random_transformed_images(images, data_transformations=data_transformations)  # randomly transform data

            # images = torch.rot90(images, k=1, dims=(-2, -1))
            group_images = group_transform_images(images,
                                                  group_name=group_name)  # dim [group_size, batch_size, c_in, H, H]
            group_images_shape = group_images.shape

            # dim [group_size * batch_size, c_in, H, H]
            group_images = group_images.reshape(group_images_shape[0] * group_images_shape[1], group_images_shape[2],
                                                group_images_shape[3], group_images_shape[3])
            # print(f"images.shape: {images.shape}")
            target = target.cuda()
            # print(f"target.shape: {target.shape}")

            # predict
            image_features = model.encode_image(group_images)  # dim [group_size * batch_size, feat_size=512]
            # print(f"image_features.shape: {image_features.shape}")
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # zeroshot weights correspond to text features for all possible classes
            # logits = 100. * image_features @ zeroshot_weights  # dim [group_size * batch_size, num_classes=1000]
            logits = args.logit_factor * image_features @ zeroshot_weights  # dim [group_size * batch_size, num_classes=1000]
            # logits = -torch.nn.functional.softmax(-logits, dim=-1)
            logits = torch.nn.functional.softmax(logits, dim=-1)
            # print(f"logits.shape: {logits.shape}")

            v_pred = equi0_accuracy(logits.reshape(-1, batch_size, num_classes)[0], target, topk=(1,), group_name="")
            v_correct = v_pred.eq(target.view(1, -1).expand_as(v_pred))

            eq_pred = equitune_accuracy(logits, target, topk=(1,), group_name=group_name)
            eq_correct = eq_pred.eq(target.view(1, -1).expand_as(eq_pred))

            eq0_pred = equi0_accuracy(logits, target, topk=(1,), group_name=group_name)
            eq0_correct = eq0_pred.eq(target.view(1, -1).expand_as(eq0_pred))

            top1_v_y_eq_n_eo_n += torch.sum((v_correct) * (~eq_correct) * (~eq0_correct))
            top1_v_y_eq_n_eo_y += torch.sum((v_correct) * (~eq_correct) * (eq0_correct))
            top1_v_y_eq_y_eo_n += torch.sum((v_correct) * (eq_correct) * (~eq0_correct))
            top1_v_y_eq_y_eo_y += torch.sum((v_correct) * (eq_correct) * (eq0_correct))

            top1_v_n_eq_n_eo_n += torch.sum((~v_correct) * (~eq_correct) * (~eq0_correct))
            top1_v_n_eq_n_eo_y += torch.sum((~v_correct) * (~eq_correct) * (eq0_correct))
            top1_v_n_eq_y_eo_n += torch.sum((~v_correct) * (eq_correct) * (~eq0_correct))
            top1_v_n_eq_y_eo_y += torch.sum((~v_correct) * (eq_correct) * (eq0_correct))



            n += images.size(0)

    top1_v_y_eq_n_eo_n = (top1_v_y_eq_n_eo_n / n) * 100
    top1_v_y_eq_n_eo_y = (top1_v_y_eq_n_eo_y / n) * 100
    top1_v_y_eq_y_eo_n = (top1_v_y_eq_y_eo_n / n) * 100
    top1_v_y_eq_y_eo_y = (top1_v_y_eq_y_eo_y / n) * 100

    top1_v_n_eq_n_eo_n = (top1_v_n_eq_n_eo_n / n) * 100
    top1_v_n_eq_n_eo_y = (top1_v_n_eq_n_eo_y / n) * 100
    top1_v_n_eq_y_eo_n = (top1_v_n_eq_y_eo_n / n) * 100
    top1_v_n_eq_y_eo_y = (top1_v_n_eq_y_eo_y / n) * 100


    print(f"top1_v_y_eq_n_eo_n: {top1_v_y_eq_n_eo_n}")
    print(f"top1_v_y_eq_n_eo_y: {top1_v_y_eq_n_eo_y}")
    print(f"top1_v_y_eq_y_eo_n: {top1_v_y_eq_y_eo_n}")
    print(f"top1_v_y_eq_y_eo_y: {top1_v_y_eq_y_eo_y}")

    print(f"top1_v_n_eq_n_eo_n: {top1_v_n_eq_n_eo_n}")
    print(f"top1_v_n_eq_n_eo_y: {top1_v_n_eq_n_eo_y}")
    print(f"top1_v_n_eq_y_eo_n: {top1_v_n_eq_y_eo_n}")
    print(f"top1_v_n_eq_y_eo_y: {top1_v_n_eq_y_eo_y}")


    current_time = time.time()
    time_elapsed = current_time - since
    print(f"time elapsed: {time_elapsed}")