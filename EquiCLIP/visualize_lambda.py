import os

import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import clip
import copy
import argparse
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torchvision


from tqdm import tqdm
from pkg_resources import packaging
from weight_models import WeightNet
from load_model import load_model
from weighted_equitune_utils import weighted_equitune_clip
from dataset_utils import imagenet_classes, imagenet_templates, get_labels_textprompts, get_dataloader, get_ft_dataloader, get_ft_visualize_dataloader
from zeroshot_weights import zeroshot_classifier
from eval_utils import eval_clip
from torch.utils.tensorboard import SummaryWriter

print("Torch version:", torch.__version__)

# observed lambda_values
# RN50
def plot_lambda_weights():
    x = [0, 1, 2, 3]
    lambda_rn50 = torch.tensor([101.3, 279.3, 164., 347.5])
    lambda_rn50 = lambda_rn50 / sum(lambda_rn50)
    lambda_rn101 = torch.tensor([18.92, 21.14, 24.08, 27.2])
    lambda_rn101 = lambda_rn101 / sum(lambda_rn101)
    lambda_vit32 = torch.tensor([42.44, 30.75, 44.53, 57.47])
    lambda_vit32 = lambda_vit32 / sum(lambda_vit32)
    lambda_vit16 = torch.tensor([244.9, 437.8, 374.3, 366.8])
    lambda_vit16 = lambda_vit16 / sum(lambda_vit16)

def main(args):
    # load model and preprocess
    model, preprocess = load_model(args)
    model_, preprocess_ = load_model(args)

    # load weight network
    weight_net = WeightNet(args)
    weight_net.to(args.device)

    # get labels and text prompts
    classnames, templates = get_labels_textprompts(args)

    # get dataloader
    train_loader, eval_loader = get_ft_dataloader(args, preprocess)
    viz_train_loader, viz_eval_loader = get_ft_visualize_dataloader(args, preprocess)

    # optimizer and loss criterion
    criterion = nn.CrossEntropyLoss()
    # only weight_net is trained not the model itself
    optimizer1 = optim.SGD(weight_net.parameters(), lr=args.prelr, momentum=0.9)

    # create text weights for different classes
    zeroshot_weights = zeroshot_classifier(args, model, classnames, templates, save_weights='True').to(args.device)

    best_top1 = 0.0
    best_model_weights = copy.deepcopy(weight_net.state_dict())
    MODEL_DIR = "saved_weight_net_models"

    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    if args.model_name in ['ViT-B/32', 'ViT-B/16']:
        args.save_model_name = ''.join(args.model_name.split('/'))
    else:
        args.save_model_name = args.model_name

    MODEL_NAME = f"{args.dataset_name}_{args.save_model_name}_aug_{args.data_transformations}_eq_{args.group_name}" \
                 f"_steps_{args.num_prefinetunes}.pt"
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

    if os.path.isfile(MODEL_PATH):
        weight_net.load_state_dict(torch.load(MODEL_PATH))

    else:
        for i in range(args.num_prefinetunes):
            print(f"weighted equitune number: {i}")
            # zeroshot prediction
            # add weight_net save code for the best model

            # evaluating for only 50 steps using val=True
            top1 = eval_clip(args, model, zeroshot_weights, train_loader, data_transformations=args.data_transformations,
                      group_name=args.group_name, device=args.device, weight_net=weight_net, val=True, model_=model_)

            if top1 > best_top1:
                best_top1 = top1
                best_model_weights = copy.deepcopy(weight_net.state_dict())

            # finetune prediction
            model = weighted_equitune_clip(args, model, weight_net, optimizer1, criterion, zeroshot_weights, train_loader, data_transformations=args.data_transformations,
                                           group_name=args.group_name, num_iterations=args.iter_per_prefinetune,
                                           iter_print_freq=args.iter_print_freq, device=args.device, model_=model_)


        torch.save(best_model_weights, MODEL_PATH)
        weight_net.load_state_dict(torch.load(MODEL_PATH))

    # compute lambda for different transformed images
    writer = SummaryWriter()
    for i, data in enumerate(eval_loader):
        x, y = data   # we only care about 1 image not the entire batch
        x = x.to(args.device)
        print(f"y: {y}")
        x_group = []
        for j in range(4):
            x = torch.rot90(x, k=1, dims=(-1, -2))
            x_group.append(x)
            image_features = model.encode_image(x)  # dim [group_size * batch_size, feat_size=512]
            weights = weight_net(image_features.float()).half()
            for k in range(len(weights)):
                weight = weights[k]
                writer.add_scalar(f'lambda{k}', weight, j)
        break

    for i, data in enumerate(viz_eval_loader):
        x, y = data  # we only care about 1 image not the entire batch
        x = x.to(args.device)
        for j in range(4):
            x = torch.rot90(x, k=1, dims=(-1, -2))

            for k in range(len(x)):
                grid = torchvision.utils.make_grid(x[k])
                writer.add_image(f'images{k}', grid, j)

        break
    writer.close()







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Weighted equituning')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default='cuda:0', type=str)
    parser.add_argument("--k", default=1., type=int)
    parser.add_argument("--img_num", default=0, type=int)
    parser.add_argument("--num_prefinetunes", default=10, type=int, help="num of iterations for learning the lambda weights")
    parser.add_argument("--num_finetunes", default=8, type=int)
    parser.add_argument("--iter_per_prefinetune", default=100, type=int)
    parser.add_argument("--iter_per_finetune", default=500, type=int)
    parser.add_argument("--iter_print_freq", default=50, type=int)
    parser.add_argument("--logit_factor", default=1., type=float)
    parser.add_argument("--prelr", default=0.33, type=float)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--data_transformations", default="", type=str, help=["", "flip", "rot90"])
    parser.add_argument("--group_name", default="", type=str, help=["", "flip", "rot90"])
    parser.add_argument("--method", default="equitune", type=str, help=["vanilla", "equitune", "equizero"])
    parser.add_argument("--model_name", default="RN50", type=str, help=['RN50', 'RN101', 'RN50x4', 'RN50x16',
                                                                        'RN50x64', 'ViT-B/32', 'ViT-B/16',
                                                                        'ViT-L/14', 'ViT-L/14@336px'])
    parser.add_argument("--dataset_name", default="ImagenetV2", type=str, help=["ImagenetV2", "CIFAR100"])
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    args.verbose = True

    pl.seed_everything(args.seed)
    main(args)

# python main_weighted_equitune.py  --dataset_name CIFAR100  --logit_factor 1.0  --iter_per_finetune 500 --method equitune --group_name rot90 --data_transformations rot90  --model_name 'ViT-B/16' --lr 0.000005 --num_finetunes 10 --num_prefinetunes 20 --k -10 --prelr 0.33