# todo: which of STE and EquiSTE work better for equizero
#   - create finetuned_models.py that loads finetuned models and performs equizero on them
#   - save the reported values in logs

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import os
import argparse
import time

from utils import initialize_equi_model, initialize_equi0_model, initialize_equi_eval_model, set_seed, print_details
from train import train_model, eval_model
from custom_transforms import RandomRot90
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip
from models import *


def create_dataloaders(args, input_size, data_dir, batch_size):
    # Transformations
    if args.data_aug == "":
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                # transforms.RandomResizedCrop(input_size),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(degrees=180),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                # transforms.RandomVerticalFlip(),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(degrees=180),
                transforms.ToTensor(),
                RandomRot90(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    elif args.data_aug == "rot90":
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                # transforms.RandomResizedCrop(input_size),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(degrees=180),
                transforms.ToTensor(),
                RandomRot90(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                # transforms.RandomVerticalFlip(),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(degrees=180),
                transforms.ToTensor(),
                RandomRot90(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    elif args.data_aug == "vflip":
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                # transforms.RandomResizedCrop(input_size),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(degrees=180),
                transforms.ToTensor(),
                RandomVerticalFlip(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                # transforms.RandomVerticalFlip(),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(degrees=180),
                transforms.ToTensor(),
                RandomVerticalFlip(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    # Datasets and Dataloaders
    if args.dataset == "Hymenoptera":
        print("Initializing Hymenoptera Datasets and Dataloaders...")
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
        dataloaders_dict = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
            ['train', 'val']}
    elif args.dataset == "CIFAR10":
        print("Initializing CIFAR10 Datasets and Dataloaders...")
        # create datasets
        trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=data_transforms['train'])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms['val'])
        # create dataloaders
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)
        dataloaders_dict = {'train': trainloader, 'val': testloader}
    return dataloaders_dict


def get_trainable_params(model_ft, feature_extract):
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
    return params_to_update


def load_finetuned_model(args, model):
    assert args.load_model, "Ensure to load a finetuned model to start with"
    if args.load_model:
        import os
        dir_path = os.path.join("finetuned_models")  # dir to save the trained models
        file_name = os.path.join(args.dataset + "_model_name_" + args.model_name + "_data_aug_" + args.data_aug +
                                 "_seed_" + str(args.seed) + ".pt")  # filename for the trained models

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_path = os.path.join(dir_path, file_name)
        model.load_state_dict(torch.load(file_path, map_location=args.device))
    return model


def equi_eval(args):
    print("Equi0!")
    input_size = 224

    if not args.not_verbose:
        print_details(args)

    set_seed(args.seed)
    dataloaders_dict = create_dataloaders(args, input_size=input_size, data_dir=args.data_dir, batch_size=args.batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    # load finetuned model without any symmetry
    model_ft, input_size = initialize_equi_model(args.model_name, device, args.num_classes, args.feature_extract,
                                                 use_pretrained=args.use_pretrained, symmetry_group=None)

    if args.load_model:
        model_ft = load_finetuned_model(args, model_ft)

    model_equift = Equi0FTCNN(pre_model=model_ft, num_classes=args.num_classes, symmetry_group=args.model_symmetry,
                              model_type=args.model_type, grad_estimator=args.grad_estimator)

    model_equift = model_equift.to(device)
    params_to_update = get_trainable_params(model_equift, args.feature_extract)

    optimizer_equift = optim.SGD(params_to_update, lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    print(f"evaluating model...")
    best_acc = eval_model(model_equift, dataloaders_dict, criterion, optimizer_equift, device, num_epochs=1,
              is_inception=(args.model_name == "inception"))

    print(f"best evaluation acc: {best_acc}")

    _, hist_equift, best_acc_equift = train_model(model_equift, dataloaders_dict, criterion, optimizer_equift, device,
                                                  num_epochs=args.num_epochs,
                                                  is_inception=(args.model_name == "inception"))

    return best_acc_equift


def equi0_equitune_train(args):
    print("Equi0!")
    input_size = 224

    if not args.not_verbose:
        print_details(args)

    set_seed(args.seed)
    dataloaders_dict = create_dataloaders(args, input_size=input_size, data_dir=args.data_dir, batch_size=args.batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    model_ft, input_size = initialize_equi_eval_model(args.model_name, device, args.num_classes, args.feature_extract,
                                                  use_pretrained=args.use_pretrained, symmetry_group="", model_type=args.model_type,
                                                  grad_estimator=args.grad_estimator)

    model_ft = load_finetuned_model(args, model_ft)

    model_equift = Equi0FTCNN(pre_model=model_ft, num_classes=args.num_classes, symmetry_group=args.model_symmetry,
                              grad_estimator=args.grad_estimator)

    model_equift = model_equift.to(device)
    params_to_update = get_trainable_params(model_equift, args.feature_extract)

    optimizer_equift = optim.SGD(params_to_update, lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    print(f"evaluating model...")
    eval_model(model_equift, dataloaders_dict, criterion, optimizer_equift, device, num_epochs=1,
              is_inception=(args.model_name == "inception"))

    print(f"finetuning model...")
    _, hist_equift, best_acc_equift = train_model(model_equift, dataloaders_dict, criterion, optimizer_equift, device,
                                 num_epochs=args.num_epochs,
                                 is_inception=(args.model_name == "inception"))
    return best_acc_equift


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EquiTune Parser')
    parser.add_argument('--seed_list', nargs='+', default=[0, 1, 2], type=int)
    parser.add_argument('--dataset', default="CIFAR10", type=str, help="[Hymenoptera, CIFAR10]")
    parser.add_argument('--data_aug', default="rot90", type=str, help="[, rot90]")
    parser.add_argument('--data_dir', default="./data/hymenoptera_data", type=str)
    parser.add_argument('--model_name', default="alexnet", type=str, help="[resnet, alexnet, vgg, densenet]")
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--model_type', default="equi0", type=str, help="[equitune, equi0]")
    parser.add_argument('--method_type', default="equi_eval", type=str, help="[equi_eval, equi0_equitune_train]")
    parser.add_argument('--grad_estimator', default="EquiSTE", type=str, help="[STE, equi0, eval_equi0]")
    parser.add_argument('--model_symmetry', default="rot90", type=str, help="[rot90, hflip, vflip, rot90_hflip, rot90_vflip]")
    parser.add_argument('--lr', default=0.0003, type=float)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--load_logs', action='store_true')
    parser.add_argument('-feature_extract', action='store_true', help='choose between fine-tuning and feature extraction')
    parser.add_argument('-not_verbose', action='store_false',
                        help='choose between printing details of the arguments')

    args = parser.parse_args()

    assert args.model_name in ["resnet", "alexnet", "vgg", "densenet"], "Model not implemented"
    assert args.dataset in ["Hymenoptera", "CIFAR10"], "Dataset not implemented"

    if args.dataset == "CIFAR10":
        args.num_classes = 10

    # todo: remove after debugging
    # args.use_pretrained = True
    # args.load_model = True

    best_acc_list = []

    elapsed_time_list = []
    for seed in args.seed_list:
        if args.method_type == "equi_eval":
            args.seed = seed
            start_time_None = time.time()
            best_acc = equi_eval(args)
            best_acc_list.append(best_acc)
            end_time_None = time.time()
            print(f"Symmetry: {args.model_symmetry} elapsed time: {end_time_None - start_time_None}")
            elapsed_time_list.append(end_time_None - start_time_None)

    elapsed_time_list = torch.tensor(elapsed_time_list)
    elapsed_time_mean, elapsed_time_std = torch.mean(elapsed_time_list, dim=0), torch.std(elapsed_time_list, dim=0)
    print(f"Symmetry: {args.model_symmetry}; Mean time: {elapsed_time_mean}, Std time: {elapsed_time_std}")

    best_acc_stack = torch.stack(best_acc_list)
    best_acc_mean, best_acc_std = torch.mean(best_acc_stack), torch.std(best_acc_stack)

    print("-" * 10 + f"model_type: {args.model_type}" + "-" * 10 + "\n")
    print("-" * 10 + f"method_type: {args.method_type}" + "-" * 10 + "\n")
    print(f"mean: {best_acc_mean}   std: {best_acc_std}\n")


