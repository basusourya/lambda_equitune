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
from train import train_model
from custom_transforms import RandomRot90


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


def equitune(args):
    print("EquiTuning!")
    if not args.not_verbose:
        print_details(args)
    set_seed(args.seed)
    dataloaders_dict = create_dataloaders(args, input_size=224, data_dir=args.data_dir, batch_size=args.batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_equift, input_size = initialize_equi_model(args.model_name, device, args.num_classes, args.feature_extract,
                                            use_pretrained=args.use_pretrained, symmetry_group=args.model_symmetry,
                                            eval_type=args.eval_type)
    model_equift = model_equift.to(device)
    params_to_update = get_trainable_params(model_equift, args.feature_extract)

    optimizer_equift = optim.SGD(params_to_update, lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model_equift, hist_equift, best_acc_equift = train_model(model_equift, dataloaders_dict, criterion,
                                                  optimizer_equift, device, num_epochs=args.num_epochs,
                                                  is_inception=(args.model_name == "inception"))

    if args.save_model:
        import os
        dir_path = os.path.join("finetuned_models")  # dir to save the trained models
        file_name = os.path.join(args.dataset + "_model_name_" + args.model_name + "_data_aug_" + args.data_aug +
                                 "_seed_" + str(args.seed) + ".pt")  # filename for the trained models

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_path = os.path.join(dir_path, file_name)
        torch.save(model_equift.state_dict(), file_path)

    return best_acc_equift


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EquiTune Parser')
    parser.add_argument('--seed_list', nargs='+', default=[0, 1, 2], type=int)
    parser.add_argument('--dataset', default="Hymenoptera", type=str, help="[Hymenoptera, CIFAR10]")
    parser.add_argument('--data_aug', default="", type=str, help="[, rot90]")
    parser.add_argument('--data_dir', default="./data/hymenoptera_data", type=str)
    parser.add_argument('--model_name', default="alexnet", type=str, help="[resnet, alexnet, vgg, densenet]")
    parser.add_argument('--eval_type', default="equitune", type=str, help="[equitune, equi0]")
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--model_type', default="equitune", type=str, help="[equitune, equi0, eval_equi0]")
    parser.add_argument('--grad_estimator', default="STE", type=str, help="[STE, equi0, eval_equi0]")
    parser.add_argument('--model_symmetry', default="", type=str, help="[rot90, hflip, vflip, rot90_hflip, rot90_vflip]")
    parser.add_argument('--lr', default=0.0003, type=float)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('-feature_extract', action='store_true', help='choose between fine-tuning and feature extraction')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_logs', action='store_true')
    parser.add_argument('-not_verbose', action='store_false',
                        help='choose between printing details of the arguments')

    args = parser.parse_args()

    assert args.model_name in ["resnet", "alexnet", "vgg", "densenet"], "Model not implemented"
    assert args.dataset in ["Hymenoptera", "CIFAR10"], "Dataset not implemented"

    if args.dataset == "CIFAR10":
        args.num_classes = 10

    best_acc_list = []

    elapsed_time_list = []
    for seed in args.seed_list:
        # no symmetry used => equitune = finetune
        args.seed = seed
        start_time_None = time.time()
        best_acc = equitune(args)
        best_acc_list.append(best_acc)
        end_time_None = time.time()
        elapsed_time_list.append(end_time_None - start_time_None)


