import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import os
import argparse
import time

from models_speed_comparison import *
from utils import initialize_equi_model, initialize_equi_model, set_seed, print_details
from train import train_model
from custom_transforms import RandomRot90
from utils import count_parameters


def create_dataloaders(args, input_size, data_dir, batch_size):
    # Transformations
    if args.data_aug is None:
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

def parallel_equitune(args):
    print("Parallel EquiTuning!")
    if not args.not_verbose:
        print_details(args)
    set_seed(args.seed)
    dataloaders_dict = create_dataloaders(args, input_size=224, data_dir=args.data_dir, batch_size=args.batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_equift = EquiFNN(input_size=224, hidden_sizes=[8, 8, 8], symmetry_group=args.model_symmetry)
    model_equift = model_equift.to(device)
    num_params = count_parameters(model_equift)
    print(f"Num of params: {num_params}")
    params_to_update = get_trainable_params(model_equift, args.feature_extract)

    optimizer_equift = optim.SGD(params_to_update, lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    import time
    start_time = time.time()
    _, hist_equift, best_acc_equift = train_model(model_equift, dataloaders_dict, criterion, optimizer_equift, device,
                                 num_epochs=args.num_epochs,
                                 is_inception=(args.model_name == "inception"))
    end_time = time.time()
    print(f"Elapsed time for Equitune: {end_time - start_time}")
    return best_acc_equift

def parallel_equitune_CNN(args):
    print("Parallel EquiTuning!")
    if not args.not_verbose:
        print_details(args)
    set_seed(args.seed)
    dataloaders_dict = create_dataloaders(args, input_size=224, data_dir=args.data_dir, batch_size=args.batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_equift = EquiCNN(in_channels=3, symmetry_group=args.model_symmetry)
    model_equift = model_equift.to(device)
    num_params = count_parameters(model_equift)
    print(f"Num of params: {num_params}")
    params_to_update = get_trainable_params(model_equift, args.feature_extract)

    optimizer_equift = optim.SGD(params_to_update, lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    import time
    start_time = time.time()
    _, hist_equift, best_acc_equift = train_model(model_equift, dataloaders_dict, criterion, optimizer_equift, device,
                                 num_epochs=args.num_epochs,
                                 is_inception=(args.model_name == "inception"))
    end_time = time.time()
    print(f"Elapsed time for Equitune: {end_time - start_time}")
    return best_acc_equift


def G_eq_train(args):
    print("G equivariant training!")
    if not args.not_verbose:
        print_details(args)
    set_seed(args.seed)
    dataloaders_dict = create_dataloaders(args, input_size=224, data_dir=args.data_dir, batch_size=args.batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # hidden size [8, 8, 8] for c4 and [11, 12, 12] for d2
    model_equift = GFNN(input_size=224, hidden_sizes=[11, 12, 12], num_classes=2, symmetry_group=None)
    model_equift = model_equift.to(device)
    num_params = count_parameters(model_equift)
    print(f"Num of params: {num_params}")
    params_to_update = get_trainable_params(model_equift, args.feature_extract)

    optimizer_equift = optim.SGD(params_to_update, lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    import time
    start_time = time.time()
    _, hist_equift, best_acc_equift = train_model(model_equift, dataloaders_dict, criterion, optimizer_equift, device,
                                 num_epochs=args.num_epochs,
                                 is_inception=(args.model_name == "inception"))
    end_time = time.time()
    print(f"Elapsed time for GFNN: {end_time - start_time}")
    return best_acc_equift


def G_eq_CNN_train(args):
    print("G equivariant training!")
    if not args.not_verbose:
        print_details(args)
    set_seed(args.seed)
    dataloaders_dict = create_dataloaders(args, input_size=224, data_dir=args.data_dir, batch_size=args.batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # hidden size [8, 8, 8] for c4 and [11, 12, 12] for d2 for FNN
    model_equift = GCNN(in_channels=3, symmetry_list=args.symmetry_list)
    model_equift = model_equift.to(device)
    num_params = count_parameters(model_equift)
    print(f"Num of params: {num_params}")
    params_to_update = get_trainable_params(model_equift, args.feature_extract)

    optimizer_equift = optim.SGD(params_to_update, lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    import time
    start_time = time.time()
    _, hist_equift, best_acc_equift = train_model(model_equift, dataloaders_dict, criterion, optimizer_equift, device,
                                 num_epochs=args.num_epochs,
                                 is_inception=(args.model_name == "inception"))
    end_time = time.time()
    print(f"Elapsed time for GCNN: {end_time - start_time}")
    return best_acc_equift


def equitune(args):
    print("EquiTuning!")
    if not args.not_verbose:
        print_details(args)
    set_seed(args.seed)
    dataloaders_dict = create_dataloaders(args, input_size=224, data_dir=args.data_dir, batch_size=args.batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_equift, input_size = initialize_equi_model(args.model_name, device, args.num_classes, args.feature_extract,
                                            use_pretrained=True, symmetry_group=args.model_symmetry)
    model_equift = model_equift.to(device)
    params_to_update = get_trainable_params(model_equift, args.feature_extract)

    optimizer_equift = optim.SGD(params_to_update, lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    _, hist_equift, best_acc_equift = train_model(model_equift, dataloaders_dict, criterion, optimizer_equift, device,
                                 num_epochs=args.num_epochs,
                                 is_inception=(args.model_name == "inception"))
    return best_acc_equift


def finetune(args):
    print("FineTuning!")
    if not args.not_verbose:
        print_details(args)
    set_seed(args.seed)
    dataloaders_dict = create_dataloaders(args, input_size=224, data_dir=args.data_dir, batch_size=args.batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft, input_size = initialize_equi_model(args.model_name, device, args.num_classes, args.feature_extract, use_pretrained=True)
    model_ft = model_ft.to(device)
    params_to_update = get_trainable_params(model_ft, args.feature_extract)

    optimizer_ft = optim.SGD(params_to_update, lr=args.lr, momentum=0.9)
    criterion_ft = nn.CrossEntropyLoss()

    _, hist_ft, best_acc_ft = train_model(model_ft, dataloaders_dict, criterion_ft, optimizer_ft, device, num_epochs=args.num_epochs,
                                 is_inception=(args.model_name == "inception"))
    return best_acc_ft


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EquiTune Parser')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--seed_list', nargs='+', default=[0, 1, 2], type=int)
    parser.add_argument('--dataset', default="Hymenoptera", type=str, help="[Hymenoptera, CIFAR10]")
    parser.add_argument('--data_aug', default=None, type=str, help="[None, rot90]")
    parser.add_argument('--data_dir', default="./data/hymenoptera_data", type=str)
    parser.add_argument('--model_type', default="param_sharing", type=str, help="[equituning, param_sharing]")
    parser.add_argument('--model_name', default="CNN", type=str, help="[FNN, CNN]")
    parser.add_argument('--model_symmetry', default="rot90_hflip", type=str, help="[rot90, hflip, vflip, rot90_hflip, rot90_vflip]")
    parser.add_argument('--lr', default=0.0003, type=float)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('-feature_extract', action='store_true', help='choose between fine-tuning and feature extraction')
    parser.add_argument('-not_verbose', action='store_false',
                        help='choose between printing details of the arguments')

    args = parser.parse_args()

    assert args.model_name in ["FNN", "CNN"], "Model not implemented"
    assert args.dataset in ["Hymenoptera", "CIFAR10"], "Dataset not implemented"

    if args.dataset == "CIFAR10":
        args.num_classes = 10

    best_acc_ft_list = []
    best_acc_rot90ft_list = []
    best_acc_rot90_hflipft_list = []

    elapsed_time_list = []
    for seed in args.seed_list:
        args.seed = seed

        if args.model_symmetry == "rot90" and args.model_type == "param_sharing" and args.model_name == "FNN":
            start_time_rot90 = time.time()
            args.model_symmetry = "rot90"
            best_acc_ft = G_eq_train(args)
            end_time_rot90 = time.time()
            best_acc_ft_list.append(best_acc_ft)
            print(f"Symmetry rot90 elapsed time: {end_time_rot90 - start_time_rot90}")
            elapsed_time_list.append(end_time_rot90 - start_time_rot90)

        elif args.model_symmetry == "rot90" and args.model_type == "param_sharing" and args.model_name == "CNN":
            start_time_rot90 = time.time()
            args.symmetry_list = ["rot90"]
            best_acc_ft = G_eq_CNN_train(args)
            end_time_rot90 = time.time()
            best_acc_ft_list.append(best_acc_ft)
            print(f"Symmetry rot90 elapsed time: {end_time_rot90 - start_time_rot90}")
            elapsed_time_list.append(end_time_rot90 - start_time_rot90)

        elif args.model_symmetry == "rot90_hflip" and args.model_type == "param_sharing" and args.model_name == "FNN":
            start_time_rot90 = time.time()
            args.model_symmetry = "rot90_hflip"
            best_acc_ft = G_eq_train(args)
            end_time_rot90 = time.time()
            best_acc_ft_list.append(best_acc_ft)
            print(f"Symmetry rot90 elapsed time: {end_time_rot90 - start_time_rot90}")
            elapsed_time_list.append(end_time_rot90 - start_time_rot90)

        elif args.model_symmetry == "rot90_hflip" and args.model_type == "param_sharing" and args.model_name == "CNN":
            start_time_rot90 = time.time()
            args.symmetry_list = ["rot90", "hflip"]
            best_acc_ft = G_eq_CNN_train(args)
            end_time_rot90 = time.time()
            best_acc_ft_list.append(best_acc_ft)
            print(f"Symmetry rot90 elapsed time: {end_time_rot90 - start_time_rot90}")
            elapsed_time_list.append(end_time_rot90 - start_time_rot90)

        elif args.model_symmetry == "rot90" and args.model_type == "equituning" and args.model_name == "FNN":
            start_time_rot90 = time.time()
            args.model_symmetry = "rot90"
            best_acc_rot90ft = parallel_equitune(args)
            end_time_rot90 = time.time()
            best_acc_rot90ft_list.append(best_acc_rot90ft)
            elapsed_time_list.append(end_time_rot90 - start_time_rot90)

        elif args.model_symmetry == "rot90" and args.model_type == "equituning" and args.model_name == "CNN":
            start_time_rot90 = time.time()
            args.symmetry_list = ["rot90"]
            best_acc_ft = parallel_equitune_CNN(args)
            end_time_rot90 = time.time()
            best_acc_ft_list.append(best_acc_ft)
            print(f"Symmetry rot90 elapsed time: {end_time_rot90 - start_time_rot90}")
            elapsed_time_list.append(end_time_rot90 - start_time_rot90)

        elif args.model_symmetry == "rot90_hflip" and args.model_type == "equituning" and args.model_name == "FNN":
            start_time_rot90 = time.time()
            args.model_symmetry = "rot90_hflip"
            best_acc_rot90ft = parallel_equitune(args)
            end_time_rot90 = time.time()
            best_acc_rot90ft_list.append(best_acc_rot90ft)
            elapsed_time_list.append(end_time_rot90 - start_time_rot90)

        elif args.model_symmetry == "rot90_hflip" and args.model_type == "equituning" and args.model_name == "CNN":
            start_time_rot90 = time.time()
            args.model_symmetry = "rot90_hflip"
            best_acc_rot90ft = parallel_equitune_CNN(args)
            end_time_rot90 = time.time()
            best_acc_rot90ft_list.append(best_acc_rot90ft)
            elapsed_time_list.append(end_time_rot90 - start_time_rot90)

    elapsed_time_list = torch.tensor(elapsed_time_list)
    elapsed_time_mean, elapsed_time_std = torch.mean(elapsed_time_list, dim=0), torch.std(elapsed_time_list, dim=0)


    print(f"Mean time: {elapsed_time_mean}, Std time: {elapsed_time_std}")



        # args.model_symmetry = None
        # best_acc_ft = finetune(args)
        # best_acc_ft_list.append(best_acc_ft)
        #
        # args.model_symmetry = "rot90"
        # best_acc_rot90ft = equitune(args)
        # best_acc_rot90ft_list.append(best_acc_rot90ft)

        # args.model_symmetry = "rot90_hflip"
        # best_acc_rot90_hflipft = equitune(args)
        # best_acc_rot90_hflipft_list.append(best_acc_rot90_hflipft)

    best_acc_ft_stack = torch.stack(best_acc_ft_list)
    best_acc_rot90ft_stack = torch.stack(best_acc_rot90ft_list)
    # # best_acc_rot90_hflipft_stack = torch.stack(best_acc_rot90_hflipft_list)
    # #
    best_acc_ft_mean, best_acc_ft_std = torch.mean(best_acc_ft_stack), torch.std(best_acc_ft_stack)
    best_acc_rot90ft_mean, best_acc_rot90ft_std = torch.mean(best_acc_rot90ft_stack), torch.std(best_acc_rot90ft_stack)
    # # best_acc_rot90_hflipft_mean, best_acc_rot90_hflipft_std = torch.mean(best_acc_rot90_hflipft_stack), torch.std(best_acc_rot90_hflipft_stack)
    #
    # print(f"dataset: {args.dataset}")
    # print(f"dataset: {args.model_name}")
    # print(f"dataset aug: {args.data_aug} \n")
    #
    print("-" * 10 + "finetuning" + "-" * 10)
    print(f"mean: {best_acc_ft_mean}   std: {best_acc_ft_std}\n")
    #
    print("-" * 10 + "rot90 equivariant finetuning" + "-" * 10)
    print(f"mean: {best_acc_rot90ft_mean}   std: {best_acc_rot90ft_std}\n")

    # print("-" * 10 + "rot90_hflip equivariant finetuning" + "-" * 10)
    # print(f"mean: {best_acc_rot90_hflipft_mean}   std: {best_acc_rot90_hflipft_std}\n")


