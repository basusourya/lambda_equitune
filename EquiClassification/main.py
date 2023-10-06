import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import os
import argparse
import time

from utils import initialize_equi_model, initialize_equi0_model, set_seed, print_details
from train import train_model, eval_model
from custom_transforms import RandomRot90
import pickle as pkl 


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
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train'])
        testset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['val'])
        # create dataloaders
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
        dataloaders_dict = {'train': trainloader, 'val': testloader}
    elif args.dataset == "STL10":
        print("Initializing STL10 Datasets and Dataloaders...")
        # create datasets
        trainset = torchvision.datasets.STL10(root='data', split='train', download=True, transform=data_transforms['train'])
        testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=data_transforms['val'])
        # create dataloaders
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_equift, input_size = initialize_equi_model(args.model_name, device, args.num_classes, args.feature_extract,
                                            use_pretrained=args.use_pretrained, symmetry_group=args.model_symmetry,
                                            eval_type=args.eval_type)

    model_equift = model_equift.to(device)
    params_to_update = get_trainable_params(model_equift, args.feature_extract)

    optimizer_equift = optim.SGD(params_to_update, lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    _, hist_equift, best_acc_equift = train_model(model_equift, dataloaders_dict, criterion, optimizer_equift, device,
                                 num_epochs=args.num_epochs,
                                 is_inception=(args.model_name == "inception"))
    return hist_equift, best_acc_equift


def equi0(args):
    print("Equi0!")
    if not args.not_verbose:
        print_details(args)
    set_seed(args.seed)
    dataloaders_dict = create_dataloaders(args, input_size=224, data_dir=args.data_dir, batch_size=args.batch_size)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_equift, input_size = initialize_equi0_model(args.model_name, device, args.num_classes, args.feature_extract,
                                            use_pretrained=args.use_pretrained, symmetry_group=args.model_symmetry,
                                            grad_estimator=args.grad_estimator, use_softmax=args.use_softmax, use_e_loss=args.use_e_loss, 
                                            use_ori_equizero=args.use_ori_equizero, use_entropy=args.use_entropy)
    model_equift = model_equift.to(device)
    params_to_update = get_trainable_params(model_equift, args.feature_extract)

    optimizer_equift = optim.SGD(params_to_update, lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    best_acc = eval_model(model_equift, dataloaders_dict, criterion, optimizer_equift, device, num_epochs=1, is_inception=False,
                compute_distance=False)

    t1, hist_equift, best_acc_equift = train_model(model_equift, dataloaders_dict, criterion, optimizer_equift, device,
                                 num_epochs=args.num_epochs,
                                 is_inception=(args.model_name == "inception"), phases=['train', 'val'], batch_size=args.batch_size)

    return hist_equift, best_acc_equift


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EquiTune Parser')
    parser.add_argument('--seed_list', nargs='+', default=[0, 1, 2], type=int)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--dataset', default="Hymenoptera", type=str, help="[Hymenoptera, CIFAR10]")
    parser.add_argument('--data_aug', default="rot90", type=str, help="[, rot90]")
    parser.add_argument('--data_dir', default="./data/", type=str)
    parser.add_argument('--model_name', default="alexnet", type=str, help="[resnet, alexnet, vgg, densenet]")
    parser.add_argument('--eval_type', default="equi0", type=str, help="[equitune, equi0]")
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--use_softmax', action='store_true')
    parser.add_argument('--use_e_loss', action='store_true')
    parser.add_argument('--use_ori_equizero', action='store_true')
    parser.add_argument('--use_entropy', action='store_true')
    parser.add_argument('--model_type', default="equi0", type=str, help="[equitune, equi0, eval_equi0]")
    parser.add_argument('--grad_estimator', default="EquiSTE", type=str, help="[STE, equi0, eval_equi0]")
    parser.add_argument('--model_symmetry', default="rot90", type=str, help="[rot90, hflip, vflip, rot90_hflip, rot90_vflip]")
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)

    parser.add_argument('-feature_extract', action='store_true', help='choose between fine-tuning and feature extraction')
    parser.add_argument('-not_verbose', action='store_false',
                        help='choose between printing details of the arguments')
    parser.add_argument('--save_logs', action='store_true')

    args = parser.parse_args()

    assert args.model_name in ["resnet", "alexnet", "vgg", "densenet"], "Model not implemented"
    assert args.dataset in ["Hymenoptera", "CIFAR10", "STL10"], "Dataset not implemented"

    if args.dataset == "CIFAR10":
        args.num_classes = 10
    elif args.dataset == "STL10":
        args.num_classes = 10

    best_acc_list = []

    elapsed_time_list = []
    for seed in args.seed_list:
        args.seed = seed
        if args.model_type == "equitune":
            start_time_None = time.time()
            output = equitune(args)
            best_acc = output[1]
            with open('./logs/' + args.model_name + '_' + args.model_type + '_' + args.eval_type + '_' + str(args.use_ori_equizero) + '_' + str(args.use_entropy) + '_' + str(seed) + '.pkl', 'wb') as f:
                pkl.dump(output, f)
            best_acc_list.append(best_acc)
            end_time_None = time.time()
            print(f"Symmetry: {args.model_symmetry} elapsed time: {end_time_None - start_time_None}")
            elapsed_time_list.append(end_time_None - start_time_None)
        elif args.model_type == "equi0":
            start_time_None = time.time()
            output = equi0(args)
            best_acc = output[1]
            with open('./logs/' + args.model_name + '_' + args.model_type + '_' + args.eval_type + '_' + str(args.use_ori_equizero) + '_' + str(args.use_entropy) + '_' + str(seed) + '.pkl', 'wb') as f:
                pkl.dump(output, f)
            print ("equitune", best_acc)
            best_acc_list.append(best_acc)
            end_time_None = time.time()
            print(f"Symmetry: {args.model_symmetry} elapsed time: {end_time_None - start_time_None}")
            elapsed_time_list.append(end_time_None - start_time_None)


    elapsed_time_list = torch.tensor(elapsed_time_list)
    elapsed_time_mean, elapsed_time_std = torch.mean(elapsed_time_list, dim=0), torch.std(elapsed_time_list, dim=0)
    print(f"Symmetry: {args.model_symmetry}; Mean time: {elapsed_time_mean}, Std time: {elapsed_time_std}")
    print (best_acc_list)
    best_acc_stack = torch.stack(best_acc_list)
    best_acc_mean, best_acc_std = torch.mean(best_acc_stack), torch.std(best_acc_stack)

    print("-" * 10 + f"finetuning method: {args.model_type}" + "-" * 10)
    print(f"mean: {best_acc_mean}   std: {best_acc_std}\n")

    if args.save_logs:
        # define file_path to save logs
        import os
        dir_path = os.path.join("logs")  # dir to save logs
        file_name = os.path.join(args.dataset + args.model_name + "__data_aug__" + args.data_aug + "__model_type__"
                                 + args.model_type + "__grad_estimator__" + args.grad_estimator + ".txt")  # filename for the trained models

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_path = os.path.join(dir_path, file_name)

        # write to the file_path
        with open(file_path, "w") as f:
            print_details(args, f)
            f.write("-" * 10 + f"finetuning method: {args.model_type}" + "-" * 10 + "\n")
            f.write(f"Symmetry: {args.model_symmetry}; Mean time: {elapsed_time_mean}, Std time: {elapsed_time_std}")
            f.write(f"mean: {best_acc_mean}   std: {best_acc_std}\n")



