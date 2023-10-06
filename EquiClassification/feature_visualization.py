import os
import imageio.v3 as iio
import imageio
import torch
import argparse
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets, transforms
from torchvision.transforms.functional import hflip, vflip
from utils import initialize_equi_model, set_seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def get_trainsets(args, input_size):
    # Transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Datasets and Dataloaders
    print("Initializing CIFAR10 Datasets and Dataloaders...")
    # create datasets
    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=data_transforms['train'])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms['val'])
    return trainset


def save_features(args):
    set_seed(args.seed)
    eq_model, input_size = initialize_equi_model(args.model, device, 2, feature_extract=True, use_pretrained=True,
                                              symmetry_group=args.model_symmetry, visualize_features=True)
    noneq_model, input_size = initialize_equi_model(args.model, device, 2, feature_extract=True, use_pretrained=True,
                                                 symmetry_group=None, visualize_features=True)

    trainset = get_trainsets(args, input_size=224)

    # get a sample from trainset
    input_image, label = trainset[77]
    input_image = input_image.unsqueeze(dim=0)
    print(f"label: {cifar10_classes[label]}")

    # get rotated images
    rotated_input_images = []
    for i in range(4):
        rotated_input_images.append(torch.rot90(input_image, k=3-i, dims=(2, 3)))


    # get equivariant features
    eq_output_features = []
    for i in range(4):
        eq_output_features.append(eq_model(rotated_input_images[i]))

    # get nonequivariant features
    noneq_output_features = []
    for i in range(4):
        noneq_output_features.append(noneq_model(rotated_input_images[i]))

    # plot equivariant and nonequivariant features
    for i in range(4):
        plt.axis('off')
        plt.imshow(torchvision.utils.make_grid(rotated_input_images[i][0]).permute(1, 2, 0))
        # plt.show()
        plt.savefig(args.saved_features_dir + "input_" + str(i) + ".png", bbox_inches='tight', pad_inches=0)

        plt.imshow(torchvision.utils.make_grid(eq_output_features[i][0, :3]).permute(1, 2, 0))
        plt.savefig(args.saved_features_dir + "eq_features_" + str(i) + ".png", bbox_inches='tight', pad_inches=0)

        plt.imshow(torchvision.utils.make_grid(noneq_output_features[i][0, :3]).permute(1, 2, 0))
        plt.savefig(args.saved_features_dir + "noneq_features_" + str(i) + ".png", bbox_inches='tight', pad_inches=0)



def create_gifs(args):
    input_filenames = []
    eq_features_filenames = []
    noneq_features_filenames = []

    for i in range(4):
        input_filenames.append(args.saved_features_dir + "input" + "_" + str(i) + ".png")

    for i in range(4):
        eq_features_filenames.append(args.saved_features_dir + "eq_features" + "_" + str(i) + ".png")

    for i in range(4):
        noneq_features_filenames.append(args.saved_features_dir + "noneq_features" + "_" + str(i) + ".png")


    images = []
    for i in range(4):
        input_img = iio.imread(input_filenames[i])
        eq_features_img = iio.imread(eq_features_filenames[i])
        noneq_features_img = iio.imread(noneq_features_filenames[i])

        img = np.concatenate([eq_features_img, input_img, noneq_features_img], axis=1)
        images.append(img)
    imageio.mimsave('feature_visualizations/saved_gifs/' + 'features_' + str(i) + '.gif', images, duration=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for feature visualizations")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--model", default="alexnet", type=str)
    parser.add_argument("--dataset", default="CIFAR10", type=str)
    parser.add_argument("--model_symmetry", default="rot90", type=str)
    parser.add_argument('--saved_features_dir', default="./feature_visualizations/saved_features/", type=str)
    parser.add_argument('--saved_gifs_dir', default="./feature_visualizations/saved_gifs/", type=str)

    args = parser.parse_args()

    # save equivariant and non-equivariant features
    save_features(args)

    # create gifs
    create_gifs(args)




