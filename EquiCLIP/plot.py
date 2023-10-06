import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def plot_0_imagenet(save_fig=False):
    fig = plt.figure()
    x_original = np.array([1, 4.5, 8, 11.5])
    x_flip = x_original + 1
    x_rot90 = x_original + 2

    plt.xticks(np.array([2, 5.5, 9, 12.5]), ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16'], fontsize=15, rotation=0)
    plt.yticks(fontsize=15)

    y_original = np.array([52.84, 56.16, 55.89, 61.89])
    y_flip = np.array([42.91, 45.97, 46.69, 53.56])
    y_rot90 = np.array([39.61, 43.53, 44.61, 52.78])

    # model_size = np.array([11.84, 11.84, 46.83])
    plt.bar(x_original, height=y_original, capsize=30, label="No transformation")
    plt.bar(x_flip, height=y_flip, capsize=30, label="Random flips")
    plt.bar(x_rot90, height=y_rot90, capsize=30, label="Random $90^{\circ}$ rotations")


    # plt.ylabel(ylabel="Inference time (seconds)", fontsize=15)
    plt.ylabel(ylabel="Top-$1$ accuracy", fontsize=15)
    # plt.ylabel(ylabel="Model size (MB)", fontsize=15)
    plt.ylim([35, 65])

    plt.legend(prop={'size': 12})
    plt.title("Imagenet V2 top-1 accuracies", fontsize=15)
    plt.tight_layout()
    plt.show()
    if save_fig:
        # fig.savefig("inf_time.png", dpi=150)
        fig.savefig("plot_0_imagenet.png", dpi=150)
        # fig.savefig("model_size.png", dpi=150)
    return


def plot_1_imagenet_rot90(save_fig=False):
    fig = plt.figure()
    x_original = np.array([1, 4.5, 8, 11.5])
    x_equitune = x_original + 1
    x_equizero = x_original + 2

    plt.xticks(np.array([2, 5.5, 9, 12.5]), ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16'], fontsize=15, rotation=0)
    plt.yticks(fontsize=15)

    y_original = np.array([39.61, 43.53, 44.61, 52.78])
    y_equitune = np.array([46.32, 50.19, 51.27, 58.54])
    y_equizero = np.array([50.74, 53.61, 54.08, 60.20])

    # model_size = np.array([11.84, 11.84, 46.83])
    plt.bar(x_original, height=y_original, capsize=30, label="Pretrained model")
    plt.bar(x_equitune, height=y_equitune, capsize=30, label="Equitune")
    plt.bar(x_equizero, height=y_equizero, capsize=30, label="Equizero")


    # plt.ylabel(ylabel="Inference time (seconds)", fontsize=15)
    plt.ylabel(ylabel="Top-1 accuracy", fontsize=15)
    # plt.ylabel(ylabel="Model size (MB)", fontsize=15)
    plt.ylim([35, 65])

    plt.legend(prop={'size': 12})
    plt.title("Top-1 accuracies for rot$90^{\circ}$-Imagenet V2", fontsize=15)
    plt.tight_layout()
    plt.show()
    if save_fig:
        # fig.savefig("inf_time.png", dpi=150)
        fig.savefig("plot_1_imagenet_rot90.png", dpi=150)
        # fig.savefig("model_size.png", dpi=150)
    return

def plot_1_imagenet_flip(save_fig=False):
    fig = plt.figure()
    x_original = np.array([1, 4.5, 8, 11.5])
    x_equitune = x_original + 1
    x_equizero = x_original + 2

    plt.xticks(np.array([2, 5.5, 9, 12.5]), ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16'], fontsize=15, rotation=0)
    plt.yticks(fontsize=15)

    y_original = np.array([42.91, 45.97, 46.69, 53.56])
    y_equitune = np.array([49.26, 52.48, 52.69, 59.64])
    y_equizero = np.array([51.63, 54.67, 54.91, 60.67])

    # model_size = np.array([11.84, 11.84, 46.83])
    plt.bar(x_original, height=y_original, capsize=30, label="Pretrained model")
    plt.bar(x_equitune, height=y_equitune, capsize=30, label="Equitune")
    plt.bar(x_equizero, height=y_equizero, capsize=30, label="Equizero")


    # plt.ylabel(ylabel="Inference time (seconds)", fontsize=15)
    plt.ylabel(ylabel="Top-1 accuracy", fontsize=15)
    # plt.ylabel(ylabel="Model size (MB)", fontsize=15)
    plt.ylim([40, 65])

    plt.legend(prop={'size': 12})
    plt.title("Top-1 accuracies for flip-Imagenet V2", fontsize=15)
    plt.tight_layout()
    plt.show()
    if save_fig:
        # fig.savefig("inf_time.png", dpi=150)
        fig.savefig("plot_1_imagenet_flip.png", dpi=150)
        # fig.savefig("model_size.png", dpi=150)
    return


def plot_0_cifar100(save_fig=False):
    fig = plt.figure()
    x_original = np.array([1, 4.5, 8, 11.5])
    x_flip = x_original + 1
    x_rot90 = x_original + 2

    plt.xticks(np.array([2, 5.5, 9, 12.5]), ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16'], fontsize=15, rotation=0)
    plt.yticks(fontsize=15)

    y_original = np.array([41.77, 48.83, 65.02, 68.25])
    y_flip = np.array([30.13, 34.76, 49.76, 52.80])
    y_rot90 = np.array([24.33, 28.26, 43.91, 46.49])

    # model_size = np.array([11.84, 11.84, 46.83])
    plt.bar(x_original, height=y_original, capsize=30, label="No transformation")
    plt.bar(x_flip, height=y_flip, capsize=30, label="Random flips")
    plt.bar(x_rot90, height=y_rot90, capsize=30, label="Random $90^{\circ}$ rotations")


    # plt.ylabel(ylabel="Inference time (seconds)", fontsize=15)
    plt.ylabel(ylabel="Top-1 accuracy", fontsize=15)
    # plt.ylabel(ylabel="Model size (MB)", fontsize=15)
    plt.ylim([22, 70])

    plt.legend(prop={'size': 12})
    plt.title("CIFAR100 top-1 accuracies", fontsize=15)
    plt.tight_layout()
    plt.show()
    if save_fig:
        # fig.savefig("inf_time.png", dpi=150)
        fig.savefig("plot_0_cifar100.png", dpi=150)
        # fig.savefig("model_size.png", dpi=150)
    return


def plot_1_cifar100_rot90(save_fig=False):
    fig = plt.figure()
    x_original = np.array([1, 4.5, 8, 11.5])
    x_equitune = x_original + 1
    x_equizero = x_original + 2

    plt.xticks(np.array([2, 5.5, 9, 12.5]), ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16'], fontsize=15, rotation=0)
    plt.yticks(fontsize=15)

    y_original = np.array([24.33, 28.26, 43.91, 46.49])
    y_equitune = np.array([28.35, 34.02, 54.88, 57.35])
    y_equizero = np.array([38.87, 45.96, 63.03, 66.44])

    # model_size = np.array([11.84, 11.84, 46.83])
    plt.bar(x_original, height=y_original, capsize=30, label="Pretrained model")
    plt.bar(x_equitune, height=y_equitune, capsize=30, label="Equitune")
    plt.bar(x_equizero, height=y_equizero, capsize=30, label="Equizero")


    # plt.ylabel(ylabel="Inference time (seconds)", fontsize=15)
    plt.ylabel(ylabel="Top-1 accuracy", fontsize=15)
    # plt.ylabel(ylabel="Model size (MB)", fontsize=15)
    plt.ylim([22, 70])

    plt.legend(prop={'size': 12})
    plt.title("Top-1 accuracies for rot$90^{\circ}$-CIFAR100", fontsize=15)
    plt.tight_layout()
    plt.show()
    if save_fig:
        # fig.savefig("inf_time.png", dpi=150)
        fig.savefig("plot_1_cifar100_rot90.png", dpi=150)
        # fig.savefig("model_size.png", dpi=150)
    return

def plot_1_cifar100_flip(save_fig=False):
    fig = plt.figure()
    x_original = np.array([1, 4.5, 8, 11.5])
    x_equitune = x_original + 1
    x_equizero = x_original + 2

    plt.xticks(np.array([2, 5.5, 9, 12.5]), ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16'], fontsize=15, rotation=0)
    plt.yticks(fontsize=15)

    y_original = np.array([30.13, 34.76, 49.76, 52.80])
    y_equitune = np.array([35.44, 42.89, 61.16, 64.17])
    y_equizero = np.array([40.15, 47.32, 63.95, 66.87])

    # model_size = np.array([11.84, 11.84, 46.83])
    plt.bar(x_original, height=y_original, capsize=30, label="Pretrained model")
    plt.bar(x_equitune, height=y_equitune, capsize=30, label="Equitune")
    plt.bar(x_equizero, height=y_equizero, capsize=30, label="Equizero")


    # plt.ylabel(ylabel="Inference time (seconds)", fontsize=15)
    plt.ylabel(ylabel="Top-1 accuracy", fontsize=15)
    # plt.ylabel(ylabel="Model size (MB)", fontsize=15)
    plt.ylim([27.5, 68.5])

    plt.legend(prop={'size': 12})
    plt.title("Top-1 accuracies for rot$90^{\circ}$-CIFAR100", fontsize=15)
    plt.tight_layout()
    plt.show()
    if save_fig:
        # fig.savefig("inf_time.png", dpi=150)
        fig.savefig("plot_1_cifar100_flip.png", dpi=150)
        # fig.savefig("model_size.png", dpi=150)
    return


def plot_cifar100_finetuning(save_fig=False, backbone='RN50', backbone_list=['RN50', 'RN101', 'ViTB32', 'ViTB16']):
    linewidth = 1
    if 'RN50' in backbone_list:
        backbone = "RN"
        num_steps = [0, 1e3, 2e3, 3e3, 4e3]
        equitune = [28.35, 29.26, 30.48, 32.53, 34.01]
        unequitune = [31.38, 34.07, 40.42, 44.18, 46.54]
        equizero = [38.87, 39.49, 39.95, 40.11, 40.24]
        plt.plot(num_steps, equizero, label="RN50 Equizero", color='green', linestyle='--', marker='o', linewidth=linewidth)
        plt.plot(num_steps, equitune, label="RN50 Equitune", color='darkorange', linestyle='dotted', marker='s', linewidth=linewidth)
        plt.plot(num_steps, unequitune, label="RN50 (Un)equitune", color='royalblue', linestyle='solid', marker='^', linewidth=linewidth)
    if 'ViTB32' in backbone_list:
        backbone = "ViT"
        num_steps = [0, 1e3, 2e3, 3e3, 4e3]
        equitune = [54.88, 55.57, 55.74, 56.05, 56.06]
        unequitune = [56.45, 56.58, 56.38, 56.60, 56.97]
        equizero = [63.03, 63.69, 63.79, 63.93, 63.82]
        plt.plot(num_steps, equizero, label="ViTB32 Equizero", color='green', linestyle='--', marker='o', linewidth=linewidth)
        plt.plot(num_steps, equitune, label="ViTB32 Equitune", color='darkorange', linestyle='dotted', marker='s', linewidth=linewidth)
        plt.plot(num_steps, unequitune, label="ViTB32 (Un)equitune", color='royalblue', linestyle='solid', marker='^', linewidth=linewidth)
    if 'RN101' in backbone_list:
        num_steps = [0, 1e3, 2e3, 3e3, 4e3]
        equitune = [34.02, 34.66, 36.02, 40.28, 41.18]
        unequitune = [36.82, 46.94, 49.16, 49.81, 49.06]
        equizero = [45.96, 46.47, 46.95, 46.94, 46.64]
        plt.plot(num_steps, equizero, label="RN101 Equizero", color='green', linestyle='--', marker="D", linewidth=linewidth)
        plt.plot(num_steps, equitune, label="RN101 Equitune", color='darkorange', linestyle='dotted', marker="*", linewidth=linewidth)
        plt.plot(num_steps, unequitune, label="RN101 (Un)equitune", color='royalblue', linestyle='solid', marker="X", linewidth=linewidth)
    if 'ViTB16' in backbone_list:
        num_steps = [0, 1e3, 2e3, 3e3, 4e3]
        equitune = [57.35, 59.39, 59.06, 58.94, 58.90]
        unequitune = [61.95, 70.39, 70.92, 71.74, 72.19]
        equizero = [66.44, 67.23, 67.28, 67.33, 67.30]
        plt.plot(num_steps, equizero, label="ViTB16 Equizero", color='green', linestyle='--', marker="D", linewidth=linewidth)
        plt.plot(num_steps, equitune, label="ViTB16 Equitune", color='darkorange', linestyle='dotted', marker="*", linewidth=linewidth)
        plt.plot(num_steps, unequitune, label="ViTB16 (Un)equitune", color='royalblue', linestyle='solid', marker="X", linewidth=linewidth)
    else:
        NotImplementedError('backbone used is not implemented yet or please check your spelling')

    
    # plt.grid(True)
    plt.title("Top-1 accuracies" + f" for {backbone}"+ " on rot$90^{\circ}$-CIFAR100", fontsize=15)
    plt.legend(prop={'size': 12}, loc="lower right")
    plt.xlabel('Finetuning steps', fontsize=15)
    plt.ylabel('Top-1 accuracy', fontsize=15)
    plt.xticks(num_steps)
    plt.tight_layout()
    if save_fig:
        plt.savefig('plot_cifar100_finetuning_' + '_'.join(backbone_list) + '.png', dpi=250)
        plt.close()



def plot_lambda_weights():
    linewidth = 2
    import torch
    from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
    import matplotlib.image as image
    # image0 = image.imread("visualize_lambda_images/imageData_0.png")
    # image1 = image.imread("visualize_lambda_images/imageData_1.png")
    # image2 = image.imread("visualize_lambda_images/imageData_2.png")
    # image3 = image.imread("visualize_lambda_images/imageData_3.png")
    #
    # imagebox0 = OffsetImage(image0, zoom=0.15)
    # imagebox1 = OffsetImage(image1, zoom=0.15)
    # imagebox2 = OffsetImage(image2, zoom=0.15)
    # imagebox3 = OffsetImage(image3, zoom=0.15)

    x = [90, 180, 270, 360]
    lambda_rn50 = torch.tensor([101.3, 279.3, 164., 347.5])
    lambda_rn50 = lambda_rn50 / sum(lambda_rn50)
    lambda_rn101 = torch.tensor([18.92, 21.14, 24.08, 27.2])
    lambda_rn101 = lambda_rn101 / sum(lambda_rn101)
    lambda_vit32 = torch.tensor([42.44, 30.75, 44.53, 57.47])
    lambda_vit32 = lambda_vit32 / sum(lambda_vit32)
    lambda_vit16 = torch.tensor([244.9, 437.8, 374.3, 366.8])
    lambda_vit16 = lambda_vit16 / sum(lambda_vit16)

    plt.plot(x, lambda_rn50, label="$\lambda$ RN50", color='green',
             linewidth=linewidth)
    plt.plot(x, lambda_rn101, label="$\lambda$ RN101", color='royalblue',
             linewidth=linewidth)
    # plt.plot(x, lambda_vit32, label="$\lambda$ ViTB32", color='darkorange',
    #          linewidth=linewidth)
    # plt.plot(x, lambda_vit16, label="$\lambda$ ViTB16", color='red',
    #          linewidth=linewidth)

    plt.title("$\lambda$ weights for an image from CIFAR100", fontsize=15)
    plt.legend(prop={'size': 12}, loc="lower right")
    plt.xlabel('Rotation angle', fontsize=15)
    plt.ylabel('$\lambda$ values', fontsize=15)
    plt.xticks(x)
    plt.tight_layout()
    plt.savefig('lambda_weights.png', dpi=250)
    plt.close()






if __name__ == "__main__":
    #plot_0_imagenet(True)
    #plot_0_cifar100(True)
    #plot_1_imagenet_rot90(True)
    #plot_1_imagenet_flip(True)
    #plot_1_cifar100_rot90(True)
    #plot_1_cifar100_flip(True)
    # plot_cifar100_finetuning(save_fig=True, backbone_list=['RN50', 'RN101'])
    # plot_cifar100_finetuning(save_fig=True, backbone_list= ['RN50', 'RN101'])
    # plot_cifar100_finetuning(save_fig=True, backbone_list= ['RN101'])
    # plot_cifar100_finetuning(save_fig=True, backbone_list= ['ViTB32'])
    # plot_cifar100_finetuning(save_fig=True, backbone_list= ['ViTB32', 'ViTB16'])
    plot_lambda_weights()
