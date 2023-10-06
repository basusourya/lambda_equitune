from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights

import matplotlib.pyplot as plt
import os
import random

from models import *


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_equi_model(model_name, device, num_classes, feature_extract, use_pretrained=True, symmetry_group=None,
                          eval_type="equitune"):
    # Initialize equivariant models
    # initialize each model with pretrained weights of [:-1] layers
    # use 2 layers before output instead of 1 (to have better impact of equivariance)

    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        pre_model = models.resnet18(pretrained=use_pretrained).to(device)
        set_parameter_requires_grad(pre_model, feature_extract)
        feat_size = 512  # pre_model.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        model_ft = EquiCNN(pre_model=pre_model, input_size=input_size, feat_size=feat_size, num_classes=num_classes,
                                   symmetry_group=symmetry_group, model_name="resnet", eval_type=eval_type)

    elif model_name == "alexnet":
        """ Alexnet
        """
        pre_model = models.alexnet(pretrained=use_pretrained).to(device)
        set_parameter_requires_grad(pre_model, feature_extract)
        feat_size = 256  # * 6 * 6  # pre_model.classifier[6].in_features
        # model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        model_ft = EquiCNN(pre_model=pre_model, input_size=input_size, feat_size=feat_size, num_classes=num_classes,
                                   symmetry_group=symmetry_group, model_name="alexnet", eval_type=eval_type)

    elif model_name == "vgg":
        """ VGG11_bn
        """
        pre_model = models.vgg11_bn(pretrained=use_pretrained).to(device)
        set_parameter_requires_grad(pre_model, feature_extract)
        feat_size = 512  # pre_model.classifier[6].in_features
        # model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        model_ft = EquiCNN(pre_model=pre_model, input_size=input_size, feat_size=feat_size, num_classes=num_classes,
                                   symmetry_group=symmetry_group, model_name="vgg", eval_type=eval_type)

    elif model_name == "densenet":
        """ Densenet
        """
        pre_model = models.densenet121(pretrained=use_pretrained).to(device)
        set_parameter_requires_grad(pre_model, feature_extract)
        feat_size = 1024  # pre_model.classifier.in_features
        # model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        model_ft = EquiCNN(pre_model=pre_model, input_size=input_size, feat_size=feat_size, num_classes=num_classes,
                                   symmetry_group=symmetry_group, model_name="densenet", eval_type=eval_type)
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def initialize_equi0_model(model_name, device, num_classes, feature_extract, use_pretrained=True, symmetry_group=None,
                           grad_estimator="STE", use_softmax=True, use_e_loss=False, use_ori_equizero=False, use_entropy=False):
    # Initialize equivariant models
    # initialize each model with pretrained weights of [:-1] layers
    # use 2 layers before output instead of 1 (to have better impact of equivariance)

    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        pre_model = models.resnet18(pretrained=use_pretrained).to(device)
        set_parameter_requires_grad(pre_model, feature_extract)
        feat_size = 512  # pre_model.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        model_ft = Equi0CNN(pre_model=pre_model, input_size=input_size, feat_size=feat_size, num_classes=num_classes,
                                   symmetry_group=symmetry_group, model_name="resnet", grad_estimator=grad_estimator, 
                                   use_softmax=use_softmax, use_e_loss=use_e_loss, use_ori_equizero=use_ori_equizero, use_entropy=use_entropy)

    elif model_name == "alexnet":
        """ Alexnet
        """
        pre_model = models.alexnet(pretrained=use_pretrained).to(device)
        set_parameter_requires_grad(pre_model, feature_extract)
        feat_size = 256  # * 6 * 6  # pre_model.classifier[6].in_features
        # model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        model_ft = Equi0CNN(pre_model=pre_model, input_size=input_size, feat_size=feat_size, num_classes=num_classes,
                                   symmetry_group=symmetry_group, model_name="alexnet", grad_estimator=grad_estimator, 
                                   use_softmax=use_softmax, use_e_loss=use_e_loss, use_ori_equizero=use_ori_equizero, use_entropy=use_entropy)

    elif model_name == "vgg":
        """ VGG11_bn
        """
        pre_model = models.vgg11_bn(pretrained=use_pretrained).to(device)
        set_parameter_requires_grad(pre_model, feature_extract)
        feat_size = 512  # pre_model.classifier[6].in_features
        # model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        model_ft = Equi0CNN(pre_model=pre_model, input_size=input_size, feat_size=feat_size, num_classes=num_classes,
                                   symmetry_group=symmetry_group, model_name="vgg", grad_estimator=grad_estimator, 
                                   use_softmax=use_softmax, use_e_loss=use_e_loss, use_ori_equizero=use_ori_equizero, use_entropy=use_entropy)

    elif model_name == "densenet":
        """ Densenet
        """
        pre_model = models.densenet121(pretrained=use_pretrained).to(device)
        set_parameter_requires_grad(pre_model, feature_extract)
        feat_size = 1024  # pre_model.classifier.in_features
        # model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        model_ft = Equi0CNN(pre_model=pre_model, input_size=input_size, feat_size=feat_size, num_classes=num_classes,
                                   symmetry_group=symmetry_group, model_name="densenet", grad_estimator=grad_estimator, 
                                   use_softmax=use_softmax, use_e_loss=use_e_loss, use_ori_equizero=use_ori_equizero, use_entropy=use_entropy)
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def initialize_equi_eval_model(model_name, device, num_classes, feature_extract, use_pretrained=True, symmetry_group=None,
                               model_type="equitune", grad_estimator="EquiSTE"):
    # Initialize equivariant models
    # initialize each model with pretrained weights of [:-1] layers
    # use 2 layers before output instead of 1 (to have better impact of equivariance)

    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        pre_model = models.resnet18(pretrained=use_pretrained).to(device)
        set_parameter_requires_grad(pre_model, feature_extract)
        feat_size = 512  # pre_model.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        model_ft = Equi0FTCNN(pre_model=pre_model, num_classes=num_classes,
                                   symmetry_group=symmetry_group, model_type=model_type, grad_estimator=grad_estimator)

    elif model_name == "alexnet":
        """ Alexnet
        """
        pre_model = models.alexnet(pretrained=use_pretrained).to(device)
        set_parameter_requires_grad(pre_model, feature_extract)
        feat_size = 256  # * 6 * 6  # pre_model.classifier[6].in_features
        # model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        model_ft = Equi0FTCNN(pre_model=pre_model, num_classes=num_classes,
                                   symmetry_group=symmetry_group, model_type=model_type, grad_estimator=grad_estimator)

    elif model_name == "vgg":
        """ VGG11_bn
        """
        pre_model = models.vgg11_bn(pretrained=use_pretrained).to(device)
        set_parameter_requires_grad(pre_model, feature_extract)
        feat_size = 512  # pre_model.classifier[6].in_features
        # model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        model_ft = Equi0FTCNN(pre_model=pre_model, num_classes=num_classes,
                                   symmetry_group=symmetry_group, model_type=model_type, grad_estimator=grad_estimator)

    elif model_name == "densenet":
        """ Densenet
        """
        pre_model = models.densenet121(pretrained=use_pretrained).to(device)
        set_parameter_requires_grad(pre_model, feature_extract)
        feat_size = 1024  # pre_model.classifier.in_features
        # model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        model_ft = Equi0FTCNN(pre_model=pre_model, num_classes=num_classes,
                                   symmetry_group=symmetry_group, model_type=model_type, grad_estimator=grad_estimator)
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize non-equivariant models
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def plot(hist_ft, hist_sc, hist_equift, num_epochs):
    hist_ft_list = [h.cpu().numpy() for h in hist_ft]
    # hist_sc_list = [h.cpu().numpy() for h in hist_sc]
    hist_equift_list = [h.cpu().numpy() for h in hist_equift]

    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, num_epochs + 1), hist_ft_list, label="Pretrained")
    # plt.plot(range(1, num_epochs + 1), hist_sc_list, label="Scratch")
    plt.plot(range(1, num_epochs + 1), hist_equift_list, label="EquiTune")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    plt.show()
    plt.savefig("plots/fine_tuning.png")


def print_details(args, f):
    f.write(f"seed: {args.seed}\n")
    f.write(f"model name: {args.model_name}\n")
    f.write(f"model symmetry: {args.model_symmetry}\n")
    f.write(f"model type: {args.model_type}\n")
    f.write(f"dataset: {args.dataset}\n")
    f.write(f"data_aug: {args.data_aug}\n")
    f.write(f"num_epochs: {args.num_epochs}\n")
    f.write(f"lr: {args.lr}\n")
    f.write(f"grad estimator: {args.grad_estimator}\n")
    f.write(f"feature_extract: {args.feature_extract}\n")
    f.write(f"batch_size: {args.batch_size}\n")
    return





