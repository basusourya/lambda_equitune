import torch

from torchvision.transforms.functional import hflip, vflip
from utils import initialize_equi_model, initialize_equi0_model, initialize_eval_equi0_model, count_parameters
from models_speed_comparison import *


def vflip_eq_test(model_name, symmetry_group, feat_extract):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, input_size = initialize_equi_model(model_name, device, 2, feat_extract, use_pretrained=True, symmetry_group=symmetry_group)

    input = torch.randn(size=(1, 3, 224, 224))
    vflip_input = vflip(input)

    output = model(input)
    eqvflip_output = model(vflip_input)

    print(f"Eq Error: {torch.norm(output - eqvflip_output)}")
    assert torch.allclose(output, eqvflip_output)


def hflip_eq_test(model_name, symmetry_group, feat_extract):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, input_size = initialize_equi_model(model_name, device, 2, feat_extract, use_pretrained=True, symmetry_group=symmetry_group)

    input = torch.randn(size=(1, 3, 224, 224))
    hflip_input = hflip(input)

    output = model(input)
    eqhflip_output = model(hflip_input)

    print(f"Eq Error: {torch.norm(output - eqhflip_output)}")
    assert torch.allclose(output, eqhflip_output)


def rot90_parallel_eq_test(model_name, symmetry_group, feat_extract, model_type="equi0", grad_estimator="STE", eval=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model_type == "equitune":
        model, input_size = initialize_equi_model(model_name, device, 2, feat_extract, use_pretrained=True,
                                                  symmetry_group=symmetry_group)
    elif model_type == "equi0":
        model, input_size = initialize_equi0_model(model_name, device, 2, feat_extract, use_pretrained=True,
                                                  symmetry_group=symmetry_group, grad_estimator=grad_estimator)
    elif model_type == "eval_equi0":
        model, input_size = initialize_eval_equi0_model(model_name, device, 2, feat_extract, use_pretrained=True,
                                                  symmetry_group=symmetry_group)
    else:
        raise NotImplementedError

    if eval:
        "needed for checking eval_equizero"
        model.eval()
    else:
        model.train()

    input = torch.randn(size=(2, 3, 224, 224))
    rot_input = torch.rot90(input, k=3, dims=(2, 3))

    output = model(input)
    eqrot_output = model(rot_input)

    print(f"Eq Error: {torch.norm(output - eqrot_output)}")
    assert torch.allclose(output, eqrot_output)


def rot90_eqfnn_parallel_eq_test(model_name, symmetry_group, feat_extract):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EquiFNN(input_size=224, hidden_sizes=[4, 4, 4], num_classes=2, symmetry_group="rot90")
    input = torch.randn(size=(1, 3, 224, 224))
    rot_input = torch.rot90(input, k=1, dims=(2, 3))

    num_params = count_parameters(model)
    print(f"Num of params: {num_params}")

    output = model(input)
    eqrot_output = model(rot_input)

    print(f"Eq Error: {torch.norm(output - eqrot_output)}")
    assert torch.allclose(output, eqrot_output)


def rot90_Gfnn_parallel_eq_test(model_name, symmetry_group, feat_extract):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GFNN(input_size=224, hidden_sizes=[8, 8, 8], num_classes=2, symmetry_group="rot90")
    input = torch.randn(size=(1, 3, 224, 224))
    rot_input = torch.rot90(input, k=1, dims=(2, 3))

    num_params = count_parameters(model)
    print(f"Num of params: {num_params}")

    output = model(input)
    eqrot_output = model(rot_input)

    print(f"Eq Error: {torch.norm(output - eqrot_output)}")
    assert torch.allclose(output, eqrot_output)


if __name__ == '__main__':
    test_name = "rot90parallel"  # choose from ["rot90", "hflip", "vflip", "rot90_hflip", None]
    model_name = "alexnet"  # choose from ["resnet", "alexnet", "vgg", "densenet"]
    symmetry_group = "rot90"  # choose from ["rot90", "hflip", "vflip", "rot90_hflip", "rot90_vflip", None]
    feat_extract = False  # * 6 * 6  # 512 for resnet, 256 * 6 * 6 for alexnet
    model_type = "equi0"
    grad_estimator="STE"
    eval=True

    if test_name == "rot90parallel":
        rot90_parallel_eq_test(model_name=model_name, symmetry_group=symmetry_group, feat_extract=feat_extract,
                               model_type=model_type, grad_estimator=grad_estimator, eval=eval)
    elif test_name == "rot90eqfnnparallel":
        rot90_eqfnn_parallel_eq_test(model_name=model_name, symmetry_group=symmetry_group, feat_extract=feat_extract)
    elif test_name == "rot90Gfnnparallel":
        rot90_Gfnn_parallel_eq_test(model_name=model_name, symmetry_group=symmetry_group, feat_extract=feat_extract)
    elif test_name == "hflip":
        hflip_eq_test(model_name=model_name, symmetry_group=symmetry_group, feat_extract=feat_extract)
    elif test_name == "vflip":
        vflip_eq_test(model_name=model_name, symmetry_group=symmetry_group, feat_extract=feat_extract)
    else:
        raise NotImplementedError
