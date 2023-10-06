import torch.nn as nn
import time

from eq_utils import *
from modules import GLinear
from torchvision.transforms.functional import hflip, vflip
from torch.nn.parameter import Parameter
import torch.nn.functional as F


# to show equituning has the exact same performance as parameter-sharing


class GConv2d(nn.Module):
    """
    Initialization:
      Input dimensions: (batch_size; c_in; H; H)
      Output dimensions: (batch_size; c_out; H'; H')
      filter dimensions: (c_out, c_in, kernel_size, kernel_size)
      symmetry_list: list of equivariant symmetries to be used
    Forward:
      input: data
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=True, symmetry_list=['rot90']):

        super(GConv2d, self).__init__()
        self.c_in = in_channels
        self.c_out = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.symmetry_list = symmetry_list

        self.eq_indices = get_equivariance_indices(nx=self.kernel_size ** 2, nh=1,
                                                   symmetry_list=self.symmetry_list)  # get equivariant indices
        self.transformations_list = None  # used for orbit averaging init when weight_pre is given
        self.num_params = max(self.eq_indices) + 1
        self.weight = Parameter(torch.Tensor(self.c_in, self.c_out, self.num_params))  # trainable parameters

        if bias:
            self.bias = Parameter(torch.Tensor(self.c_out))
        else:
            self.bias = self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.c_in * self.c_out * self.kernel_size ** 2
        stdv = 1. / math.sqrt(n)

        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        weight = self.weight
        eq_weight_shape = (self.c_out, self.c_in, self.kernel_size, self.kernel_size)

        assert self.eq_indices is not None, "eq_indices must not be None"
        weight_g = weight.view(self.c_in, self.c_out, -1)  # shape = (c_out, c_in, num_params)
        weight_g = weight_g[:, :, self.eq_indices]  # shape = (c_out, c_in, kernel_size * kernel_size)
        weight_g = weight_g.view(eq_weight_shape)

        # no stride arguments provided because padding='same' is not supported for strided convolutions
        output = F.conv2d(input, weight=weight_g, bias=None, padding=self.padding)

        if self.bias is not None:
            bias = self.bias.view(1, self.c_out, 1, 1)
            output = output + bias
        return output

class GCNN(nn.Module):
    """
    Initialization:
      Input dimensions: (batch_size; c_in; H; H)
      Output dimensions: (batch_size; c_out; H'; H')
      filter dimensions: (c_out, c_in, kernel_size, kernel_size)
      symmetry_list: list of equivariant symmetries to be used
    Forward:
      input: data
    """

    def __init__(self, in_channels, num_layers=3, kernel_size=5, padding='same', bias=True, symmetry_list=['rot90']):

        super(GCNN, self).__init__()
        self.c_in = in_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.padding = padding
        self.symmetry_list = symmetry_list
        # rot90 hidden sizes [40, 40, 40]
        # rot90_hflip hidden sizes [44, 44, 44]
        self.conv1 = GConv2d(kernel_size=5, in_channels=in_channels, out_channels=40, symmetry_list=symmetry_list)
        self.conv2 = GConv2d(kernel_size=5, in_channels=40, out_channels=40, symmetry_list=symmetry_list)
        self.conv3 = GConv2d(kernel_size=5, in_channels=40, out_channels=3, symmetry_list=symmetry_list)
        self.fc = nn.Linear(3, 2)


    def forward(self, x):
        x = self.conv1(x)  # dim [batch_size, c_in, h, h]
        x = self.conv2(x)  # dim [batch_size, c_h, h, h]
        x = self.conv3(x)  # dim [batch_size, c_h, h, h]
        x = torch.mean(x, dim=(2, 3))  # dim [batch_size, c_h]
        x = self.fc(x)
        return x


class CNN(nn.Module):
    """
    Initialization:
      Input dimensions: (batch_size; c_in; H; H)
      Output dimensions: (batch_size; c_out; H'; H')
      filter dimensions: (c_out, c_in, kernel_size, kernel_size)
      symmetry_list: list of equivariant symmetries to be used
    Forward:
      input: data
    """

    def __init__(self, in_channels, num_layers=3, kernel_size=5, padding='same'):

        super(CNN, self).__init__()
        self.c_in = in_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.padding = padding
        # rot90 hidden sizes [20, 20, 20]
        # rot90_hflip hidden sizes [20, 20, 20]
        self.conv1 = nn.Conv2d(in_channels=in_channels, kernel_size=5, out_channels=21, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=21, kernel_size=5, out_channels=20, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=20, kernel_size=5, out_channels=3, padding=padding)

    def forward(self, x):
        x = self.conv1(x)  # dim [batch_size, c_in, h, h]
        x = self.conv2(x)  # dim [batch_size, c_h, h, h]
        x = self.conv3(x)  # dim [batch_size, c_h, h, h]
        return x


class EquiCNN(nn.Module):
    """
    Simple group equivariant fully connected neural network
    """
    def __init__(self, in_channels, num_layers=3, kernel_size=5, padding='same', num_classes=2, symmetry_group=None):
        super(EquiCNN, self).__init__()
        self.symmetry_group=symmetry_group
        self.cnn = CNN(in_channels=3)
        self.fc = nn.Linear(3, 2)

    def group_transformed_input(self, x):
        if self.symmetry_group is None:
            return [x]
        elif self.symmetry_group == "rot90":
            hidden_output_list = []  # x dim [batch_size, c, d, d]
            for i in range(4):
                hidden_output_list.append(torch.rot90(x, k=i, dims=[2, 3]))
            return hidden_output_list
        elif self.symmetry_group == "hflip":
            hidden_output_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(hflip(x))  # hflips the last two dimension of x
            return hidden_output_list
        elif self.symmetry_group == "vflip":
            hidden_output_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(vflip(x))  # hflips the last two dimension of x
            return hidden_output_list
        elif self.symmetry_group == "rot90_hflip":
            input_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                input_list.append(hflip(x))  # hflips the last two dimension of x
            hidden_output_list = []
            for i in range(2):
                for j in range(4):
                    hidden_output_list.append(torch.rot90(input_list[i], k=j, dims=[2, 3]))
            return hidden_output_list
        elif self.symmetry_group == "rot90_vflip":
            input_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                input_list.append(vflip(x))  # vflips the last two dimension of x
            hidden_output_list = []
            for i in range(2):
                for j in range(4):
                    hidden_output_list.append(torch.rot90(input_list[i], k=j, dims=[2, 3]))
            return hidden_output_list
        else:
            raise NotImplementedError

    def inverse_group_transformed_hidden_output(self, x_list):
        if self.symmetry_group is None:
            return [x_list]
        elif self.symmetry_group == "rot90":
            hidden_output_list = []  # x dim [batch_size, d, d, c]
            for i in range(4):
                hidden_output_list.append(torch.rot90(x_list[i], k=4-i, dims=[2, 3]))
            return hidden_output_list
        elif self.symmetry_group == "hflip":
            hidden_output_list = [x_list[0]]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(hflip(x_list[i+1]))  # hflips the last two dimension of x
            return hidden_output_list
        elif self.symmetry_group == "vflip":
            hidden_output_list = [x_list[0]]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(vflip(x_list[i+1]))  # hflips the last two dimension of x
            return hidden_output_list
        elif self.symmetry_group == "rot90_hflip":
            hidden_output_list = []
            for i in range(len(x_list)):
                hidden_output_list.append(torch.rot90(x_list[i], k=4-(i % 4), dims=[2, 3]))  # x dim [batch_size, c, d, d]

            for i in range(len(x_list)):
                if i > 3:
                    hidden_output_list[i] = hflip(hidden_output_list[i])  # hflips the last two dimension of x
            return hidden_output_list
        elif self.symmetry_group == "rot90_vflip":
            for i in range(len(x_list)):
                x_list[i] = torch.rot90(x_list[i], k=4-(i % 4), dims=[2, 3])  # x dim [batch_size, c, d, d]

            for i in range(len(x_list)):
                if i > 3:
                    x_list[i] = vflip(x_list[i])  # hflips the last two dimension of x
            return x_list
        else:
            raise NotImplementedError

    def forward(self, x):
        #  transform the inputs
        # x dim [batch_size, channel_dim, H, H]
        x = torch.stack(self.group_transformed_input(x), dim=0)  # [G_dim, batch_size, channel_dim, H, H]
        # [G_dim * batch_size * channel_dim, H * H]
        input_shape = x.shape
        x = x.reshape(-1, input_shape[2], input_shape[3], input_shape[4])
        x = self.cnn(x)  # [G_dim * batch_size, channel_dim, H,  H]
        x = x.reshape(input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4]).contiguous()
        # inverse transform the obtained features

        x = self.inverse_group_transformed_hidden_output(x)

        x = torch.stack(x)  # dim [group_size, batch_size, channels, d, d]
        x = torch.mean(x, dim=(0,))
        x = torch.mean(x, dim=(2, 3))  # dim [batch_size, channels, d, d]
        x_shape = x.shape
        x = x.reshape(x_shape[0], -1)
        x = self.fc(x)  #
        return x


class EquiFNN(nn.Module):
    """
    Simple group equivariant fully connected neural network
    """
    def __init__(self, input_size=224, hidden_sizes=[16, 16, 16], num_classes=2, symmetry_group=None):
        super(EquiFNN, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.symmetry_group = symmetry_group
        self.fc1 = nn.Linear(224 * 224, hidden_sizes[0] * hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0] * hidden_sizes[0], hidden_sizes[1] * hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1] * hidden_sizes[1], hidden_sizes[2] * hidden_sizes[2])
        self.fc4 = nn.Linear(1, num_classes)

    def group_transformed_input(self, x):
        if self.symmetry_group is None:
            return [x]
        elif self.symmetry_group == "rot90":
            hidden_output_list = []  # x dim [batch_size, c, d, d]
            for i in range(4):
                hidden_output_list.append(torch.rot90(x, k=i, dims=[2, 3]))
            return hidden_output_list
        elif self.symmetry_group == "hflip":
            hidden_output_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(hflip(x))  # hflips the last two dimension of x
            return hidden_output_list
        elif self.symmetry_group == "vflip":
            hidden_output_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(vflip(x))  # hflips the last two dimension of x
            return hidden_output_list
        elif self.symmetry_group == "rot90_hflip":
            input_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                input_list.append(hflip(x))  # hflips the last two dimension of x
            hidden_output_list = []
            for i in range(2):
                for j in range(4):
                    hidden_output_list.append(torch.rot90(input_list[i], k=j, dims=[2, 3]))
            return hidden_output_list
        elif self.symmetry_group == "rot90_vflip":
            input_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                input_list.append(vflip(x))  # vflips the last two dimension of x
            hidden_output_list = []
            for i in range(2):
                for j in range(4):
                    hidden_output_list.append(torch.rot90(input_list[i], k=j, dims=[2, 3]))
            return hidden_output_list
        else:
            raise NotImplementedError

    def inverse_group_transformed_hidden_output(self, x_list):
        if self.symmetry_group is None:
            return [x_list]
        elif self.symmetry_group == "rot90":
            hidden_output_list = []  # x dim [batch_size, d, d, c]
            for i in range(4):
                hidden_output_list.append(torch.rot90(x_list[i], k=4-i, dims=[2, 3]))
            return hidden_output_list
        elif self.symmetry_group == "hflip":
            hidden_output_list = [x_list[0]]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(hflip(x_list[i+1]))  # hflips the last two dimension of x
            return hidden_output_list
        elif self.symmetry_group == "vflip":
            hidden_output_list = [x_list[0]]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(vflip(x_list[i+1]))  # hflips the last two dimension of x
            return hidden_output_list
        elif self.symmetry_group == "rot90_hflip":
            hidden_output_list = []
            for i in range(len(x_list)):
                hidden_output_list.append(
                    torch.rot90(x_list[i], k=4 - (i % 4), dims=[2, 3]))  # x dim [batch_size, c, d, d]

            for i in range(len(x_list)):
                if i > 3:
                    hidden_output_list[i] = hflip(hidden_output_list[i])  # hflips the last two dimension of x
            return hidden_output_list
        elif self.symmetry_group == "rot90_vflip":
            for i in range(len(x_list)):
                x_list[i] = torch.rot90(x_list[i], k=4-(i % 4), dims=[2, 3])  # x dim [batch_size, c, d, d]

            for i in range(len(x_list)):
                if i > 3:
                    x_list[i] = vflip(x_list[i])  # hflips the last two dimension of x
            return x_list
        else:
            raise NotImplementedError

    def forward(self, x):
        #  transform the inputs
        # x dim [batch_size, channel_dim, H, H]
        x = torch.stack(self.group_transformed_input(x), dim=0)  # [G_dim, batch_size, channel_dim, H, H]
        # [G_dim * batch_size * channel_dim, H * H]
        input_shape = x.shape
        x = x.reshape(input_shape[0]*input_shape[1]*input_shape[2], input_shape[3]*input_shape[4]).contiguous()

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)  # [G_dim * batch_size * channel_dim, H * H]

        x = x.reshape(input_shape[0], input_shape[1], input_shape[2], self.hidden_sizes[2], self.hidden_sizes[2]).contiguous()

        # inverse transform the obtained features
        x = self.inverse_group_transformed_hidden_output(x)
        x = torch.stack(x)  # dim [group_size, batch_size, channels, d, d]
        x = torch.mean(x, dim=0)
        x = x.reshape(input_shape[1], input_shape[2]*self.hidden_sizes[2]*self.hidden_sizes[2]).contiguous()
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.fc4(x)
        x = x.reshape(input_shape[1], 2).contiguous()
        return x


class GFNN(nn.Module):
    """
    Simple group equivariant fully connected neural network
    """
    def __init__(self, input_size=224, hidden_sizes=[16, 16, 16], num_classes=2, symmetry_group=None):
        super(GFNN, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.symmetry_group = symmetry_group
        # todo: fix input to symmetry list
        self.eq_fc1 = GLinear(input_dim=224*224, output_dim=hidden_sizes[0]*hidden_sizes[0], symmetry_list=["rot90", "hflip"], bias=True)
        self.eq_fc2 = GLinear(input_dim=hidden_sizes[0]*hidden_sizes[0], output_dim=hidden_sizes[1]*hidden_sizes[1], symmetry_list=["rot90", "hflip"], bias=True)
        self.eq_fc3 = GLinear(input_dim=hidden_sizes[1]*hidden_sizes[1], output_dim=hidden_sizes[2]*hidden_sizes[2], symmetry_list=["rot90", "hflip"], bias=True)
        self.fc4 = nn.Linear(1, num_classes)

    def group_transformed_input(self, x):
        if self.symmetry_group is None:
            return [x]
        elif self.symmetry_group == "rot90":
            hidden_output_list = []  # x dim [batch_size, c, d, d]
            for i in range(4):
                hidden_output_list.append(torch.rot90(x, k=i, dims=[2, 3]))
            return hidden_output_list
        elif self.symmetry_group == "hflip":
            hidden_output_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(hflip(x))  # hflips the last two dimension of x
            return hidden_output_list
        elif self.symmetry_group == "vflip":
            hidden_output_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(vflip(x))  # hflips the last two dimension of x
            return hidden_output_list
        elif self.symmetry_group == "rot90_hflip":
            input_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                input_list.append(hflip(x))  # hflips the last two dimension of x
            hidden_output_list = []
            for i in range(2):
                for j in range(4):
                    hidden_output_list.append(torch.rot90(input_list[i], k=j, dims=[2, 3]))
            return hidden_output_list
        elif self.symmetry_group == "rot90_vflip":
            input_list = [x]  # x dim [batch_size, c, d, d]
            for i in range(1):
                input_list.append(vflip(x))  # vflips the last two dimension of x
            hidden_output_list = []
            for i in range(2):
                for j in range(4):
                    hidden_output_list.append(torch.rot90(input_list[i], k=j, dims=[2, 3]))
            return hidden_output_list
        else:
            raise NotImplementedError

    def inverse_group_transformed_hidden_output(self, x_list):
        if self.symmetry_group is None:
            return [x_list]
        elif self.symmetry_group == "rot90":
            hidden_output_list = []  # x dim [batch_size, d, d, c]
            for i in range(4):
                hidden_output_list.append(torch.rot90(x_list[i], k=4-i, dims=[2, 3]))
            return hidden_output_list
        elif self.symmetry_group == "hflip":
            hidden_output_list = [x_list[0]]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(hflip(x_list[i+1]))  # hflips the last two dimension of x
            return hidden_output_list
        elif self.symmetry_group == "vflip":
            hidden_output_list = [x_list[0]]  # x dim [batch_size, c, d, d]
            for i in range(1):
                hidden_output_list.append(vflip(x_list[i+1]))  # hflips the last two dimension of x
            return hidden_output_list
        elif self.symmetry_group == "rot90_hflip":
            for i in range(len(x_list)):
                x_list[i] = torch.rot90(x_list[i], k=4-(i % 4), dims=[2, 3])  # x dim [batch_size, c, d, d]

            for i in range(len(x_list)):
                if i > 3:
                    x_list[i] = hflip(x_list[i])  # hflips the last two dimension of x
            return x_list
        elif self.symmetry_group == "rot90_vflip":
            for i in range(len(x_list)):
                x_list[i] = torch.rot90(x_list[i], k=4-(i % 4), dims=[2, 3])  # x dim [batch_size, c, d, d]

            for i in range(len(x_list)):
                if i > 3:
                    x_list[i] = vflip(x_list[i])  # hflips the last two dimension of x
            return x_list
        else:
            raise NotImplementedError

    def forward(self, x):
        #  transform the inputs
        # x dim [batch_size, channel_dim, H, H]
        x = torch.stack(self.group_transformed_input(x), dim=0)  # [G_dim, batch_size, channel_dim, H, H]
        # [G_dim * batch_size * channel_dim, H * H]
        input_shape = x.shape
        x = x.reshape(input_shape[0] * input_shape[1] * input_shape[2], input_shape[3] * input_shape[4]).contiguous()

        x = self.eq_fc1(x)
        x = self.eq_fc2(x)
        x = self.eq_fc3(x)  # [G_dim * batch_size * channel_dim, H * H]

        x = x.reshape(input_shape[0], input_shape[1], input_shape[2], self.hidden_sizes[2],
                      self.hidden_sizes[2]).contiguous()

        # inverse transform the obtained features
        x = self.inverse_group_transformed_hidden_output(x)
        x = torch.stack(x)  # dim [group_size, batch_size, channels, d, d]
        x = torch.mean(x, dim=0)
        x = x.reshape(input_shape[1], input_shape[2] * self.hidden_sizes[2] * self.hidden_sizes[2]).contiguous()
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.fc4(x)
        x = x.reshape(input_shape[1], 2).contiguous()
        return x



