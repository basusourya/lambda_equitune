from eq_utils import get_equivariance_indices

import math
import torch
import torch.nn as nn
from torch.nn import Parameter


class GLinear(nn.Module):
    """
    Initialization:
      Input dimension: m = h1xh1 for some h1
      Output dimension: n = h2xh2 for some h2
      symmetry_list: list of equivariant symmetries to be used
      weight_init: 'pre' for using orbit avg, 'random' for usual random initialization
    Forward:
      input data
      sigma (if available)
      pre (if pretrained data is available)
    """

    def __init__(self, input_dim, output_dim=1, weight_init='random', symmetry_list=['rot'], bias=None):

        super(GLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.symmetry_list = symmetry_list

        self.eq_indices = get_equivariance_indices(nx=self.input_dim, nh=self.output_dim,
                                                   symmetry_list=self.symmetry_list)  # get equivariant indices
        self.bias_eq_indices = get_equivariance_indices(nx=1, nh=self.output_dim,
                                                        symmetry_list=self.symmetry_list)  # get equivariant indices
        self.weight = Parameter(torch.Tensor(len(set(self.eq_indices))))  # trainable parameters


        if bias:
            self.bias = Parameter(torch.Tensor(len(set(self.bias_eq_indices))))
        else:
            self.bias = self.register_parameter('bias', None)

        if weight_init == 'random':
            self.reset_parameters()  # reset parameters
        elif weight_init == 'pre':
            self.orbit_avg_init()  # use orbit averaging for parameter initialization

    def reset_parameters(self):
        n = self.input_dim * self.output_dim
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        weight_eq = self.weight[self.eq_indices]
        weight_eq = weight_eq.view(self.input_dim, self.output_dim)
        output = torch.mm(input, weight_eq)

        if self.bias is not None:
            bias_eq = self.bias[self.bias_eq_indices]
            output = output + bias_eq
        return output
