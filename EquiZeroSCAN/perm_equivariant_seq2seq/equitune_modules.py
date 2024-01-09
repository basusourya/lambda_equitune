from perm_equivariant_seq2seq.eq_utils import get_equivariance_indices, get_scan_equivariance_indices

import math
import torch
import torch.nn as nn
from torch.nn import Parameter


class EquiSCANLinear(nn.Module):
    """
    Initialization:
      input_dim: output vocab size
      output_dim: output vocab size
      G: output group G
      eq_word_indices: list of equivariant word indices
    """

    def __init__(self, input_dim, output_dim, G, weight_init='random', eq_word_indices=[2, 7], bias=None, group_type='cyclic'):

        super(EquiSCANLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.G = G
        self.eq_word_indices = eq_word_indices
        # get equivariant indices for weight matrix
        self.eq_indices = get_scan_equivariance_indices(in_dim=input_dim, out_dim=output_dim,
                                                        eq_word_indices=eq_word_indices, G=G, group_type=group_type)
        # get equivariant indices for bias
        if group_type != 'permutation':
            self.bias_eq_indices = get_scan_equivariance_indices(in_dim=1, out_dim=output_dim,
                                                            eq_word_indices=eq_word_indices, G=G, group_type=group_type)
        else:
            bias = None


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
