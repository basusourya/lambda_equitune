import torch.nn as nn
import time

from eq_utils import *
from modules import GLinear
from torchvision.transforms.functional import hflip, vflip
from weight_models import WeightNet

# same as models, but processing done parallely


def entropy(x, dim=-1):
    # x dim [a, b, c]
    p = torch.nn.functional.softmax(x, dim=-1)  # dim [a, b, c]
    log2p = torch.log2(p)  # dim [a, b, c]
    out = torch.sum(-p * log2p, dim=-1, keepdim=False)  # dim [a, b]
    return out

def max_prob(x, dim=-1):
    # x dim [a, b, c]
    p = torch.nn.functional.softmax(x, dim=-1)  # dim [a, b, c]
    log2p = torch.log2(p)  # dim [a, b, c]
    # out = torch.sum(-p * log2p, dim=-1, keepdim=False)  # dim [a, b]
    out, out_indices = torch.max(p, dim=-1)  # dim [a, b], [a, b]
    return out



class EquiCNN(nn.Module):
    """
    Equituned CNN
    """
    def __init__(self, pre_model, input_size=224, feat_size=512, num_classes=2, symmetry_group=None,
                 model_name="alexnet", eval_type="equi0"):
        super(EquiCNN, self).__init__()
        self.input_size = input_size
        self.feat_size = feat_size
        self.num_classes = num_classes
        self.eval_type = eval_type
        self.symmetry_group = symmetry_group
        self.model_name = model_name
        self.pre_model_features = list(pre_model.children())[:-1]  # requires grad set to false already
        self.fc2 = nn.Linear(feat_size, num_classes)

        # initialize equivariant networks to take into account the different output feature sizes for different networks
        if self.model_name == "alexnet":
            self.fc1 = nn.Linear(6*6, 1)
        elif self.model_name == "vgg":
            self.fc1 = nn.Linear(7*7, 1)
        elif self.model_name == "densenet":
            self.fc1 = nn.Linear(7*7, 1)
        elif self.model_name == "resnet":
            self.fc1 = nn.Linear(1, 1)

    def group_transformed_input(self, x):
        if self.symmetry_group is None or self.symmetry_group == "":
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
        if self.symmetry_group is None or self.symmetry_group == "":
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

    def get_equi0output(self, x):
        # x dim [group_size, batch_size, num_classes]

        ent = entropy(x, dim=-1)  # dim [group_size, batch_size]
        indices = torch.argmin(ent, dim=0)  # dim [batch_size]
        out = torch.stack([x[indices[i], i, :] for i in range(len(indices))])  # dim [batch_size, num_classes]
        return out

    def equituneforward(self, x):
        #  transform the inputs
        x = torch.stack(self.group_transformed_input(x), dim=0)  # [G_dim, batch_size, channel_dim, H, H]
        x_shape = x.shape  # (group_size, batch_size, channel_dim, H, H)
        group_size, batch_size = x_shape[0], x_shape[1]
        x = x.reshape(-1, x_shape[2], x_shape[3], x_shape[4]).contiguous()  # [G_dim * batch_size, channel_dim, H, H]

        # pass through pretrained layer
        for layer in self.pre_model_features:
            x = layer(x)  # dim [group_size * batch_size, channels, d, d]
        hidden_output_shape = x.shape  # dim [group_size * batch_size, channels, d, d]

        # reshape to dim [group_size * batch_size * channels, d * d]
        x = x.reshape(group_size * batch_size * hidden_output_shape[1], hidden_output_shape[2] * hidden_output_shape[3]).contiguous()

        # pass through final linear layers
        x = self.fc1(x)  # dim [group_size * batch_size * channels, 1]
        x = x.reshape(group_size * batch_size, -1).contiguous()  # dim [group_size * batch_size, channels]
        x = self.fc2(x)  # dim [group_size * batch_size, num_classes]

        # average over group dim
        x = x.reshape(group_size, batch_size, -1).contiguous()  # dim [group_size, batch_size, num_classes]
        x = torch.mean(x, dim=0)  # # dim [batch_size, num_classes]
        return x

    def equi0forward(self, x):
        #  transform the inputs
        x = torch.stack(self.group_transformed_input(x), dim=0)  # [G_dim, batch_size, channel_dim, H, H]
        x_shape = x.shape  # (group_size, batch_size, channel_dim, H, H)
        group_size, batch_size = x_shape[0], x_shape[1]
        x = x.reshape(-1, x_shape[2], x_shape[3], x_shape[4]).contiguous()  # [G_dim * batch_size, channel_dim, H, H]

        # pass through pretrained layer
        for layer in self.pre_model_features:
            x = layer(x)  # dim [group_size * batch_size, channels, d, d]
        hidden_output_shape = x.shape  # dim [group_size * batch_size, channels, d, d]

        # reshape to dim [group_size * batch_size * channels, d * d]
        x = x.reshape(group_size * batch_size * hidden_output_shape[1], hidden_output_shape[2] * hidden_output_shape[3]).contiguous()

        # pass through final linear layers
        x = self.fc1(x)  # dim [group_size * batch_size * channels, 1]
        x = x.reshape(group_size * batch_size, -1).contiguous()  # dim [group_size * batch_size, channels]
        x = self.fc2(x)  # dim [group_size * batch_size, num_classes]
        x = x.reshape(group_size, batch_size, -1).contiguous()  # dim [group_size, batch_size, num_classes]

        # compute equizero output
        equi0x = self.get_equi0output(x)  # dim [batch_size, num_classes]

        return equi0x

    def forward(self, x):
        if self.training:
            return self.equituneforward(x)
        else:
            if self.eval_type == "equi0":
                return self.equi0forward(x)
            else:
                return self.equituneforward(x)


class Equi0CNN(nn.Module):
    """
    Equituned CNN
    grad_estimators: choose from ["STE", "Equituning", "Equizero"]
    """
    def __init__(self, pre_model, input_size=224, feat_size=512, num_classes=2, symmetry_group=None, model_name="alexnet",
                 grad_estimator="STE", use_softmax=True, use_e_loss=True, use_ori_equizero=False, use_entropy=True):
        super(Equi0CNN, self).__init__()
        self.input_size = input_size
        self.feat_size = feat_size
        self.num_classes = num_classes
        self.symmetry_group = symmetry_group
        self.grad_estimator = grad_estimator
        self.model_name = model_name
        self.use_softmax = use_softmax
        self.use_e_loss = use_e_loss
        self.use_ori_equizero = use_ori_equizero
        self.use_entropy = use_entropy
        self.pre_model_features = list(pre_model.children())[:-1]  # requires grad set to false already
        self.fc2 = nn.Linear(feat_size, num_classes)
        self.weight_net = WeightNet(model_name=model_name)


        # initialize equivariant networks to take into account the different output feature sizes for different networks
        if self.model_name == "alexnet":
            self.fc1 = nn.Linear(6*6, 1)
        elif self.model_name == "vgg":
            self.fc1 = nn.Linear(7*7, 1)
        elif self.model_name == "densenet":
            self.fc1 = nn.Linear(7*7, 1)
        elif self.model_name == "resnet":
            self.fc1 = nn.Linear(1, 1)

    def group_transformed_input(self, x):
        if self.symmetry_group is None or self.symmetry_group == "":
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
        if self.symmetry_group is None or self.symmetry_group == "":
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

    def get_equi0output(self, x, features):
        # \sum_i l_i * L(M_G_i, y) # MoE
        # x dim [group_size, batch_size, num_classes]

        # ent = entropy(x, dim=-1)  # dim [group_size, batch_size]
        # indices = torch.argmin(ent, dim=0)  # dim [batch_size]
        # out = torch.stack([x[indices[i], i, :] for i in range(len(indices))])  # dim [batch_size, num_classes]

        if self.use_ori_equizero:
            if self.use_entropy:
                ent = entropy(x, dim=-1)  # dim [group_size, batch_size]
            else:
                ent = max_prob(x, dim=-1)
            indices = torch.argmin(ent, dim=0)  # dim [batch_size]
            out = torch.stack([x[indices[i], i, :] for i in range(len(indices))])  # dim [batch_size, num_classes]
            return out 
        else:

            x_shape = x.shape
            features_shape = features.shape  # dim [group_size * batch_size, channels, d, d]

            x = x.reshape(-1, x_shape[-1])
            features = features.reshape(features_shape[0], -1)
            weights = self.weight_net(features)  # dim [group_size * batch_size, 1]
            weights = weights.reshape(x_shape[0], x_shape[1], -1)  # dim [group_size, batch_size, 1]
            if self.use_softmax:
                weights = nn.functional.softmax(weights, dim=0)
            else:
                weights_norm = torch.sum(weights, dim=0, keepdim=True)  # dim [1, batch_size, 1]
                weights /= weights_norm
            weights = weights.reshape(-1, 1)

            if self.use_e_loss:
                return x, weights
            else:
                out = torch.einsum('ij, ik -> ij', x.clone(), weights)  # dim [group_size * batch_size, 1]
                out = out.reshape(x_shape[0], x_shape[1], -1)
                out = torch.sum(out, dim=0)
            return out

    def forward(self, x):
        #  transform the inputs
        if self.use_ori_equizero:
            x = torch.stack(self.group_transformed_input(x), dim=0)  # [G_dim, batch_size, channel_dim, H, H]
            x_shape = x.shape  # (group_size, batch_size, channel_dim, H, H)
            group_size, batch_size = x_shape[0], x_shape[1]
            x = x.reshape(-1, x_shape[2], x_shape[3], x_shape[4]).contiguous()  # [G_dim * batch_size, channel_dim, H, H]

            # pass through pretrained layer
            for layer in self.pre_model_features:
                x = layer(x)  # dim [group_size * batch_size, channels, d, d]
            hidden_output_shape = x.shape  # dim [group_size * batch_size, channels, d, d]

            features = x.clone()  # dim [group_size * batch_size, channels, d, d]
            # reshape to dim [group_size * batch_size * channels, d * d]
            x = x.reshape(group_size * batch_size * hidden_output_shape[1], hidden_output_shape[2] * hidden_output_shape[3]).contiguous()

            # pass through final linear layers
            x = self.fc1(x)  # dim [group_size * batch_size * channels, 1]
            x = x.reshape(group_size * batch_size, -1).contiguous()  # dim [group_size * batch_size, channels]
            x = self.fc2(x)  # dim [group_size * batch_size, num_classes]
            x = x.reshape(group_size, batch_size, -1).contiguous()  # dim [group_size, batch_size, num_classes]

            # compute equizero output
            equi0x = self.get_equi0output(x, features)  # dim [batch_size, num_classes]
        
            if self.grad_estimator == "STE":
                out = x[0] + (equi0x - x[0]).detach()
                return out 
            elif self.grad_estimator == "EquiSTE":
                # average over group dim
                equi_x = torch.mean(x, dim=0)  # # dim [batch_size, num_classes]
                out = equi_x + (equi0x - equi_x).detach()
                return out 
            else:
                raise NotImplementedError
        else:
            x = torch.stack(self.group_transformed_input(x), dim=0)  # [G_dim, batch_size, channel_dim, H, H]
            x_shape = x.shape  # (group_size, batch_size, channel_dim, H, H)
            group_size, batch_size = x_shape[0], x_shape[1]
            x = x.reshape(-1, x_shape[2], x_shape[3], x_shape[4]).contiguous()  # [G_dim * batch_size, channel_dim, H, H]

            # pass through pretrained layer
            for layer in self.pre_model_features:
                x = layer(x)  # dim [group_size * batch_size, channels, d, d]
            hidden_output_shape = x.shape  # dim [group_size * batch_size, channels, d, d]

            features = x.clone()  # dim [group_size * batch_size, channels, d, d]
            # reshape to dim [group_size * batch_size * channels, d * d]
            x = x.reshape(group_size * batch_size * hidden_output_shape[1], hidden_output_shape[2] * hidden_output_shape[3]).contiguous()

            # pass through final linear layers
            x = self.fc1(x)  # dim [group_size * batch_size * channels, 1]
            x = x.reshape(group_size * batch_size, -1).contiguous()  # dim [group_size * batch_size, channels]
            x = self.fc2(x)  # dim [group_size * batch_size, num_classes]
            x = x.reshape(group_size, batch_size, -1).contiguous()  # dim [group_size, batch_size, num_classes]

            # compute equizero output
            equi0x = self.get_equi0output(x, features)  # dim [batch_size, num_classes]
            out = equi0x
        

        return out


class Equi0FTCNN(nn.Module):
    """
    Equi0/Equitune CNN but finetuned on downstream tasks
    model_type: choose from ["equitune", "equi0"]
    grad_estimators: choose from ["STE", "Equituning", "Equizero"]
    """
    def __init__(self, pre_model, num_classes=2, symmetry_group=None, model_type="equi0", grad_estimator="STE"):
        super(Equi0FTCNN, self).__init__()
        self.num_classes = num_classes
        self.symmetry_group = symmetry_group
        self.model_type = model_type
        self.grad_estimator = grad_estimator
        self.pre_model = pre_model  # requires grad set to false already

    def group_transformed_input(self, x):
        if self.symmetry_group is None or self.symmetry_group == "":
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
        if self.symmetry_group is None or self.symmetry_group == "":
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

    def get_equi0output(self, x):
        # x dim [group_size, batch_size, num_classes]
        ent = entropy(x, dim=-1)  # dim [group_size, batch_size]
        # _, indices = torch.mode(ent, dim=0)  # dim [batch_size]
        # indices = torch.argmin(ent, dim=0)  # dim [batch_size]
        indices = torch.argmax(ent, dim=0)  # dim [batch_size]
        out = torch.stack([x[indices[i], i, :] for i in range(len(indices))])  # dim [batch_size, num_classes]
        return out

    def forward(self, x):
        #  transform the inputs
        x = torch.stack(self.group_transformed_input(x), dim=0)  # [G_dim, batch_size, channel_dim, H, H]
        x_shape = x.shape  # (group_size, batch_size, channel_dim, H, H)
        group_size, batch_size = x_shape[0], x_shape[1]
        x = x.reshape(-1, x_shape[2], x_shape[3], x_shape[4]).contiguous()  # [G_dim * batch_size, channel_dim, H, H]

        # pass through pretrained model
        x = self.pre_model(x)  # [G_dim * batch_size, num_classes]
        x = x.reshape(group_size, batch_size, -1).contiguous()  # dim [group_size, batch_size, num_classes]

        # compute equizero output
        equi0x = self.get_equi0output(x)  # dim [batch_size, num_classes]

        if self.model_type == "equi0":
            # equi0 output with different gradient estimators
            if not self.training:
                if self.grad_estimator == "STE":
                    out = x[0] + (equi0x - x[0]).detach()
                elif self.grad_estimator == "EquiSTE":
                    # average over group dim
                    equi_x = torch.mean(x, dim=0)  # dim [batch_size, num_classes]
                    out = equi_x + (equi0x - equi_x).detach()
                else:
                    raise NotImplementedError
            else:
                equi_x = torch.mean(x, dim=0)  # dim [batch_size, num_classes]
                out = equi_x
        elif self.model_type == "equitune":
            # equitune output
            # average over group dim
            equi_x = torch.mean(x, dim=0)  # # dim [batch_size, num_classes]
            out = equi_x
        else:
            raise NotImplementedError

        return out


class EvalEqui0CNN(nn.Module):
    """
    Equituned CNN
    grad_estimators: choose from ["STE", "Equituning", "Equizero"]
    """
    def __init__(self, pre_model, input_size=224, feat_size=512, num_classes=2, symmetry_group=None, model_name="alexnet"):
        super(EvalEqui0CNN, self).__init__()
        self.input_size = input_size
        self.feat_size = feat_size
        self.num_classes = num_classes
        self.symmetry_group = symmetry_group
        self.model_name = model_name
        self.pre_model_features = list(pre_model.children())[:-1]  # requires grad set to false already
        self.fc2 = nn.Linear(feat_size, num_classes)

        # initialize equivariant networks to take into account the different output feature sizes for different networks
        if self.model_name == "alexnet":
            self.fc1 = nn.Linear(6*6, 1)
        elif self.model_name == "vgg":
            self.fc1 = nn.Linear(7*7, 1)
        elif self.model_name == "densenet":
            self.fc1 = nn.Linear(7*7, 1)
        elif self.model_name == "resnet":
            self.fc1 = nn.Linear(1, 1)

    def group_transformed_input(self, x):
        if self.symmetry_group is None or self.symmetry_group == "":
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
        if self.symmetry_group is None or self.symmetry_group == "":
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

    def get_equi0output(self, x):
        # x dim [group_size, batch_size, num_classes]
        ent = entropy(x, dim=-1)  # dim [group_size, batch_size]
        indices = torch.argmin(ent, dim=0)  # dim [batch_size]
        out = torch.stack([x[indices[i], i, :] for i in range(len(indices))])  # dim [batch_size, num_classes]
        return out

    def equi0forward(self, x):
        #  transform the inputs
        x = torch.stack(self.group_transformed_input(x), dim=0)  # [G_dim, batch_size, channel_dim, H, H]
        x_shape = x.shape  # (group_size, batch_size, channel_dim, H, H)
        group_size, batch_size = x_shape[0], x_shape[1]
        x = x.reshape(-1, x_shape[2], x_shape[3], x_shape[4]).contiguous()  # [G_dim * batch_size, channel_dim, H, H]

        # pass through pretrained layer
        for layer in self.pre_model_features:
            x = layer(x)  # dim [group_size * batch_size, channels, d, d]
        hidden_output_shape = x.shape  # dim [group_size * batch_size, channels, d, d]

        # reshape to dim [group_size * batch_size * channels, d * d]
        x = x.reshape(group_size * batch_size * hidden_output_shape[1], hidden_output_shape[2] * hidden_output_shape[3]).contiguous()

        # pass through final linear layers
        x = self.fc1(x)  # dim [group_size * batch_size * channels, 1]
        x = x.reshape(group_size * batch_size, -1).contiguous()  # dim [group_size * batch_size, channels]
        x = self.fc2(x)  # dim [group_size * batch_size, num_classes]
        x = x.reshape(group_size, batch_size, -1).contiguous()  # dim [group_size, batch_size, num_classes]

        # compute equizero output
        equi0x = self.get_equi0output(x)  # dim [batch_size, num_classes]

        return equi0x

    def forward(self, x):
        if not self.training:
            return self.equi0forward(x)
        else:
            # pass through pretrained layer
            for layer in self.pre_model_features:
                x = layer(x)  # dim [batch_size, channels, d, d]
            hidden_output_shape = x.shape  # dim [batch_size, channels, d, d]

            # pass through final linear layers
            batch_size, channel_size = hidden_output_shape[0], hidden_output_shape[1]
            x = x.reshape(batch_size * channel_size, -1).contiguous()  # dim [batch_size * channels, d * d]
            x = self.fc1(x)  # dim [group_size * batch_size * channels, 1]
            x = x.reshape(batch_size, -1).contiguous()  # dim [batch_size, channels]
            x = self.fc2(x)  # dim [batch_size, num_classes]
            x = x.reshape(batch_size, -1).contiguous()  # dim [batch_size, num_classes]
            return x



