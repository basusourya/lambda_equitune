import torch
from operator import itemgetter


def cyclic_group_generator(vocab_size, eq_indices):
    """
    :param vocab_size: size of the vocab
    :param eq_indices: indices of the equivariant words
    :return: a group generator of the required cyclic group consisting of all the equivariant words
    """
    g = {i: i for i in range(vocab_size)}  # generator initialized as id

    group_size = len(eq_indices)
    for i in range(group_size):
        next_group_element = (i + 1) % group_size
        g[eq_indices[i]] = eq_indices[next_group_element]

    g['size'] = group_size  # add length of the group as a value
    return g


def cyclic_group(g, group_size, vocab_size):
    """
    :param g: cyclic group generator
    :param group_size: size of the group
    :return: return a list of elements of a cyclic group
    """
    # add id to the group G
    G = [{i: i for i in range(vocab_size)}]

    for i in range(group_size - 1):
        # apply the generator repeatedly to obtain the entire group
        curr_g = G[-1]
        next_g = {i: g[curr_g[i]] for i in range(vocab_size)}
        G.append(next_g)

    return G


def g_transform_data(data, G, device):
    '''
    :param data: any tensor data of input on which group is applied
    :param G: set of group elements
    :return: list of transformed data for equituning
    '''
    untransformed_data = data[:, 0]
    transformed_data = [data[:, 0]]

    for i in range(len(G)-1):
        curr_g = G[i+1]
        current_data = torch.tensor(itemgetter(*(untransformed_data.tolist()))(curr_g), device=device)
        transformed_data.append(current_data)

    transformed_data = torch.stack(transformed_data)[:, :, None]
    transformed_data.to(device)

    return transformed_data


def g_inv_transform_prob_data(data_list, G):
    '''
    Note: Group actions are on batch_size x |V|, instead of batch_size x 1
    :param data: any tensor data
    :param g: group generator
    :return: list of transformed data for equituning
    '''
    output_data_list = data_list.clone()
    g_indices = []
    for g in G:
        g_index = [g[i] for i in range(len(g))]
        g_indices.append(g_index)

    for i in range(len(data_list)):
        output_data_list[i, :, g_indices[i]] = data_list[i, :, :].clone()

    return output_data_list


def g_transform_prob_data(data_list, G, group_index=1):
    '''
    Note: Group actions are on batch_size x |V|, instead of batch_size x 1
    :param data: any tensor data
    :param g: group generator
    :return: list of transformed data for equituning
    '''
    output_data_list = data_list.clone()
    g_indices = []
    for g in G:
        g_index = [g[i] for i in range(len(g))]
        g_indices.append(g_index)

    for i in range(len(data_list)):
        output_data_list[:, :] = data_list[:, g_indices[group_index]].clone()

    return output_data_list


def min_loss_index(prob_dist, loss_func_name="entropy"):
        """
        :param prob_dist: dim [group_size, max_sentence_len=50, vocab_size]
        :return: return group index to choose from
        """
        import torch.nn as nn
        relu = nn.ReLU()
        if loss_func_name == "top1":
            prob_dist = torch.softmax(prob_dist, dim=2)  # dim [group_size, max_sentence_len=50, vocab_size]
            top1_prob, _ = torch.max(prob_dist, dim=2)  # dim [group_size, max_sentence_len=50]
            top1_prob_mean = torch.mean(top1_prob, dim=1)  # dim [group_size]
            min_loss_index = torch.argmax(top1_prob_mean, dim=0)
        elif loss_func_name == "entropy":
            # dim prob_dist [group_size, max_sentence_size=50, vocab_size=8]
            log_prob_dist = torch.log2(torch.softmax(prob_dist, dim=2))  # compute log(p) dim [group_size, max_sentence_size=50, vocab_size=8]
            loss = -torch.sum(torch.einsum('ijk, ijk->ijk', prob_dist, log_prob_dist), dim=(1,2))  # compute entropy dim [group_size]
            min_loss_index = torch.argmin(loss, dim=0)
        else:
            raise NotImplementedError

        return min_loss_index
