import math
import torch
import numpy as np


def squeeze_orbits(orbits):
    """
    Input: orbits with randomly assignmed numbers for each orbit
    Output: orbits with minimum numbers assigned to them
    """
    sorted_orbits, arg_sorted_orbits = np.sort(orbits), np.argsort(orbits)
    rank_orbits = np.argsort(arg_sorted_orbits)
    sorted_ranked_orbits = []

    sorted_orbits = sorted_orbits - min(sorted_orbits)
    sorted_orbits = sorted_orbits.tolist()

    current_rank = 0
    current_orbit = 0
    for i in range(len(sorted_orbits)):
        if sorted_orbits[i] > current_orbit:
            current_orbit = sorted_orbits[i]
            sorted_orbits[i] = current_rank + 1
            current_rank += 1
        else:
            sorted_orbits[i] = current_rank

    orbits_new = [sorted_orbits[rank_orbits[i]] for i in range(len(sorted_orbits))]
    return orbits_new


def get_equivariance_indices(nx, nh, symmetry_list):
    """
    Size of input layer = nx
    Size of hidden layer = nh
    """

    m = int(math.sqrt(nx))  # nx must be of the form m*m
    h = int(math.sqrt(nh))  # nh must be of the form h*h
    I = I_prev = [_ for _ in range(nx * nh)]  # records the permutation of indices of W,
    # which is of size n*n. For no previous symmetry, I_prev is given as the original indices
    V = np.zeros(nx * nh) - 1  # -1 if not visited, else 1
    current_orbit = -1
    indices_queue = []
    indices_queue_pair = []

    if symmetry_list is None:
        return I

    for i in range(nx):
        for j in range(nh):
            i_0 = i
            j_0 = j
            if V[i_0 * nh + j_0] < 0:
                V[i_0 * nh + j_0] = 1
                index = i_0 * nh + j_0
                indices_queue.append(index)
                indices_queue_pair.append([i_0, j_0])
                current_orbit += 1
                while len(indices_queue) > 0:
                    index = indices_queue.pop(0)
                    i_0, j_0 = indices_queue_pair.pop(0)
                    I[index] = current_orbit

                    if 'permutation' in symmetry_list:
                        # permutation equivariance
                        # In case of permutation equivariance, all the other three
                        # equivariances are subset of it, hance return directly
                        if m == h:
                            diagonal = np.ones(m * m)
                            I = torch.tensor(np.diag(diagonal)).long().view(m * m * h * h).tolist()
                        return squeeze_orbits(I)

                    if "rot90" in symmetry_list:
                        # rotations
                        alpha, beta = i_0 // m, i_0 % m
                        gamma, delta = j_0 // h, j_0 % h

                        # rotate by 90
                        alpha_r, beta_r = beta, m - 1 - alpha
                        gamma_r, delta_r = delta, h - 1 - gamma

                        i_r = alpha_r * m + beta_r
                        j_r = gamma_r * h + delta_r

                        index_r = i_r * nh + j_r
                        if V[index_r] < 0:
                            V[index_r] = 1
                            I[index_r] = current_orbit
                            indices_queue.append(index_r)
                            indices_queue_pair.append([i_r, j_r])

                        # rotate by 180
                        alpha_rr, beta_rr = beta_r, m - 1 - alpha_r
                        gamma_rr, delta_rr = delta_r, h - 1 - gamma_r

                        i_rr = alpha_rr * m + beta_rr
                        j_rr = gamma_rr * h + delta_rr

                        index_rr = i_rr * nh + j_rr
                        if V[index_rr] < 0:
                            V[index_rr] = 1
                            I[index_rr] = current_orbit
                            indices_queue.append(index_rr)
                            indices_queue_pair.append([i_rr, j_rr])

                        # rotate by 270
                        alpha_rrr, beta_rrr = beta_rr, m - 1 - alpha_rr
                        gamma_rrr, delta_rrr = delta_rr, h - 1 - gamma_rr

                        i_rrr = alpha_rrr * m + beta_rrr
                        j_rrr = gamma_rrr * h + delta_rrr

                        index_rrr = i_rrr * nh + j_rrr
                        if V[index_rrr] < 0:
                            V[index_rrr] = 1
                            I[index_rrr] = current_orbit
                            indices_queue.append(index_rrr)
                            indices_queue_pair.append([i_rrr, j_rrr])

                    if "hflip" in symmetry_list:
                        # horizontal flip
                        alpha, beta = i_0 // m, i_0 % m
                        gamma, delta = j_0 // h, j_0 % h

                        # hflip
                        alpha_h, beta_h = alpha, m - 1 - beta
                        gamma_h, delta_h = gamma, h - 1 - delta

                        i_h = alpha_h * m + beta_h
                        j_h = gamma_h * h + delta_h

                        index_h = i_h * nh + j_h
                        if V[index_h] < 0:
                            V[index_h] = 1
                            I[index_h] = current_orbit
                            indices_queue.append(index_h)
                            indices_queue_pair.append([i_h, j_h])

                    if "vflip" in symmetry_list:
                        # vertical flip
                        alpha, beta = i_0 // m, i_0 % m
                        gamma, delta = j_0 // h, j_0 % h

                        # vflip
                        alpha_h, beta_h = m - 1 - alpha, beta
                        gamma_h, delta_h = h - 1 - gamma, delta

                        i_h = alpha_h * m + beta_h
                        j_h = gamma_h * h + delta_h

                        index_h = i_h * nh + j_h
                        if V[index_h] < 0:
                            V[index_h] = 1
                            I[index_h] = current_orbit
                            indices_queue.append(index_h)
                            indices_queue_pair.append([i_h, j_h])

    return squeeze_orbits(I)