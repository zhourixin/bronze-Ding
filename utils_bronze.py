import numpy as np
import torch
import random



trees_bronze = [
    [1, 1],
    [2, 1],
    [3, 2],
    [4, 2],
    [5, 2],
    [6, 3],
    [7, 3],
    [8, 3],
    [9, 4],
    [10, 4],
    [11, 4],
]

def get_order_family_target(targets, device, shape_labels):

    order_target_list = []
    target_list_sig = []
    shape_targets_list = []

    for i in range(targets.size(0)):
        
        order_target_list.append(trees_bronze[targets[i]][1]-1)
        target_list_sig.append(int(targets[i])+4)
        shape_targets_list.append(int(shape_labels[i])+15)

    order_target_list = torch.from_numpy(np.array(order_target_list)).to(device)
    target_list_sig = torch.from_numpy(np.array(target_list_sig)).to(device)
    shape_targets_list = torch.from_numpy(np.array(shape_targets_list)).to(device)

    return order_target_list, target_list_sig, shape_targets_list

def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 448 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws