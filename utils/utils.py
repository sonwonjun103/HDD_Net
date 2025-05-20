import torch
import numpy as np

from scipy.ndimage import gaussian_filter, binary_erosion
from collections import defaultdict

def make_fullvolume(volume, a):
    copy_volume = volume.copy()
    for p in a:
        xy = p[0]
        z = p[1]

        for i in range(len(z)-1):
            copy_volume[z[i]][xy[1]][xy[0]]=1
            if z[i] + 1 == z[i+1]:
                continue
            else:
                for j in range(z[i], z[i+1]):
                    copy_volume[j][xy[1]][xy[0]]=1

    return copy_volume

def find_coordinate(x,y,z):
    xy_dict = defaultdict(list)

    for xi, yi, zi in zip(x,y,z):
        xy_dict[(xi, yi)].append(zi)

    xy_coodinate = [(xy, zs) for xy, zs in xy_dict.items() if len(zs) > 1]

    return xy_coodinate

def thresholding(volume, threshold):
    copy_volume = volume.copy()

    copy_volume[copy_volume >= threshold] = 1
    copy_volume[copy_volume <= threshold] = 0

    return copy_volume

def inpaint(volume):
    full = []
    
    for i in range(volume.shape[0]):
        thresholding_volume = thresholding(volume[i].squeeze(0), 0.5)
        assert np.all((thresholding_volume == 0) | (thresholding_volume == 1)) == 1

        z, y, x = np.where(thresholding_volume == 1)
        a = find_coordinate(x, y, z)
        full_volume = make_fullvolume(thresholding_volume, a)

        full.append(np.expand_dims(full_volume, axis=0))

    return torch.from_numpy(np.array(full))

def get_boundary_map(volume):
    edge=[]

    for i in range(volume.shape[0]):
        filter_data = gaussian_filter(volume[i].astype(np.float32), 1)
        threshold = 0.4

        binary_mask = filter_data > threshold

        eroded_mask = binary_erosion(binary_mask)
        boundary_map = binary_mask.astype(int) - eroded_mask.astype(int)
        edge.append(boundary_map)

    return torch.from_numpy(np.array(edge))


def dice_coefficient(prediction, target):

    intersection = np.sum(prediction * target)
    dice = (2. * intersection) / (np.sum(prediction) + np.sum(target))
    return dice

def iou_coefficient(prediction, target):
    interesction = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou = np.sum(interesction) / np.sum(union)

    return iou