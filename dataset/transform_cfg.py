from __future__ import print_function

import numpy as np
from PIL import Image
import torchvision.transforms as transforms


mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
normalize = transforms.Normalize(mean=mean, std=std)


def unnormalize_transform(mean, std):
    unmean = -np.array(mean) / np.array(std)
    unstd  = 1. / np.array(std)
    return transforms.Normalize(unmean, unstd)


transform_A = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(84, padding=8),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        normalize
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        normalize
    ])
]

transform_A_test = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(84, padding=8),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        normalize
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        normalize
    ])
]


# CUB transform
mean = [0.4856074, 0.4994159, 0.4323767]
std  =  [0.1817421, 0.1811020, 0.1927458]
normalize_cub = transforms.Normalize(mean=mean, std=std)
unnormalize_cub = unnormalize_transform(mean, std)


transform_C = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(84, padding=8),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        normalize_cub
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        normalize_cub
    ])
]

transform_C_test = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(84, padding=8),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        normalize_cub
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        normalize_cub
    ])
]


# CIFAR style transformation
mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]
normalize_cifar100 = transforms.Normalize(mean=mean, std=std)
transform_D = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        normalize_cifar100
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        normalize_cifar100
    ])
]

transform_D_test = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        normalize_cifar100
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        normalize_cifar100
    ])
]


transforms_list = ['A', 'C', 'D']


transforms_options = {
    'A': transform_A,
    'C': transform_C,
    'D': transform_D,
}

transforms_test_options = {
    'A': transform_A_test,
    'C': transform_C_test,
    'D': transform_D_test,
}
