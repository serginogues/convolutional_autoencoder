"""
Dataset class to work with CIFAR-10
sample dim: C × H × W (channel, height, width)
"""

import torch
from torchvision import datasets
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import get_transform
from config import *


def load_CIFAR10():
    """
    Load (or download) train and test CIFAR10 dataset and split train into 10% validation, 90% train

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4942, 0.4851, 0.4504),
                             (0.2467, 0.2429, 0.2616))
    ])
    """

    print("...loading train and test datasets")
    train_dataset = datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transforms.ToTensor())

    train_transform = get_transform(train_dataset)
    test_transform = get_transform(test_dataset)

    train_dataset = datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=test_transform)

    print("...concatenating test and train. Splitting into 80, 10, 10")
    concat_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    len_ = len(concat_dataset)
    train_set, test_set, valid_set = random_split(concat_dataset, [round(len_ * TRAIN_SIZE), round(len_ * TEST_SIZE), round(len_ * VALIDATION_SIZE)])

    print("Train samples:", len(train_set))
    print("Test samples:", len(test_set))
    print("Validation samples:", len(valid_set))
    total_samp = len(train_set) + len(test_set) + len(valid_set)
    print("Sample distribution: " + str(round((len(train_set) / total_samp) * 100))
          + "% train, " + str(round((len(test_set) / total_samp) * 100)) + "% test, "
          + str(round((len(valid_set) / total_samp) * 100)) + "% validation, ")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=1, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)

    # print mean and std
    """print(next(iter(train_loader))[0].mean())
    print(next(iter(train_loader))[0].std())
    print(next(iter(test_loader))[0].mean())
    print(next(iter(test_loader))[0].std())
    print(next(iter(valid_loader))[0].mean())
    print(next(iter(valid_loader))[0].std())"""

    return train_loader, test_loader, valid_loader