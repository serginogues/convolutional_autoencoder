"""
Dataset class to work with CIFAR-10

torchvision.transforms.ToTensor:
Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
"""

import torch
from torchvision import datasets
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from config import *


def load_CIFAR10(standarize=False):
    """
    Load CIFAR10 dataset
    :param standarize: if True, data is normalized with mean = 0 and st deviation = 1
    :return: train_loader, test_loader, valid_loader, classes
    """
    print("...loading train and test datasets")
    # transforms.ToTensor() already scales from PIL Images with range [0, 255] to Tensors with range [0, 1]
    train_dataset = datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transforms.ToTensor())
    classes = train_dataset.classes

    if standarize:
        print("...computing mean and std for standarization")
        train_transform = get_transform(train_dataset)
        test_transform = get_transform(test_dataset)

        print("...re-loading train and test normalised datasets")
        train_dataset = datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=test_transform)

    print("...concatenating and splitting")
    concat_dataset = ConcatDataset([train_dataset, test_dataset])
    len_ = len(concat_dataset)
    train_set, test_set, valid_set = random_split(concat_dataset, [round(len_ * TRAIN_SIZE), round(len_ * TEST_SIZE), round(len_ * VALIDATION_SIZE)])

    print("")
    print("Train samples:", len(train_set))
    print("Test samples:", len(test_set))
    print("Validation samples:", len(valid_set))
    total_samp = len(train_set) + len(test_set) + len(valid_set)
    print("Sample distribution: " + str(round((len(train_set) / total_samp) * 100))
          + "% train, " + str(round((len(test_set) / total_samp) * 100)) + "% test, "
          + str(round((len(valid_set) / total_samp) * 100)) + "% validation")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

    if standarize:
        # print mean and std
        print("Mean = ", next(iter(train_loader))[0].mean())
        print("Std = ", next(iter(train_loader))[0].std())

    return train_loader, test_loader, valid_loader, classes


def get_transform(dataset):
    """
    - ToTensor: transform PIL images to Tensors.
    Turns the data into a 32-bit floating-point per channel, scaling the values down from 0.0 to 1.0
    - Normalize: normalize data across the 3 rgb channels
    compute mean and st deviation of each RGB channel and normalize values by doing:
    v'[c] = (v[c] - mean[c])/stdev[c] where c is the channel index

    https://www.manning.com/books/deep-learning-with-pytorch
    'Keeping the data in the same range (0-1 or -1-1) means it’s more likely that neurons have nonzero gradients
    thus, faster learning. Also, normalizing each channel so that it has the same distribution will ensure that
    channel information can be mixed and updated through gradient descent using the same learning rate.'
    The values of mean and stdev must be computed offline.'

    """

    # we can work with the whole dataset because is small (60k samples)
    mean, std = get_mean_std(dataset)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean[0].item(), mean[1].item(), mean[2].item()),
                             (std[0].item(), std[1].item(), std[2].item()))
    ])
    return transform


def get_mean_std(dataset):
    """
    :param dataset: a dataset where samples are tensors (not PIL images!)
    :return:
    """
    all_samples = torch.stack([img_t for img_t, _ in dataset], dim=3)
    # here view(3, -1) keeps the first dimension and merges the rest into 1024 elements. So 3 x 1024 vector
    reshaped = all_samples.view(3, -1)
    mean = reshaped.mean(dim=1)
    std = reshaped.std(dim=1)

    return mean, std


def plot_img_dataloader(dataloader):
    Xs, Ys = iter(dataloader).next()
    images = Xs.numpy()
    images = images / 2 + 0.5
    plt.imshow(np.transpose(images[0], (1, 2, 0)))
    plt.show()


def plot_img_dataset(dataset, idx=120):
    """
    print label and plot image
    :param idx: sample index
    :param dataset: torch.utils.data.Dataset object
    """
    # show samples
    img, label = dataset[idx]
    print("image label:", dataset.classes[label])

    # since we already used the transform, type(img) = torch.Tensor
    print(img.shape)

    # plot with original axis before converting PIL Image to Tensor, otherwise an Exception arises
    # C × H × W to H × W × C
    plt.imshow(img.permute(1, 2, 0))
    plt.show()