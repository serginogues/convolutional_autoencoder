"""
ACML Assignment 2: Convolutional Autoencoder (CAE)
CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
"""
from config import TRAIN
from autoencoder import train, test, get_model, model

if __name__ == '__main__':
    if TRAIN:
        train()
    test(get_model())
