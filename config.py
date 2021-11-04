import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = 'D:/UM/ACML/Assignments/'
VALIDATION_SIZE = 0.1  # percentage of the training set used for validation
TEST_SIZE = 0.1
TRAIN_SIZE = 0.8