"""
Configuration file
"""
import torch
DATA_PATH = 'D:/UM/ACML/Assignments/'
VALIDATION_SIZE = 0.1  # percentage of the training set used for validation
TRAIN_SIZE = 0.8
TEST_SIZE = 0.1
BATCH_SIZE = 64
EPOCHS = 10
SAVE_PATH = 'models/cae.pth'

TRAIN = True

LR = 0.5

# Compatibility with CUDA and GPU -> remember to move into GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')