"""
ACML Assignment 2: Convolutional Autoencoder (CAE)

CAE features:
    - The output is evaluated by comparing the reconstructed image by the original one
    - CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html

Tasks:
1) Reconstruction with Deep Learning library  (PyTorch)
    - Goal: Reconstruct any CIFAR-10 image
    - Cost function: MSE or binary cross-entropy between input and reconstructed image
    - Activation for conv layers: ReLU
    - Architecture:
        Input: 32x32x3 (colored image)
        Encoder:
            Layer 1: Filter: 3x3 and 8 channels -> Max pooling 2x2
            Layer 2: Filter: 3x3 + Channels: 12 -> Max pooling 2x2
            Layer 3: 


"""


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
