"""
ACML Assignment 2: Convolutional Autoencoder (CAE)

CAE features:
    - The output is evaluated by comparing the reconstructed image by the original one
    - CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html

Tasks:
1) Reconstruction (use DL library (PyTorch))
    exc 1) Divide your dataset into training (80%), validation (10%) and test (10%). Normalize the data.
    exc 2) Implement the autoencoder network specified above.
        Run the training for at least 10 epochs, and plot the evolution of the error with epochs.
        Report also the test error.
    exc 3) What is the size of the latent space representation of the above network?
        For instance, the first convolutional layer has an input volume of 32 × 32 (W = 32), a kernel size of 3 × 3 (K = 3), a
        padding (P) of 1, a stride (S) of 1 and 8 channels (C). Therefore, the size of the first convolutional layer representation
        can be calculated as follows:
        size = C * (((W - K - 2P)/S) + 1)**2 = 8 * ((32 − 3 + 2 * 1)/1 + 1)**2 = 8192
    exc 4) Try other architectures (e.g. fewer intermediate layers, different number of channels, filter sizes or stride and
        padding configurations) to answer questions such as: What is the impact of those in the reconstruction error
        after training? Is there an obvious correlation between the size of the latent space representation and the error?

2) Colorization
    exc 1) Adapt your network from the previous part such that it learns to reconstruct colors by feeding in grayscale
        images but predicting all RGB channels. As a starting point, use the hyperparameters (including the network
        architecture) that you identified to yield the best performance in Exercise 3.2
    exc 2) Report on your results and reason about potential shortcomings of your network. What aspects of the architecture/hyperparameters/optimization could be improved upon to fit the model more adequately to this
        application? Try out some ideas.

        (Hint) A neat trick is to not predict the full color image, but only its chrominance - the proportion of the image determining the colors but not the luminance. By predicting the chrominance, we relieve the model of also reconstructing
        the details (such as contours) that we already have in the grayscale image. The predicted chrominance can then be
        merged with the luminance captured in grayscale to reconstruct the full image.

"""
from config import TRAIN
from autoencoder import train, test, get_model, model

if __name__ == '__main__':
    if TRAIN:
        train()
    test(get_model())
