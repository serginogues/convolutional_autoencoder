"""
Convolutional Autoencoder

1) Goal: Reconstruct any CIFAR-10 image
- Cost function: MSE or binary cross-entropy between input and reconstructed image
- Activation for conv layers: ReLU
- Architecture:
    Input: 32x32x3 (colored image)
    Encoder:
        Layer 1: Filter: 3x3 and 8 channels -> Max pooling 2x2
        Layer 2: Filter: 3x3 + Channels: 12 -> Max pooling 2x2
    Decoder:
        Layer 3: Filter: 3x3 and 16 channels -> Upsampling 2x2
        Layer 4: Filter: 3x3 and 12 channels -> Upsampling 2x2
        Layer 5: Filter: 3x3 and 3 channels

"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from config import *
from dataset import load_CIFAR10

train_loader, test_loader, valid_loader, classes = load_CIFAR10()

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=1, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=1, padding=0)
        self.maxpool_indices = nn.MaxPool2d(kernel_size=(1, 1), stride=1, padding=0, return_indices=True)

        # Decoder
        # self.unpool = nn.MaxUnpool2d(kernel_size=(3, 3), stride=1, padding=0)
        self.t_conv1 = nn.ConvTranspose2d(in_channels=16, out_channels=12, kernel_size=(4, 4), padding=1, stride=1)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=(4, 4), padding=1, stride=1)

    def forward(self, x):
        # x.shape = batch_size x 3 x 32 x 32 where the batch_size = 32

        # conv filter formula: W' = (W - K + 2P)/S + 1
        # maxpool formula e.g. 11x11x32 -> (maxpool 2x2) -> 5x5x32

        # encoder
        x = F.relu(self.conv1(x))  # out = batch_size x 8 x 32 x 32 where 32 = 32 - 3 + 2 + 1
        x = self.maxpool(x)  # out = batch_size x 8 x 31 x 31
        x = F.relu(self.conv2(x))  # out = batch_size x 12 x 31 x 31
        x = self.maxpool(x)  # out = batch_size x 12 x 30 x 30
        x = F.relu(self.conv3(x))  # out = batch_size x 16 x 30 x 30
        #x1, indices = self.maxpool_indices(x)  # out = input to get the indices

        # decoder
        #x = self.unpool(x, indices)
        x = F.relu(self.t_conv1(x))
        #x = self.unpool(x, indices)
        y = F.sigmoid(self.t_conv2(x))

        return y


model = CAE().to(device)
criterion = nn.BCELoss()  # loss function
optimizer = optim.SGD(model.parameters(), lr=LR)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def train():
    loss_history = []
    running_loss = 0.0
    for epoch in range(EPOCHS):
        print('Epoch ', epoch + 1, 'out of ', EPOCHS)

        # TRAIN MODEL
        loss_sum = 0
        n = 0
        for i, data in enumerate(train_loader, 0):
            n = i
            # get the training data
            images, label = data
            images = images.to(device)

            # Before the backward pass, set gradients to zero
            optimizer.zero_grad()

            # predict
            output = model.forward(images)

            # compute loss
            loss = criterion(output, images)
            loss_sum += round(float(loss.item()), 4)

            # backpropagate loss error
            loss.backward()

            # optimize with backprop
            optimizer.step()
            del data, images, label

        # region print current loss
        loss_epoch = loss_sum/n
        loss_history.append(loss_epoch)
        print('Loss = ', loss_epoch)
        # endregion

    # SAVE THE MODEL
    torch.save(model.state_dict(), SAVE_PATH)
    print("Training finished")

    # PLOT ACCURACY
    plt.plot(loss_history)
    plt.title('Loss')
    plt.show()


def get_model():
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()
    model.to(device)
    return model


def test(model):
    with torch.no_grad():
        images, labels = iter(test_loader).next()
        images = images.to(device)

        # Sample outputs
        output = model.forward(images)
        images = images.cpu().numpy()

        output = output.cpu() #.view(BATCH_SIZE, 3, 32, 32)
        output = output.detach().numpy()

        ff, axarr = plt.subplots(2, 5)
        axarr[0,0].imshow(np.transpose(images[0], (1, 2, 0)))
        axarr[0, 0].axis('off')
        axarr[0,1].imshow(np.transpose(images[1], (1, 2, 0)))
        axarr[0, 1].axis('off')
        axarr[0,2].imshow(np.transpose(images[2], (1, 2, 0)))
        axarr[0, 2].axis('off')
        axarr[0,3].imshow(np.transpose(images[3], (1, 2, 0)))
        axarr[0, 3].axis('off')
        axarr[0,4].imshow(np.transpose(images[4], (1, 2, 0)))
        axarr[0, 4].axis('off')
        axarr[1, 0].imshow(np.transpose(output[0], (1, 2, 0)))
        axarr[1, 0].axis('off')
        axarr[1, 1].imshow(np.transpose(output[1], (1, 2, 0)))
        axarr[1, 1].axis('off')
        axarr[1, 2].imshow(np.transpose(output[2], (1, 2, 0)))
        axarr[1, 2].axis('off')
        axarr[1, 3].imshow(np.transpose(output[3], (1, 2, 0)))
        axarr[1, 3].axis('off')
        axarr[1, 4].imshow(np.transpose(output[4], (1, 2, 0)))
        axarr[1, 4].axis('off')

        plt.tight_layout()
        plt.show()
