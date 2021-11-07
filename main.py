"""
ACML Assignment 2: Convolutional Autoencoder (CAE)
CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
"""
from config import *
from autoencoder import *
import cv2

if __name__ == '__main__':
    class CAE4(nn.Module):
        def __init__(self):
            super(CAE4, self).__init__()

            channels = [8, 16, 32, 64, 128, 32, 8]
            padding = 1
            stride = 1
            kernel = 3

            # Encoder
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=channels[0], kernel_size=(3, 3), padding=padding,
                                   stride=stride)
            self.conv2 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=(3, 3),
                                   padding=padding, stride=stride)
            self.conv3 = nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=(3, 3),
                                   padding=padding, stride=stride)
            self.conv4 = nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=(3, 3),
                                   padding=padding, stride=stride)
            self.conv5 = nn.Conv2d(in_channels=channels[3], out_channels=channels[4], kernel_size=(3, 3),
                                   padding=padding, stride=stride)
            self.maxpool = nn.MaxPool2d(kernel_size=kernel - 1, stride=stride, padding=0)

            # Decoder
            self.t_conv1 = nn.ConvTranspose2d(in_channels=channels[4], out_channels=channels[5], kernel_size=(5, 5),
                                              padding=padding, stride=stride)
            self.t_conv2 = nn.ConvTranspose2d(in_channels=channels[5], out_channels=channels[6], kernel_size=(4, 4),
                                              padding=padding, stride=stride)
            self.t_conv3 = nn.ConvTranspose2d(in_channels=channels[6], out_channels=2, kernel_size=(4, 4),
                                              padding=padding, stride=stride)
            self.print_latent_shape = 0

        def forward(self, x):
            # encoder
            x = F.relu(self.conv1(x))
            x = self.maxpool(x)
            x = F.relu(self.conv2(x))
            x = self.maxpool(x)
            x = F.relu(self.conv3(x))
            x = self.maxpool(x)
            x = F.relu(self.conv4(x))
            x = self.maxpool(x)
            x = F.relu(self.conv5(x))
            if self.print_latent_shape == 0:
                print("Latent space shape: " + str(x.shape))
                self.print_latent_shape += 1

            # decoder
            x = F.relu(self.t_conv1(x))
            x = F.relu(self.t_conv2(x))
            y = F.sigmoid(self.t_conv3(x))
            return y

    model_colorization = CAE4().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model_colorization.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    EPOCHS = 70

    from torchvision import transforms
    def train4(model, save):
        loss_history = []
        running_loss = 0.0
        for i in range(EPOCHS):
            # TRAIN MODEL
            loss_sum = 0
            n = 0
            for j, data in enumerate(train_loader, 0):
                n = j
                # get the training data
                imgs, _ = data

                # rgb to yuv
                luminance = torch.zeros((BATCH_SIZE, 1, 32, 32))
                chrominance = torch.zeros((BATCH_SIZE, 2, 32, 32))
                for idx in range(BATCH_SIZE):
                    img = imgs[idx].numpy()
                    yuv_image = cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_BGR2YUV)
                    luminance[idx] = transforms.ToTensor()(yuv_image[:, :, 0])
                    chrominance[idx] = transforms.ToTensor()(yuv_image[:, :, 1:])

                luminance = luminance.float().to(device)
                chrominance = chrominance.float().to(device)

                # Before the backward pass, set gradients to zero
                optimizer.zero_grad()

                # predict
                #input = images_yuv_tensor.float().to(device)
                output = model.forward(luminance)  # chrominance
                # compute loss
                loss = criterion(output, chrominance)
                loss_sum += round(float(loss.item()), 4)

                # backpropagate loss error
                loss.backward()

                # optimize with backprop
                optimizer.step()
                del data, imgs

            scheduler.step()

            # region print current loss
            loss_epoch = loss_sum / n
            loss_history.append(loss_epoch)
            print("Epoch " + str(i) + ", Loss = " + str(loss_epoch))

            if i > 1 and loss_history[-1] < loss_history[-2]:
                # SAVE THE MODEL
                print("model saved")
                torch.save(model.state_dict(), save)

            if (i > 2 and loss_epoch > loss_history[-2]) or (i > 15 and (loss_history[1] - loss_epoch) < 0.02):
                break

        print("Training finished")

        # PLOT ACCURACY
        plt.plot(loss_history)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.title('Training Loss per epoch')
        plt.show()


    # %%
    train4(model_colorization, " ")
