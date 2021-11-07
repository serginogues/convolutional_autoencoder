"""
ACML Assignment 2: Convolutional Autoencoder (CAE)
CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
"""
from config import *
from autoencoder import *
import cv2

if __name__ == '__main__':
    images, _ = iter(test_loader).next()
    images_plot = images.numpy()


    class CAE_colorization(nn.Module):
        def __init__(self):
            super(CAE_colorization, self).__init__()

            channels = [8, 12, 16, 20, 16]

            # Encoder
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=channels[0], kernel_size=(3, 3), padding=1, stride=1)
            self.conv2 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=(3, 3), padding=1,
                                   stride=1)
            self.conv3 = nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=(3, 3), padding=1,
                                   stride=1)
            self.conv4 = nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=(3, 3), padding=1,
                                   stride=1)
            self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=1, padding=0)

            # Decoder
            self.t_conv1 = nn.ConvTranspose2d(in_channels=channels[3], out_channels=channels[4], kernel_size=(4, 4),
                                              padding=1, stride=1)
            self.t_conv2 = nn.ConvTranspose2d(in_channels=channels[4], out_channels=channels[1], kernel_size=(4, 4),
                                              padding=1, stride=1)
            self.t_conv3 = nn.ConvTranspose2d(in_channels=channels[1], out_channels=channels[0], kernel_size=(3, 3),
                                              padding=1, stride=1)
            self.t_conv4 = nn.ConvTranspose2d(in_channels=channels[0], out_channels=2, kernel_size=(3, 3), padding=1,
                                              stride=1)
            self.print_latent_shape = 0

        def forward(self, x):
            # encoder
            x1 = F.relu(self.conv1(x))

            x2 = self.maxpool(x1)
            x3 = F.relu(self.conv2(x2))
            x4 = self.maxpool(x3)
            x5 = F.relu(self.conv3(x4))
            x5 = F.relu(self.conv4(x5))

            if self.print_latent_shape == 0:
                print(str(x1.shape) + " -> ")
                print(str(x2.shape) + " -> ")
                print(str(x3.shape) + " -> ")
                print(str(x4.shape) + " -> ")
                print("Latent space shape = " + str(x5.shape))
                self.print_latent_shape += 1

            # decoder
            x6 = F.relu(self.t_conv1(x5))
            x6 = F.relu(self.t_conv2(x6))
            x6 = F.relu(self.t_conv3(x6))
            y = F.sigmoid(self.t_conv4(x6))

            if self.print_latent_shape == 0:
                print(str(x6.shape) + " -> " + str(y.shape))
                self.print_latent_shape += 1

            return y

    # model = CAE_colorization(channels=[5, 12, 16, 12])
    model_colorization = CAE_colorization()
    model_colorization.load_state_dict(torch.load('models/cae3.pth'))
    model.eval()
    img = images[50].numpy()
    img = np.transpose(img, (1, 2, 0))
    yuv_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # %%
    luminance_image = torch.tensor(np.reshape(yuv_image[:, :, 0], (1, 1, 32, 32)))
    # chr_image = yuv_image[:,:,1:]
    chr_image = model.forward(luminance_image)
    chr_image = np.reshape(chr_image.detach().numpy(), (32, 32, 2))
    yuv_reconstructed = cv2.merge((yuv_image[:, :, 0], chr_image[:, :, 0], chr_image[:, :, 1]))
    # %%
    reconstructed = cv2.cvtColor(yuv_reconstructed, cv2.COLOR_YUV2BGR)
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original RGB')
    plt.subplot(1, 3, 2)
    plt.imshow(yuv_image[:, :, 0], cmap="gray")
    plt.title('Original Gray')
    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed)
    plt.title('Colorized')
    plt.show()
