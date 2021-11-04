import torch
import matplotlib.pyplot as plt
from torchvision import transforms


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
    print(mean)
    print(std)
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


def show_train_sample(dataset):
    # show samples
    img, label = dataset[120]
    print("image label:", dataset.classes[label])

    # since we already used the transform, type(img) = torch.Tensor
    print(img.shape)

    # plot with original axis before converting PIL Image to Tensor, otherwise an Exception arises
    # C × H × W to H × W × C
    plt.imshow(img.permute(1, 2, 0))
    plt.show()