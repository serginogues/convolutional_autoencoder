"""
Dataset class to work with CIFAR-10

sample dim: C × H × W (channel, height, width)
"""
from config import *

# region Load Train and Test Dataset with ToTensor transform
# load (or download if not already) train and test dataset and store at data_path
train_dataset = datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transforms.ToTensor())

total_samples = len(train_dataset) + len(test_dataset)
print("Sample distribution before splitting: "
      + str(round(len(train_dataset) / total_samples * 100)) + "% train, "
      + str(round(len(test_dataset) / total_samples * 100)) + "% test, 0% validation")


def show_train_sample():
    # show samples
    img, label = train_dataset[120]
    print("image label:", train_dataset.classes[label])

    # since we already used the transform, type(img) = torch.Tensor
    print(img.shape)

    # plot with original axis before converting PIL Image to Tensor, otherwise an Exception arises
    # C × H × W to H × W × C
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

show_train_sample()
# endregion

"""
https://www.manning.com/books/deep-learning-with-pytorch
Write transforms to transform PIL images to Tensors and normalize
    - ToTensor transform turns the data into a 32-bit floating-point per channel,
    scaling the values down from 0.0 to 1.0
    - Normalize the dataset so that each channel has zero mean and unitary standard deviation.

'Keeping the data in the same range (0-1 or -1-1) means it’s more likely that neurons have nonzero gradients
thus, faster learning. Also, normalizing each channel so that it has the same distribution will ensure that 
channel information can be mixed and updated through gradient descent using the same learning rate.'
The values of mean and stdev must be computed offline'
"""

imgs = torch.stack([img_t for img_t, _ in train_dataset], dim=3)

normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)

to_tensor = transforms.ToTensor()

transform_valid = transforms.Compose([
    to_tensor,
    normalize,
])

transform_train = transforms.Compose([
    to_tensor,
    normalize,
])

transform_test = transforms.Compose([
    to_tensor,
    normalize,
])

"""
train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
"""

"""
BALANCING

total = 0
counter = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
for data in trainset:
    Xs, ys = data
    for y in ys:
        counter[int(y)] += 1
        total += 1
for i in counter:
    print(f"{i}: {counter[i] / total * 100}")
    
out:
        0: 98.71666666666667
        1: 112.36666666666666
        2: 99.3
        3: 102.18333333333334
        4: 97.36666666666667
        5: 90.35
        6: 98.63333333333333
        7: 104.41666666666667
        8: 97.51666666666667
        9: 99.15
"""
