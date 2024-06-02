# import cv2
# import torch
# import numpy as np

from torchvision import transforms


# class GaussianBlur(object):
#     # Implements Gaussian blur as described in the SimCLR paper
#     def __init__(self, kernel_size, min=0.1, max=2.0):
#         self.min = min
#         self.max = max
#         # kernel size is set to be 10% of the image height/width
#         self.kernel_size = kernel_size
#
#     def __call__(self, sample):
#         sample = np.array(sample)
#
#         # blur the image with a 50% chance
#         prob = np.random.random_sample()
#
#         if prob < 0.5:
#             sigma = (self.max - self.min) * np.random.random_sample() + self.min
#             sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
#
#         return torch.FloatTensor(sample)


train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=int(0.1 * 32), sigma=(0.1, 2.0))],
            p=0.5,
        ),
        # GaussianBlur(kernel_size=int(0.1 * 32)),
        # transforms.ToTensor(),        # Removed it as it permuted the order, instead move it to the stop
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ]
)
