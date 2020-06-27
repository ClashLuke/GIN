import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.folder import ImageFolder


class Dataset(ImageFolder):
    """
    Custom dataset yielding original and censored images
    """
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = [transform(sample) for transform in self.transform]
        return sample, target


def get_dataset(dataroot, image_size, erase_scale=5):
    """
    Create pytorch dataset
    :param dataroot: Root of dataset
    :param image_size: Size of images generated
    :param erase_scale: Maximum scale of erasing (from 1:x to x:1)
    :return: Pytorch dataset
    """
    dataroot = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataroot)
    base = [transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]

    erase = base.copy()
    erase.append(transforms.RandomErasing(p=1, ratio=(1 / erase_scale, erase_scale)))

    return Dataset(root=dataroot,
                   transform=(transforms.Compose(erase),
                              transforms.Compose(base)))


def get_dataloader(dataset, batch_size, workers=None):
    """
    Create a data loader from a dataset
    :param dataset: Pytorch dataset
    :param batch_size: Batch size in loaded samples
    :param workers: Number of threads in data loader
    :return: Pytorch data loader
    """
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=workers if workers is not None else
                      torch.get_num_threads(),
                      drop_last=True,
                      )


