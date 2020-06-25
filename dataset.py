import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.folder import ImageFolder


class Dataset(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = [transform(sample) for transform in self.transform]
        return sample, target


def get_dataset(dataroot, image_size, erase_scale=5):
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
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=workers if workers is not None else
                      torch.get_num_threads(),
                      drop_last=True,
                      )


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
