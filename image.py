import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils


def plot_images(folder: str, images: torch.Tensor, epoch: int, index: int):
    """
    Plot images
    :param folder: Name of the folder images will be saved in
    :param images: Pytorch tensor containing images
    :param epoch: Current epoch index
    :param index: Current batch index
    :return: None
    """
    plt.clf()
    plt.axis('off')
    plt.imsave(f'{folder}/{epoch}_{index}.png',
               np.transpose(vutils.make_grid(images,
                                             padding=2,
                                             normalize=True).cpu().numpy(),
                            (1, 2, 0)))
