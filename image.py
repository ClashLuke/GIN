import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np


def prepare_plot():
    plt.clf()
    plt.axis('off')


def plot_images(folder, images, epoch, index):
    prepare_plot()
    plt.imsave(f'{folder}/{epoch}_{index}.png',
               np.transpose(vutils.make_grid(images,
                                             padding=2,
                                             normalize=True).cpu().numpy(),
                            (1, 2, 0)))
