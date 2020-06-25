import os
import time

import torch
from torch import Tensor
from torch.nn import ModuleList
from torch.optim import AdamW as Optimizer

from block import ZeroPad, linear_dilated_model
from dataset import check_mkdir, get_dataloader, get_dataset
from image import plot_images
from loss import distance
import random


class Module(torch.nn.Module):
    def __init__(self, classes, overhead=0, depth=8):
        super(Module, self).__init__()
        self.classes = classes
        classes += overhead
        self.classes_overhead = classes
        layers = linear_dilated_model(classes, classes, depth=depth, wrap=False, revnet=True)
        layers.insert(0, ZeroPad(1, overhead))
        self.module_list = ModuleList(layers)
        self.output_pad = ZeroPad(1, overhead)

    def forward(self, image_tensor: Tensor) -> Tensor:
        for module in self.module_list:
            image_tensor = module(image_tensor)
        return image_tensor[:, 0:self.classes, :, :]

    def inverse(self, output: Tensor) -> Tensor:
        if output.size(1) != self.classes_overhead:
            output = self.output_pad(output, 2*random.random()-0.5)
        for module in self.module_list[::-1]:
            output = module.inverse(output)
        return output[:, 0:3, :, :]


class Model:
    def __init__(self, classes=3, overhead=29, depth=8, learning_rate=1e-4, betas=(0.5, 0.9), input_folder='data',
                 image_size=32, batch_size=256, output_folder='Output'):
        self.module = Module(classes, overhead, depth)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.module.to(self.device)

        def _init(module: torch.nn.Module):
            if hasattr(module, "weight") and hasattr(module.weight, "data"):
                if "norm" in module.__class__.__name__.lower() or (
                        hasattr(module, "__str__") and "norm" in str(module).lower()):
                    torch.nn.init.uniform_(module.weight.data, 0.998, 1.002)
                else:
                    torch.nn.init.orthogonal_(module.weight.data)
            if hasattr(module, "bias") and hasattr(module.bias, "data"):
                torch.nn.init.constant_(module.bias.data, 0)

        self.module.apply(_init)

        self.opt = Optimizer(self.module.parameters(),
                             lr=learning_rate,
                             betas=betas)
        self.dataset = get_dataloader(get_dataset(input_folder, image_size), batch_size)
        self.output_folder = output_folder
        self.decensored_folder = os.path.join(output_folder, 'Decensored')
        self.recensored_folder = os.path.join(output_folder, 'Recensored')
        any(map(check_mkdir, (output_folder, self.decensored_folder, self.recensored_folder)))

    def __str__(self):
        return str(self.module)

    def fit(self, epochs=None, print_interval=16, plot_interval=64):
        epoch = 0
        while epochs is None or epoch < epochs:
            epoch += 1
            start_time = time.time()
            item_count = str(len(self.dataset))
            item_count_len = len(item_count)
            for idx, ((censored, original), _) in enumerate(self.dataset, 1):
                censored = censored.to(self.device)
                original = original.to(self.device)
                censored_clone = censored.clone()
                original_clone = original.clone()

                recensored_out = self.module(original)
                forward_loss = distance(recensored_out, censored)
                forward_loss.backward()
                self.opt.step()

                decensored_out = self.module.inverse(censored_clone)
                inverse_loss = distance(decensored_out, original_clone)
                inverse_loss.backward()
                self.opt.step()

                if idx % plot_interval == 0:
                    plot_images(self.decensored_folder, decensored_out.detach(), epoch, idx)
                    plot_images(self.recensored_folder, recensored_out.detach(), epoch, idx)
                if idx % print_interval == 0:
                    print(f'\r[{epoch}][{idx:{item_count_len}d}/{item_count}] '
                          f'CensorLoss: {forward_loss.item():.5f} '
                          f'- DeCensorLoss: {inverse_loss.item() / 2:.5f} '
                          f'| {idx / (time.time() - start_time):.2f} Batch/s',
                          end='')
