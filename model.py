import random
import time
from typing import Union

import torch
import torch.backends.cudnn
from torch import Tensor
from torch.nn import ModuleList
from torch.optim import AdamW as Optimizer

from block import ConstantPad, linear_dilated_model
from data import DatasetCtx, FolderCtx, ModuleCtx, OptimizerCtx
from dataset import get_dataloader, get_dataset
from image import plot_images
from loss import distance

torch.backends.cudnn.deterministic = True


class InvertibleModule(torch.nn.Module):
    """
    Fully invertible pytorch module with comfort functions to simply the "inverse" pass.
    """
    def __init__(self, module_ctx: ModuleCtx):
        super(InvertibleModule, self).__init__()
        self.classes = module_ctx.classes
        self.classes_overhead = module_ctx.classes + module_ctx.overhead
        layers = linear_dilated_model(self.classes_overhead,
                                      depth=module_ctx.depth,
                                      target_coverage=module_ctx.image_size)
        layers.insert(0, ConstantPad(1, module_ctx.overhead))
        self.module_list = ModuleList(layers)
        self.output_pad = ConstantPad(1, module_ctx.overhead)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Function overwriting pytorch's forward method. Calculates a forward pass through the model (front to back).
        :param input_tensor: Tensor used as input for the model
        :return: Output classesd given by the model
        """
        for module in self.module_list:
            input_tensor = module(input_tensor)
        return input_tensor[:, 0:self.classes, :, :]

    def inverse(self, output: Tensor) -> Tensor:
        """
        Comfort function to pass from a tensor from the back to the front of the model.
        :param output: "output" tensor used as input for the final layer
        :return: "input" that would result in the given "output" tensor
        """
        if output.size(1) != self.classes_overhead:
            output = self.output_pad(output, 2 * random.random() - 0.5)
        for module in self.module_list[::-1]:
            output = module.inverse(output)
        return output[:, 0:3, :, :]


def _call_model(fn_input: torch.Tensor, target_output: torch.Tensor, function: torch.nn.Module.__call__,
                optimizer: torch.optim.AdamW) -> (torch.Tensor, torch.Tensor):
    fn_output = function(fn_input)
    loss = distance(fn_output, target_output)
    loss.backward()
    optimizer.step()
    return fn_output.detach(), loss


def _init(module: torch.nn.Module):
    if hasattr(module, "weight") and hasattr(module.weight, "data"):
        if "norm" in module.__class__.__name__.lower() or (
                hasattr(module, "__str__") and "norm" in str(module).lower()):
            torch.nn.init.uniform_(module.weight.data, 0.998, 1.002)
        else:
            torch.nn.init.orthogonal_(module.weight.data)
    if hasattr(module, "bias") and hasattr(module.bias, "data"):
        torch.nn.init.constant_(module.bias.data, 0)


class Model:
    """
    Model wrapper accepting context objects and providing an easy-to-use API similar to that of well-known keras models.
    """
    def __init__(self,
                 module_ctx=ModuleCtx(),
                 folders=FolderCtx(),
                 dataset_ctx=DatasetCtx(),
                 optimizer_ctx=OptimizerCtx()):
        self.module = InvertibleModule(module_ctx)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.module.to(self.device)
        self.module.apply(_init)

        self.opt: Optimizer = optimizer_ctx.optimizer(self.module.parameters(),
                                                      lr=optimizer_ctx.lr,
                                                      betas=optimizer_ctx.betas)
        self.dataset: iter = get_dataloader(get_dataset(dataset_ctx.input_folder, module_ctx.image_size),
                                            dataset_ctx.batch_size, dataset_ctx.workers)
        self.image_size = module_ctx.image_size
        self.folders = folders

    def __str__(self):
        return str(self.module)

    def fit(self, epochs: Union[int, None] = None, print_interval=16, plot_interval=64, reverse=True):
        """
        Fit (train) network on data loader given at initialization
        :param epochs: Number of epochs to train on
        :param print_interval: Every how many batches to print loss
        :param plot_interval: Every how many batches to plot generated images
        :param reverse: Whether to do a reverse pass through the model
        :return: None
        """
        epoch = 0
        while epochs is None or epoch < epochs:
            epoch += 1
            start_time = time.time()
            item_count = str(len(self.dataset))
            item_count_len = len(item_count)
            for idx, ((censored, original), _) in enumerate(self.dataset, 1):
                censored = censored.to(self.device)
                original = original.to(self.device)
                if reverse:
                    censored_clone = censored.clone()
                    original_clone = original.clone()

                decensored_out, forward_loss = _call_model(original, censored, self.module.__call__, self.opt)
                if reverse:
                    recensored_out, inverse_loss = _call_model(censored_clone, original_clone, self.module.inverse,
                                                               self.opt)

                if idx % plot_interval == 0:
                    plot_images(self.folders.decensor, decensored_out, epoch, idx)
                    if reverse:
                        plot_images(self.folders.recensor, recensored_out, epoch, idx)
                if idx % print_interval == 0:
                    print(f'\r[{epoch}][{idx:{item_count_len}d}/{item_count}] '
                          f'CensorLoss: {forward_loss.item():.5f} '
                          f'- DeCensorLoss: {inverse_loss.item() / 2:.5f} ' if reverse else ''
                          f'| {idx / (time.time() - start_time):.2f} Batch/s',
                          end='')
            print('')
