import os

from torch.optim import AdamW as Optimizer


def check_mkdir(dir_name):
    """
    Check if a directory exists, if not, create it
    :param dir_name: Name of directory to check
    :return: None
    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


class FolderCtx:
    """
    Context object holding the names of relevant folders
    """
    def __init__(self, output_folder='Output', decensored='Decensored', recensored='Recensored'):
        self.output = output_folder
        self.decensor = os.path.join(output_folder, decensored)
        self.recensor = os.path.join(output_folder, recensored)
        any(map(check_mkdir, (self.output, self.decensor, self.recensor)))


class OptimizerCtx:
    """
    Context object holding data for optimizer
    """
    def __init__(self,
                 lr=1e-5,
                 betas=(0.5, 0.9),
                 optimizer=Optimizer):
        self.lr = lr
        self.betas = betas
        self.optimizer = optimizer


class DatasetCtx:
    """
    Context object holding data for dataset
    """
    def __init__(self,
                 input_folder='data',
                 batch_size=256,
                 workers=12):
        self.input_folder = input_folder
        self.batch_size = batch_size
        self.workers = workers


class ModuleCtx:
    """
    Context object holding data required to create a InversibleModule
    """
    def __init__(self,
                 classes=3,
                 overhead=13,
                 depth=8,
                 image_size=32):
        self.classes = classes
        self.overhead = overhead
        self.depth = depth
        self.image_size = image_size
