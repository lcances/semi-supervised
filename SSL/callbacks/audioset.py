import numpy
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_lr_lambda(nb_epoch):
    def lr_lambda(epoch):
        return (1.0 + numpy.cos((epoch-1)*numpy.pi/nb_epoch)) * 0.5
    return lr_lambda


def supervised(nb_epoch: int, optimizer: Optimizer, **kwargs) -> list:
#     loader = kwargs.get("loader", None)
    
#     nb_step = nb_epoch
#     if loader is not None:
#         nb_step = len(loader) * nb_epoch
        
#     lr_scheduler = LambdaLR(optimizer, get_lr_lambda(nb_step))
#     return [lr_scheduler]
    return []


def dct(nb_epoch: int, optimizer: Optimizer, **kwargs) -> list:
    return supervised(nb_epoch, optimizer, **kwargs)


def dct_uniloss(nb_epoch: int, optimizer: Optimizer, **kwargs) -> list:
    return supervised(nb_epoch, optimizer, **kwargs)


def mean_teacher(nb_epoch: int, optimizer: Optimizer, **kwargs) -> list:
    return supervised(nb_epoch, optimizer, **kwargs)


def fixmatch(nb_epoch: int, optimizer: Optimizer, **kwargs) -> list:
    return supervised(nb_epoch, optimizer, **kwargs)