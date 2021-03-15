from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def lr_lambda(e):
    if e < 60:
        return 1

    elif 60 <= e < 120:
        return 0.2

    elif 120 <= e < 160:
        return 0.04

    else:
        return 0.008


def supervised(optimizer: Optimizer, **kwargs) -> list:
    lr_scheduler = LambdaLR(optimizer, lr_lambda)
    return [lr_scheduler]


def dct(optimizer: Optimizer, **kwargs) -> list:
    lr_scheduler = LambdaLR(optimizer, lr_lambda)
    return [lr_scheduler]


def dct_uniloss(optimizer: Optimizer) -> list:
    lr_scheduler = LambdaLR(optimizer, lr_lambda)
    return [lr_scheduler]


def mean_teacher(optimizer: Optimizer) -> list:
    lr_scheduler = LambdaLR(optimizer, lr_lambda)
    return [lr_scheduler]


def fixmatch(optimizer: Optimizer) -> list:
    lr_scheduler = LambdaLR(optimizer, lr_lambda)
    return [lr_scheduler]
