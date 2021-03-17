
from torch.optim import Optimizer, Adam


def supervised(model,
               lr: float = 0.001,
               betas=(0.9, 0.999),
               eps=1e-08,
               weight_decay=0.,
               amsgrad=True,
               **kwargs) -> Optimizer:

    return Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)


def dct(model1, model2, learning_rate: float = 0.001, **kwargs) -> Optimizer:
    raise NotImplementedError


def dct_uniloss(model1, model2, learning_rate: float = 0.001, **kwargs) -> Optimizer:
    raise NotImplementedError


def mean_teacher(student, learning_rate: float = 0.003, **kwargs) -> Optimizer:
    raise NotImplementedError


def fixmatch(model, **kwargs):
    return supervised(model, **kwargs)
