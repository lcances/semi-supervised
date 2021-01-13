from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Tuple
from .speechcommand import mean_teacher_helper, SpeechCommand10
from .speechcommand import dct_uniloss, dct, supervised


def mean_teacher(
        dataset_root,
        supervised_ratio: float = 0.1,
        batch_size: int = 128,

        train_transform: Module = None,
        val_transform: Module = None,

        **kwargs) -> Tuple[DataLoader, DataLoader]:
    return mean_teacher_helper(SpeechCommand10, **locals())