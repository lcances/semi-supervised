import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


class ToTensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            t = torch.from_numpy(x)
            return t.float()

        return x.float()


class PadUpTo(nn.Module):
    def __init__(self, target_length, mode: str = "constant", value: int = 0):
        super().__init__()
        self.target_length = target_length
        self.mode = mode
        self.value = value

    def forward(self, x):
        actual_length = x.size()[-1]
        return F.pad(input=x, pad=(0, (self.target_length - actual_length)),
                     mode=self.mode, value=self.value)


class Squeeze(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


class Mean(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim)


class ComposeAugmentation:
    def __init__(self, pre_process_rule: Callable, post_process_rule: Callable, method='pick_one', to_tensor: bool = True):
        self.pre_process = []
        self.process = []
        self.post_process = []
        self.augmentation_pool = []

        self.pre_process_rule = pre_process_rule
        self.post_process_rule = post_process_rule

        self.method = method
        self.to_tensor = to_tensor

    def set_process(self, pool: list) -> None:
        self.process = pool

    def set_augmentation_pool(self, augmentation_pool: list) -> None:
        self.augmentation_pool = augmentation_pool

    def __call__(self, x) -> nn.Sequential:
        self.pre_process = []
        self.post_process = []

        if self.method == 'pick_one':
            tmp_transform = self._compose_pick_one()
            return tmp_transform(x)

        else:
            raise ValueError(f'Methods {self.method} doesn\'t exist.')

    def _compose_pick_one(self) -> nn.Sequential:
        """Select only one augmentation randomly."""
        aug_idx = random.randint(0, len(self.augmentation_pool) - 1)
        selected_aug = self.augmentation_pool[aug_idx]

        # check pre-process rules
        if self.pre_process_rule(selected_aug):
            self.pre_process = [selected_aug]

        # Check post-process rules
        elif self.post_process_rule(selected_aug):
            self.post_process = [selected_aug]

        # Compose the new sequential
        final_transform = [
            nn.Sequential(*self.pre_process),
            self.process,
            nn.Sequential(*self.post_process),
        ]

        # Add convertion to tensor at the beginning if required
        if self.to_tensor:
            final_transform = [ToTensor()] + final_transform

        return nn.Sequential(*final_transform)
