from typing import Tuple
import torch
from torch.nn import Module
from torch.nn import Sequential
from SSL.util.transforms import PadUpTo
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


class Squeeze(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


commun_transforms = Sequential(
    PadUpTo(target_length=48000, mode="constant", value=0),
    MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=512, n_mels=64),
    AmplitudeToDB(),
    Squeeze(0),
)


def supervised() -> Tuple[Module, Module]:
    train_transform = commun_transforms
    val_transform = commun_transforms

    return train_transform, val_transform


def dct() -> Tuple[Module, Module]:
    return supervised()


def dct_uniloss() -> Tuple[Module, Module]:
    return supervised()


def dct_aug4adv() -> Tuple[Module, Module]:
    raise NotImplementedError


def mean_teacher() -> Tuple[Module, Module]:
    return supervised()


def fixmatch() -> Tuple[Module, Module]:
    return supervised()