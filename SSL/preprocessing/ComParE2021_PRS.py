from typing import Tuple
from torch.nn import Module
from torch.nn import Sequential
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from SSL.util.transforms import PadUpTo, Squeeze
from SSL.util.augments import create_composer, augmentation_factory

aug_pool = augmentation_factory('weak', ratio=0.5, sr=16000)

spec_transforms = Sequential(
    PadUpTo(target_length=16000 * 3, mode="constant", value=0),
    MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=512, n_mels=64),
    AmplitudeToDB(),
    Squeeze(0),
)


def supervised(use_augmentation: bool = False) -> Tuple[Module, Module]:
    train_transform = create_composer(use_augmentation, aug_pool, spec_transforms)
    val_transform = create_composer(False, aug_pool, spec_transforms)

    return train_transform, val_transform


def dct(use_augmentation: bool = False) -> Tuple[Module, Module]:
    return supervised(use_augmentation)


def dct_uniloss(use_augmentation: bool = False) -> Tuple[Module, Module]:
    return supervised(use_augmentation)


def dct_aug4adv(use_augmentation: bool = False) -> Tuple[Module, Module]:
    raise NotImplementedError


def mean_teacher(use_augmentation: bool = False) -> Tuple[Module, Module]:
    return supervised(use_augmentation)


def fixmatch(use_augmentation: bool = False) -> Tuple[Module, Module]:
    return supervised(use_augmentation)
