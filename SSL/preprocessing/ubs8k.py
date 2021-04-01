from typing import Tuple
from torch.nn import Module
from torch.nn import Sequential
from SSL.util.transforms import PadUpTo, Squeeze, Mean
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from SSL.util.augments import create_composer, augmentation_factory

aug_pool = augmentation_factory('weak', ratio=0.5, sr=22050)


# Define the seuquence to transform a waveform into log-mel spectrogram
spec_transforms = Sequential(
    PadUpTo(target_length=22050*4, mode="constant", value=0),
    MelSpectrogram(sample_rate=22050, n_fft=2048, hop_length=512, n_mels=64),
    AmplitudeToDB(),
)


def supervised(use_augmentation: bool = False):
    train_transform = create_composer(use_augmentation, aug_pool, spec_transforms)
    val_transform = create_composer(None, aug_pool, spec_transforms)

    return train_transform, val_transform


def dct(use_augmentation: bool = False):
    return supervised(use_augmentation)


def dct_uniloss(use_augmentation: bool = False):
    return supervised(use_augmentation)


def dct_aug4adv(use_augmentation: bool = False):
    return supervised(use_augmentation)


def mean_teacher(use_augmentation: bool = False):
    return supervised(use_augmentation)


def fixmatch(use_augmentation: bool = False):
    return supervised(use_augmentation)
