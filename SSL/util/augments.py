from augmentation_utils.augmentations import SignalAugmentation, SpecAugmentation
from augmentation_utils.signal_augmentations import Occlusion
from mlu.transforms import WaveformTransform, SpectrogramTransform
from mlu.transforms.spectrogram import CutOutSpec
from mlu.transforms.waveform import StretchPadCrop
from SSL.util.transforms import ComposeAugmentation


class WeakPreset:
    @staticmethod
    def Occlusion(ratio, sr):
        return Occlusion(ratio, sampling_rate=sr, max_size=0.25)

    @staticmethod
    def CutOutSpec(ratio):
        return CutOutSpec(width_scales=(0.1, 0.5), height_scales=(0.1, 0.5), fill_value=-80.0, p=ratio),

    @staticmethod
    def StretchPadCrop(ratio):
        return StretchPadCrop(rates=(0.5, 1.5), align="random", p=ratio)


class StrongPreset:
    pass


def augmentation_factory(type: str, ratio: float, sr: int):
    assert type in ['weak', 'strong']
    preset = WeakPreset
    if type == 'weak': preset = WeakPreset
    elif type == 'strong': preset = StrongPreset

    return [
        preset.Occlusion(ratio, sr),
        preset.CutOutSpec(ratio),
        preset.StretchPadCrop(ratio),
    ]


# Define the different augmentation pool

def create_composer(use_aug: bool, aug_pool: list, process) -> ComposeAugmentation:
    if not use_aug:
        return process

    composer = ComposeAugmentation(
        pre_process_rule=lambda x: isinstance(x, (SignalAugmentation, WaveformTransform)),
        post_process_rule=lambda x: isinstance(x, (SpecAugmentation, SpectrogramTransform))
    )
    composer.set_process(process)  # spec transform
    composer.set_augmentation_pool(aug_pool)

    return composer