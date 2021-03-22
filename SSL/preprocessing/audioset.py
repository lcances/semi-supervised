from SSL.util.transforms import ComposeAugmentation, ToTensor
from torch.nn import Sequential
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from augmentation_utils.signal_augmentations import Occlusion, SignalAugmentation
from augmentation_utils.spec_augmentations import SpecAugmentation
from mlu.transforms import WaveformTransform, SpectrogramTransform
from mlu.transforms.waveform import StretchPadCrop
from mlu.transforms.spectrogram import CutOutSpec

def supervised():
    return None, None


def dct():
    return None, None


def dct_uniloss():
    return None, None


def dct_aug4adv():
    return None, None


def mean_teacher():
    return None, None


def fixmatch():
    n_mels = 64
    n_time = 500
    sr = 32000
    hop_length = sr * 10 // n_time


    # Weak augmentation
    ratio = 0.5
    weak_aug_pool = [
        Occlusion(ratio, sampling_rate=sr, max_size=0.25),
        CutOutSpec(width_scales=(0.1, 0.5), height_scales=(0.1, 0.5), fill_value=-80.0, p=ratio),
        StretchPadCrop(rates=(0.5, 1.5), align="random", p=ratio),
    ]

    # Strong augmentation
    ratio = 1.0
    strong_aug_pool = [
        Occlusion(ratio, sampling_rate=sr, max_size=0.75),
        CutOutSpec(width_scales=(0.5, 1.0), height_scales=(0.5, 1.0), fill_value=-80.0, p=ratio),
        StretchPadCrop(rates=(0.25, 1.75), align="random", p=ratio),
    ]

    # Transform to spectrogramSpecAugmentation
    spec_transform = Sequential(
        ToTensor(),
        MelSpectrogram(sample_rate=sr, n_fft=2048, hop_length=hop_length, n_mels=n_mels),
        AmplitudeToDB(),
    )
    
    # Composer
    rules = {
        'pre_process_rule': lambda x: isinstance(x, (SignalAugmentation, WaveformTransform)),
        'post_process_rule': lambda x: isinstance(x, (SpecAugmentation, SpectrogramTransform))
    }
    weak_composer = ComposeAugmentation(**rules)
    strong_composer = ComposeAugmentation(**rules)
    
    # Add the main transformation (comvert to log-mel-spectrogram)
    weak_composer.set_process(spec_transform)
    strong_composer.set_process(spec_transform)
    
    # Add the two different augmentation pool
    weak_composer.set_augmentation_pool(weak_aug_pool)
    strong_composer.set_augmentation_pool(strong_aug_pool)


    return (weak_composer, strong_composer), spec_transform