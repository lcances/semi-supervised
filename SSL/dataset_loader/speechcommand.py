import copy
import torchaudio
import random
import tqdm
import os
import torch
import numpy as np
from torch.nn import Module
from torch import Tensor
from SSL.util.utils import DotDict, ZipCycle, get_datetime
import torch.utils.data as torch_data
from torchaudio.datasets import SPEECHCOMMANDS
from tqdm import trange
import functools

from typing import Tuple
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

URL = "speech_commands_v0.02"
EXCEPT_FOLDER = "_background_noise_"

target_mapper = {
    "bed": 0,
    "bird": 1,
    "cat": 2,
    "dog": 3,
    "down": 4,
    "eight": 5,
    "five": 6,
    "follow": 7,
    "forward": 8,
    "four": 9,
    "go": 10,
    "happy": 11,
    "house": 12,
    "learn": 13,
    "left": 14,
    "marvin": 15,
    "nine": 16,
    "no": 17,
    "off": 18,
    "on": 19,
    "one": 20,
    "right": 21,
    "seven": 22,
    "sheila": 23,
    "six": 24,
    "stop": 25,
    "three": 26,
    "tree": 27,
    "two": 28,
    "up": 29,
    "visual": 30,
    "wow": 31,
    "yes": 32,
    "zero": 33,
    "backward": 34
}
all_classes = target_mapper


# =============================================================================
# UTILITY FUNCTION
# =============================================================================


def _split_s_u(train_dataset, s_ratio: float = 1.0):
    _train_dataset = SpeechCommandsStats.from_dataset(train_dataset)

    nb_class = len(target_mapper)
    dataset_size = len(_train_dataset)

    if s_ratio == 1.0:
        return list(range(dataset_size)), []

    s_idx, u_idx = [], []
    nb_s = int(np.ceil(dataset_size * s_ratio) // nb_class)
    cls_idx = [[] for _ in range(nb_class)]

    # To each file, an index is assigned, then they are split into classes
    for i in trange(dataset_size):
        y, _, _ = _train_dataset[i]
        cls_idx[y].append(i)

    # Recover only the s_ratio % first as supervised, rest is unsupervised
    for i in trange(len(cls_idx)):
        random.shuffle(cls_idx[i])
        s_idx += cls_idx[i][:nb_s]
        u_idx += cls_idx[i][nb_s:]

    return s_idx, u_idx


def cache_feature(func):
    def decorator(*args, **kwargs):
        key = ",".join(map(str, args))

        if key not in decorator.cache:
            decorator.cache[key] = func(*args, **kwargs)

        return decorator.cache[key]

    decorator.cache = dict()
    decorator.func = func
    return decorator


class SpeechCommands(SPEECHCOMMANDS):
    def __init__(self,
                 root: str,
                 subset: str = "train",
                 url: str = URL,
                 download: bool = False,
                 transform: Module = None) -> None:
        super().__init__(root, url, download, transform)

        assert subset in ["train", "validation", "testing"]
        self.subset = subset
        self._keep_valid_files()

    @cache_feature
    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        waveform, _, label, _, _ = super().__getitem__(index)
        return waveform, target_mapper[label]

    def save_cache_to_disk(self, name) -> None:
        path = os.path.join(self._path, f"{name}_features.cache")
        torch.save(self.__getitem__.cache, path)

    def load_cache_from_disk(self, name) -> bool:
        path = os.path.join(self._path, f"{name}_features.cache")

        if os.path.isfile(path):
            disk_cache = torch.load(path)
            self.__getitem__.cache.update(disk_cache)
            return True

        return False

    def _keep_valid_files(self):
        bn = os.path.basename

        def file_list(filename):
            path = os.path.join(self._path, filename)
            with open(path, "r") as f:
                to_keep = f.read().splitlines()
                return set([bn(path) for path in to_keep])

        # Recover file list for validaiton and testing.
        validation_list = file_list("validation_list.txt")
        testing_list = file_list("testing_list.txt")

        # Create it for training
        training_list = [
            path
            for path in self._walker
            if bn(path) not in validation_list
            and bn(path) not in testing_list
        ]

        # Map the list to the corresponding subsets
        mapper = {
            "train": training_list,
            "validation": validation_list,
            "testing": testing_list,
        }

        self._walker = mapper[self.subset]

        # self._walker = [f for f in self._walker if bn(f) in mapper[self.subset]]


class SpeechCommandsStats(SpeechCommands):
    @classmethod
    def from_dataset(cls, dataset: SPEECHCOMMANDS):
        root = dataset.root

        newone = cls(root=root)
        newone.__dict__.update(dataset.__dict__)
        return newone

    def _load_item(self, filepath: str, path: str) -> Tuple[Tensor, int, str,
                                                            str, int]:
        HASH_DIVIDER = "_nohash_"
        relpath = os.path.relpath(filepath, path)
        label, filename = os.path.split(relpath)
        speaker, _ = os.path.splitext(filename)

        speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
        utterance_number = int(utterance_number)

        # remove Load audio
        # waveform, sample_rate = torchaudio.load(filepath)
        # return waveform, sample_rate, label, speaker_id, utterance_number
        return label, speaker_id, utterance_number

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        fileid = self._walker[index]

        label, speaker_id, utterance_number = self._load_item(
            fileid, self._path)

        return target_mapper[label], speaker_id, utterance_number


class SpeechCommand10(SpeechCommands):
    TRUE_CLASSES = ["yes", "no", "up", "down", "left",
                    "right", "off", "on", "go", "stop"]

    def __init__(self,
                 root: str,
                 subset: str = "train",
                 url: str = URL,
                 download: bool = False,
                 transform: Module = None,
                 percent_to_drop: float = 0.5) -> None:
        super().__init__(root, subset, url, download, transform)

        assert 0.0 < percent_to_drop < 1.0

        self.percent_to_drop = percent_to_drop

        self.target_mapper = {
            "yes": 0,
            "no": 1,
            "up": 2,
            "down": 3,
            "left": 4,
            "right": 5,
            "off": 6,
            "on": 7,
            "go": 8,
            "stop": 9,
            "_background_noise_": 11
        }

        # the rest of the classes belong to the "junk / trash / poubelle class"
        for cmd in all_classes:
            if cmd not in self.target_mapper:
                self.target_mapper[cmd] = 10

        self.drop_some_trash()
        self.add_silence()

    @cache_feature
    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        filepath = self._walker[index]

        label = filepath.split("/")[-2]
        target = self.target_mapper[label]

        if target == 11:
            waveform = self.get_noise(index)

        else:
            waveform, _ = super().__getitem__(index)

        return waveform, target

    def drop_some_trash(self):
        def is_trash(path):
            if self.target_mapper[path.split("/")[-2]] == 10:
                return True

            return False

        # Create the complete list of trash class
        trash_list = [path for path in self._walker if is_trash(path)]

        # choice only x% of it that will be removed
        nb_to_drop = int(len(trash_list) * self.percent_to_drop)
        to_drop = np.random.choice(trash_list, size=nb_to_drop, replace=False)

        # remove it from the _walker
        self._walker = list(set(self._walker) - set(to_drop))

    def add_silence(self):
        """simply add the filepath to _walker.
        The class "silence" will be processes differently
        """
        root_path = self._walker[0].split("/")[:-2]
        noise_path = os.path.join(*root_path, "_background_noise_")

        to_add = []
        for file in os.listdir(os.path.join(*root_path, EXCEPT_FOLDER)):
            if file[-4:] == ".wav":
                to_add.append(os.path.join(noise_path, file))

        self._walker.extend(to_add)

    @functools.lru_cache
    def get_complete_noise(self, filepath):
        waveform, sr = torchaudio.load(filepath)

        return waveform, sr

    def get_noise(self, index):
        filepath = self._walker[index]

        # Get the complete waveform of the corresponding noise file
        complete_waveform, sr = self.get_complete_noise(filepath)

        # randomly select 1 seconds
        nb_second = 1
        start_time = np.random.randint(0, len(complete_waveform[0]) - sr * nb_second)
        return complete_waveform[0, start_time:start_time+(nb_second * sr)]




if __name__ == "__main__":
    sc = SpeechCommand10(root=os.path.join("..", "..", "datasets"))

    for w, s in tqdm.tqdm(sc):
        pass

    for w, s in tqdm.tqdm(sc):
        pass


def dct(
    dataset_root,
    supervised_ratio: float = 0.1,
    batch_size: int = 100,

    train_transform: Module = None,
    val_transform: Module = None,

    **kwargs) -> Tuple[DataLoader, DataLoader]:

    loader_args = dict(
        num_workers=kwargs.get("num_workers", 0),
        pin_memory=kwargs.get("pin_memory", False),
    )
    dataset_path = os.path.join(dataset_root)

    # Validation subset
    val_dataset = SpeechCommands(root=dataset_path, subset="validation", transform=train_transform, download=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, **loader_args)

    # Training subset
    train_dataset = SpeechCommands(root=dataset_path, subset="train", transform=val_transform, download=True)
    s_idx, u_idx = _split_s_u(train_dataset, supervised_ratio)

    # Calc the size of the Supervised and Unsupervised batch
    nb_s_file = len(s_idx)
    nb_u_file = len(u_idx)

    s_batch_size = int(np.floor(batch_size * supervised_ratio))
    u_batch_size = int(np.ceil(batch_size * (1 - supervised_ratio)))

    sampler_s = SubsetRandomSampler(s_idx)
    sampler_u = SubsetRandomSampler(u_idx)

    train_loader_s1 = DataLoader(
        train_dataset, batch_size=s_batch_size, sampler=sampler_s, **loader_args)
    train_loader_s2 = DataLoader(
        train_dataset, batch_size=s_batch_size, sampler=sampler_s, **loader_args)
    train_loader_u = DataLoader(
        train_dataset, batch_size=u_batch_size, sampler=sampler_u, **loader_args)

    # combine the three loader into one
    train_loader = ZipCycle([train_loader_s1, train_loader_s2, train_loader_u])

    return None, train_loader, val_loader


def dct_uniloss(
        dataset_root,
        supervised_ratio: float = 0.1,
        batch_size: int = 100,

        train_transform: Module = None,
        val_transform: Module = None,

        **kwargs) -> Tuple[DataLoader, DataLoader]:
    return dct(**locals())


def mean_teacher(
        dataset_root,
        supervised_ratio: float = 0.1,
        batch_size: int = 128,

        train_transform: Module = None,
        val_transform: Module = None,

        **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Load the SpeechCommand for a student teacher learning
    """
    loader_args = dict(
        num_workers=kwargs.get("num_workers", 0),
        pin_memory=kwargs.get("pin_memory", False),
    )
    dataset_path = os.path.join(dataset_root)

    # validation subset
    val_dataset = SpeechCommands(root=dataset_path, subset="validation", transform=train_transform, download=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, **loader_args)

    # Training subset
    train_dataset = SpeechCommands(root=dataset_path, subset="train", transform=val_transform, download=True)
    s_idx, u_idx = _split_s_u(train_dataset, supervised_ratio)
    nb_s_file = len(s_idx)
    nb_u_file = len(u_idx)

    s_batch_size = int(np.floor(batch_size * supervised_ratio))
    u_batch_size = int(np.ceil(batch_size * (1 - supervised_ratio)))

    sampler_s = torch_data.SubsetRandomSampler(s_idx)
    sampler_u = torch_data.SubsetRandomSampler(u_idx)

    train_s_loader = torch_data.DataLoader(train_dataset, batch_size=s_batch_size, sampler=sampler_s)
    train_u_loader = torch_data.DataLoader(train_dataset, batch_size=u_batch_size, sampler=sampler_u)

    train_loader = ZipCycle([train_s_loader, train_u_loader])

    return None, train_loader, val_loader


# =============================================================================
#        SUPERVISED DATASETS
# =============================================================================
def supervised(
        dataset_root,
        supervised_ratio: float = 1.0,
        batch_size: int = 128,

        train_transform: Module = None,
        val_transform: Module = None,

        **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Load the SppechCommand for a supervised training
    """
    loader_args = dict(
        num_workers=kwargs.get("num_workers", 0),
        pin_memory=kwargs.get("pin_memory", False),
    )
    dataset_path = os.path.join(dataset_root)

    # validation subset
    val_dataset = SpeechCommands(
        root=dataset_path, subset="validation", transform=train_transform, download=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, **loader_args)

    # Training subset
    train_dataset = SpeechCommands(
        root=dataset_path, subset="train", transform=val_transform, download=True)

    if supervised_ratio == 1.0:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, **loader_args)

    else:
        s_idx, u_idx = _split_s_u(train_dataset, supervised_ratio)

        sampler_s = SubsetRandomSampler(s_idx)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler_s, **loader_args)

    return None, train_loader, val_loader