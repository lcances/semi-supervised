import functools
import itertools
import random
from typing import Tuple

import numpy
from tqdm import tqdm
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from SSL.util.utils import cache_to_disk, ZipCycleInfinite, Cacher, cache_to_disk, cache_feature
from SSL.dataset.ComParE2021_PRS import COMPARE2021_PRS


class ComParE2021_PRS(COMPARE2021_PRS):
    def __init__(self, root, subset, transform: Module = None, cache: bool = False):
        super().__init__(root, subset)
        self.transform = transform
        self.cache = cache

        self.cached_getitem = Cacher(self._cacheable_getitem)
        self.cached_transform = Cacher(self._cacheable_transform)

    def __getitem__(self, idx: int):
        data, target = self.cached_getitem(idx=idx, caching=True)
        data = self.cached_transform(data, key=idx, caching=self.cache)

        return data, target

    def _cacheable_getitem(self, idx: int):
        return super().__getitem__(idx)

    def _cacheable_transform(self, x, key):
        if self.transform is not None:
            return self.transform(x)


@cache_to_disk(path='.ComParE2021_PRS')
def class_balance_split(dataset,
                        supervised_ratio: float = 0.1,
                        unsupervised_ratio: float = None,
                        batch_size: int = 64,
                        verbose: bool = True,
                        seed: int = 1234,
                        ):

    def to_one_hot(idx):
        one_hot = [0] * len(COMPARE2021_PRS.CLASSES)
        one_hot[idx] = 1

        return one_hot

    def fill_subset(remaining_samples, expected):
        nb_classes = len(COMPARE2021_PRS.CLASSES)

        subset_occur = numpy.zeros(shape=(nb_classes,))
        subset = []

        with tqdm(total=sum(expected)) as progress:
            for class_idx in range(nb_classes):
                idx = 0
                while idx < len(remaining_samples) and subset_occur[class_idx] < expected[class_idx]:
                    if remaining_samples[idx][0][class_idx] == 1:
                        target, target_idx = remaining_samples.pop(idx)
                        subset_occur += target
                        subset.append(target_idx)
                        progress.update(sum(target))

                    idx += 1

        return numpy.asarray(subset), remaining_samples

    if unsupervised_ratio is None:
        unsupervised_ratio = 1 - supervised_ratio

    assert 0.0 <= supervised_ratio <= 1.0
    assert 0.0 <= unsupervised_ratio <= 1.0
    assert supervised_ratio + unsupervised_ratio <= 1.0

    if supervised_ratio == 1.0:
        return list(range(len(dataset))), []

    all_targets = list(map(to_one_hot, dataset.subsets_info['target']))
    all_target_idx = list(range(len(all_targets)))

    # expected occurance and tolerance
    total_occur = numpy.sum(all_targets, axis=0)
    s_expected_occur = numpy.ceil(total_occur * supervised_ratio)
    u_expected_occur = numpy.ceil(total_occur * unsupervised_ratio)
    print(' s_expected_occur: ', s_expected_occur)
    print("s expected occur: ", sum(s_expected_occur))

    all_sample = list(zip(all_targets, all_target_idx))
    s_subset, remaining_sample = fill_subset(all_sample, s_expected_occur)
    u_subset = numpy.asarray([s[1] for s in remaining_sample])

    return s_subset, u_subset


class IterationBalancedSampler(Sampler):
    def __init__(self, dataset: ComParE2021_PRS, index_list: list, shuffle: bool = True):
        super().__init__(None)
        self.dataset = dataset
        self.shuffle = shuffle

        self.index_list = index_list
        self.all_targets = dataset.subsets_info['target']
        self.sorted_sample_indexes = self._sort_per_class()

    def _sort_per_class(self):
        """ Pre-sort all the sample among the 527 different class.
        It will used to pick the correct file to feed the model
        """
        nb_classes = len(COMPARE2021_PRS.CLASSES)

        print('Sort the classes')
        class_indexes = [[] for _ in range(nb_classes)]

        for sample_idx, target in zip(self.index_list, self.all_targets):
            target_idx = [target]

            for t_idx in target_idx:
                class_indexes[t_idx].append(sample_idx)

        return class_indexes

    def _shuffle(self):
        # Sort the file for each class
        for i in self.sorted_sample_indexes:
            random.shuffle(i)

        # Sort the class order
        random.shuffle(self.sorted_sample_indexes)

    def __len__(self):
        return len(self.index_list)

    def __iter__(self):
        """ Round Robin algorithm to fetch file one by one from each class.
        """
        if self.shuffle:
            self._shuffle()

        nb_classes = len(COMPARE2021_PRS.CLASSES)

        global_index = 0
        for cls_idx in itertools.cycle(range(nb_classes)):
            # Increment the global index everytime we looped through all the classes
            if cls_idx == 0:
                global_index += 1

            selected_class = self.sorted_sample_indexes[cls_idx]
            local_idx = global_index % len(selected_class)

            yield selected_class[local_idx]


class InfiniteSampler(Sampler):
    def __init__(self, index_list: list, shuffle: bool = True):
        super().__init__(None)
        self.index_list = index_list

    def _shuffle(self):
        random.shuffle(self.index_list)

    @functools.lru_cache(maxsize=1)
    def __len__(self):
        return len(self.index_list)

    def __iter__(self):
        for i, idx in enumerate(itertools.cycle(self.index_list)):
            if i % len(self) == 0:
                self._shuffle()

            yield idx


def supervised(dataset_root,
               supervised_ratio: float = 0.1,
               batch_size: int = 128,

               train_transform: Module = None,
               val_transform: Module = None,
               augmentation: str = None,

               num_workers: int = 5,
               pin_memory: bool = False,
               seed: int = 1234,

               **kwargs) -> Tuple[DataLoader, DataLoader]:

    use_cache = True
    if augmentation is not None:
        use_cache = False
        print('Augmentation are used, disabling transform cache ...')
        
    loader_args = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }

    # Training subset
    train_dataset = ComParE2021_PRS(root=dataset_root, subset='train', transform=train_transform, cache=use_cache)
    s_idx, u_idx = class_balance_split(train_dataset, supervised_ratio, batch_size=batch_size, seed=seed)

    s_batch_size = int(numpy.floor(batch_size * supervised_ratio))
    # u_batch_size = int(numpy.ceil(batch_size * (1 - supervised_ratio)))

    sampler_s = IterationBalancedSampler(train_dataset, s_idx, shuffle=True)
    # sampler_u = InfiniteSampler(u_idx, shuffle=True)

    train_s_loader = DataLoader(
        train_dataset, batch_size=s_batch_size, sampler=sampler_s, **loader_args)
    # train_u_loader = DataLoader(train_dataset, batch_size=u_batch_size, sampler=sampler_u, **loader_args)

    # train_loader = ZipCycleInfinite([train_s_loader, train_u_loader])
    train_loader = train_s_loader

    # validation subset
    val_dataset = ComParE2021_PRS(root=dataset_root, subset='devel', transform=val_transform, cache=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_args)

    return None, train_loader, val_loader


def mean_teacher(
        dataset_root,
        supervised_ratio: float = 0.1,
        batch_size: int = 128,

        train_transform: Module = None,
        val_transform: Module = None,

        num_workers: int = 5,
        pin_memory: bool = False,
        seed: int = 1234,

        **kwargs) -> Tuple[DataLoader, DataLoader]:

    loader_args = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }

    # Training subset
    train_dataset = ComParE2021_PRS(
        root=dataset_root, subset='train', transform=train_transform)
    print(train_dataset.__getitem__)
    s_idx, u_idx = class_balance_split(
        train_dataset, supervised_ratio, batch_size=batch_size, seed=seed)

    s_batch_size = int(numpy.floor(batch_size * supervised_ratio))
    u_batch_size = int(numpy.ceil(batch_size * (1 - supervised_ratio)))

    print('s_idx: ', len(s_idx))
    print('u_idx: ', len(u_idx))

    sampler_s = IterationBalancedSampler(train_dataset, s_idx, shuffle=True)
#     sampler_s = SubsetRandomSampler(s_idx)
    sampler_u = InfiniteSampler(u_idx, shuffle=True)

    train_s_loader = DataLoader(
        train_dataset, batch_size=s_batch_size, sampler=sampler_s, **loader_args)
    train_u_loader = DataLoader(
        train_dataset, batch_size=u_batch_size, sampler=sampler_u, **loader_args)

    train_loader = ZipCycleInfinite([train_s_loader, train_u_loader])

    # validation subset
    val_dataset = ComParE2021_PRS(
        root=dataset_root, subset='devel', transform=val_transform)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **loader_args)

    return None, train_loader, val_loader


def dct(**kwargs):
    pass


def fixmatch(**kwargs):
    pass
