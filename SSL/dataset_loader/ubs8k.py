from SSL.util.utils import ZipCycle, Cacher
from SSL.dataset.ubs8k import URBANSOUND8K
from torch.nn import Module
from tqdm import tqdm


import os
import random
import numpy as np
from copy import copy
import torch.utils.data as torch_data


class UrbanSound8K(URBANSOUND8K):
    def __init__(self, root, folds, transform: Module = None, cache: bool = False):
        super().__init__(root, folds)
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


def split_s_u(dataset, s_ratio: float) -> list:
    idx_list = list(range(len(dataset.meta['filename'])))
    s_idx, u_idx = [], []

    # sort the classes
    class_idx = [[] for _ in range(URBANSOUND8K.NB_CLASS)]
    for idx in tqdm(idx_list):
        class_idx[dataset.meta['target'][idx]].append(idx)

    # split each class seperatly to keep distribution
    for i in range(URBANSOUND8K.NB_CLASS):
        random.shuffle(class_idx[i])

        nb_item = len(class_idx[i])
        nb_s = int(np.ceil(nb_item * s_ratio))

        s_idx += class_idx[i][:nb_s]
        u_idx += class_idx[i][nb_s:]

    return s_idx, u_idx


def supervised(
    dataset_root,
    supervised_ratio: float = 1.0,
    batch_size: int = 64,

    train_folds: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9),
    val_folds: tuple = (10, ),

    train_transform: Module = None,
    val_transform: Module = None,
    augmentation: str = None,

    num_workers: int = 0,
    pin_memory: bool = False,

    verbose=1,
    **kwargs,
):
    """
    Load the UrbanSound dataset for supervised systems.
    """
    use_cache = True
    if augmentation is not None:
        use_cache = False
        print('Augmentation are used, disabling transform cache ...')

    loader_args = {
        'num_workers': num_workers,
        'pin_memory': pin_memory
    }

    # validation subset
    # val_dataset = Dataset(manager, folds=val_folds, cached=True)
    val_dataset = UrbanSound8K(dataset_root, val_folds, transform=val_transform, cache=True)
    print('nb file validation: ', len(val_dataset))
    val_loader = torch_data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, **loader_args)

    # training subset
    # train_dataset = Dataset(manager, folds=train_folds, cached=True)
    train_dataset = UrbanSound8K(dataset_root, train_folds, transform=train_transform, cache=use_cache)
    print('nb file training: ', len(train_dataset))

    if supervised_ratio == 1.0:
        train_loader = torch_data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_args)

    else:
        s_idx, u_idx = split_s_u(train_dataset, supervised_ratio)

        # Train loader only use the s_idx
        sampler_s = torch_data.SubsetRandomSampler(s_idx)
        train_loader = torch_data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler_s)

    return None, train_loader, val_loader


def mean_teacher(
    dataset_root,
    supervised_ratio: float = 0.1,
    batch_size: int = 64,

    train_folds: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9),
    val_folds: tuple = (10, ),

    train_transform: Module = None,
    val_transform: Module = None,
    augmentation: str = None,

    num_workers: int = 0,
    pin_memory: bool = False,

    verbose=1,
    **kwargs):

    use_cache = True
    if augmentation is not None:
        use_cache = False
        print('Augmentation are used, disabling transform cache ...')

    loader_args = {
        'num_workers': num_workers,
        'pin_memory': pin_memory
    }

    # validation subset
    val_dataset = UrbanSound8K(dataset_root, val_folds, transform=val_transform, cache=True)
    val_loader = torch_data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, **loader_args)

    # training subset
    train_dataset = UrbanSound8K(dataset_root, train_folds, transform=train_transform, cache=use_cache)

    # Calc the size of the Supervised and Unsupervised batch
    s_idx, u_idx = split_s_u(train_dataset, supervised_ratio)

    s_batch_size = int(np.floor(batch_size * supervised_ratio))
    u_batch_size = int(np.ceil(batch_size * (1 - supervised_ratio)))

    sampler_s = torch_data.SubsetRandomSampler(s_idx)
    sampler_u = torch_data.SubsetRandomSampler(u_idx)

    train_s_loader = torch_data.DataLoader(train_dataset, batch_size=s_batch_size, sampler=sampler_s, **loader_args)
    train_u_loader = torch_data.DataLoader(train_dataset, batch_size=u_batch_size, sampler=sampler_u, **loader_args)

    train_loader = ZipCycle([train_s_loader, train_u_loader], align='max')

    return None, train_loader, val_loader


def dct(
        dataset_root,
        supervised_ratio: float = 0.1,
        batch_size: int = 100,

        train_folds: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9),
        val_folds: tuple = (10, ),

        train_transform: Module = None,
        val_transform: Module = None,
        augmentation: str = None,

        num_workers: int = 0,
        pin_memory: bool = False,

        verbose=1, **kwargs):

    use_cache = True
    if augmentation is not None:
        use_cache = False
        print('Augmentation are used, disabling transform cache ...')

    loader_args = {
        'num_workers': num_workers,
        'pin_memory': pin_memory
    }

    # validation subset
    val_dataset = UrbanSound8K(dataset_root, val_folds, transform=val_transform, cache=True)
    val_loader = torch_data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, **loader_args)

    # training subset
    train_dataset = UrbanSound8K(dataset_root, train_folds, transform=train_transform, cache=use_cache)

    # Calc the size of the Supervised and Unsupervised batch
    s_idx, u_idx = split_s_u(train_dataset, supervised_ratio)

    s_batch_size = int(np.floor(batch_size * supervised_ratio))
    u_batch_size = int(np.ceil(batch_size * (1 - supervised_ratio)))

    sampler_s1 = torch_data.SubsetRandomSampler(s_idx)
    sampler_s2 = torch_data.SubsetRandomSampler(s_idx)
    sampler_u = torch_data.SubsetRandomSampler(u_idx)

    train_loader_s1 = torch_data.DataLoader(train_dataset, batch_size=s_batch_size, sampler=sampler_s1, **loader_args)
    train_loader_s2 = torch_data.DataLoader(train_dataset, batch_size=s_batch_size, sampler=sampler_s2, **loader_args)
    train_loader_u = torch_data.DataLoader(train_dataset, batch_size=u_batch_size, sampler=sampler_u, **loader_args)
    train_loader = ZipCycle([train_loader_s1, train_loader_s2, train_loader_u])

    return None, train_loader, val_loader





def dct_uniloss(
        dataset_root,
        supervised_ratio: float = 0.1,
        batch_size: int = 100,

        train_folds: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9),
        val_folds: tuple = (10, ),

        verbose=1, **kwargs):
    pass


def fixmatch(**kwargs):
    pass
