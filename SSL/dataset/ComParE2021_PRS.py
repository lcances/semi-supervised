import os
from typing import Tuple

import torchaudio
from torch.utils.data import Dataset
from torch import Tensor
from torch.nn import Module
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)


class COMPARE2021_PRS(Dataset):
    CLASSES = ['background', 'chimpanze', 'geunon', 'mandrille', 'redcap']

    def __init__(self, root, subset: str) -> None:
        assert subset in ['train', 'test', 'devel']
        self.root = root
        self.subset = subset
        
        self.subsets_info = self._load_csv()
        self.wav_dir = os.path.join(self.root, 'ComParE2021_PRS', 'dist', 'wav')
        
        
    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        audio_name = self.subsets_info['audio_names'][idx]
        target = self.subsets_info['target'][idx]
        file_path = os.path.join(self.wav_dir, audio_name)
        
        waveform, sr = torchaudio.load(file_path)
        
        return waveform, target
    
    def __len__(self):
        return len(self.subsets_info['audio_names'])
    
    def _to_cls_idx(self, target_str: str) -> int:
        if target_str == '?':
            return -1

        return COMPARE2021_PRS.CLASSES.index(target_str)
        
    def _load_csv(self):
        def read_csv(path) -> Tuple[list, list]:
            with open(path, 'r') as f:
                lines = f.read().splitlines()
                lines = lines[1:]

            output = {
                'audio_names': [l.split(',')[0] for l in lines],
                'target': [self._to_cls_idx(l.split(',')[1]) for l in lines],
            }

            return output
        
        csv_root = os.path.join(self.root, 'ComParE2021_PRS','dist', 'lab')
        
        if self.subset == 'train':
            return read_csv(os.path.join(csv_root, 'train.csv'))
        
        elif self.subset == 'test':
            return read_csv(os.path.join(csv_root, 'test.csv'))
            
        return read_csv(os.path.join(csv_root, 'devel.csv'))