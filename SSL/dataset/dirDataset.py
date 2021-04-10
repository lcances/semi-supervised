import os
from typing import Tuple

import torchaudio
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset

class DirDataset(Dataset):
    def __init__(self, path: str, transform: Module = None) -> None:
        super().__init__()

        self.path = path
        self.transform = transform

        self.filenames = os.listdir(self.path)

    def __getitem__(self, idx: int) -> Tensor:
        path = os.path.join(self.path, self.filenames[idx])
        data, _ = torchaudio.load(path)

        if self.transform is not None:
            data = self.transform(data)
            data = data.squeeze()

        return self.filenames[idx], data

    def __len__(self) -> int:
        return len(self.filenames)

