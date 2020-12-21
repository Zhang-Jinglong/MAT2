import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader

from .triplet import Triplet


class Preprocess:
    def __init__(
            self,
            data: pd.Series,
            metadata: pd.Series,
            norm: str = None):
        self.norm = norm
        data = np.array(data, dtype=np.float).T
        if norm:
            data = normalize(data, norm=norm)
            data = data * data.shape[1]

        self.dataset = data
        self.metadata = metadata

    def fix_data(self, data: np.array):
        if self.norm:
            data = normalize(data, norm=self.norm)
            data = data * data.shape[1]
        return data


class TrainingData:
    def __init__(
            self,
            data: pd.Series,
            metadata: pd.Series,
            mode: str,
            anchor: pd.Series = None,
            norm='l1',
            gamma: float = 0.0,
            shuffle: bool = True,
            num_workers: int = 2,
            batch_size: int = 256,
            pin_memory: bool = False):
        pre_data = Preprocess(data=data,
                              metadata=metadata,
                              norm=norm)
        anchor = Triplet(
            pre_data,
            mode=mode,
            anchor=anchor,
            record=False,
            gamma=gamma)
        label_code = pd.Categorical(metadata['batch']).codes

        self.anchor = anchor
        self._data_fix = pre_data
        self._shuffle = shuffle
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._pin_memory = pin_memory
        self._label_code = label_code

    def loader(self, stage: str = 'encoder', record: bool = False):
        self.anchor.set_stage(stage)
        self.anchor.set_record(record=record)
        dataloader = DataLoader(
            self.anchor,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory)
        return dataloader

    def set_attr(self,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 batch_size: int = 256,
                 pin_memory: bool = False):
        self._shuffle = shuffle
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._pin_memory = pin_memory

    def get_attr(self, attr: str = None):
        if attr == "num_workers":
            return self._num_workers
        elif attr == "batch_size":
            return self._batch_size
        elif attr == "pin_memory":
            return self._pin_memory
        elif attr == "label":
            return self._label_code
        else:
            return attr

    def fix_data(self, data: np.array):
        return self._data_fix.fix_data(data=data)


class TestingData(Dataset):
    def __init__(self,
                 model,
                 data: pd.Series):

        data = np.array(data).T
        data = model.data.fix_data(data=data)
        self.data = data
        self.loader = DataLoader(
            self,
            shuffle=False,
            num_workers=model.data.get_attr(attr="num_workers"),
            batch_size=model.data.get_attr(attr="batch_size"),
            pin_memory=model.data.get_attr(attr="pin_memory"))

    def __getitem__(self, item):
        return self.data[item], item

    def __len__(self):
        return len(self.data)
