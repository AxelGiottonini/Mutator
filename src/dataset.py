import os

import typing

import pandas as pd

from torch.utils import Dataset as __Dataset__

class Dataset(__Dataset__):
    def __init__(
        self,
        path: typing.Union[str, os.PathLike],
        min_length: typing.Optional[int] = None,
        max_length: typing.Optional[int] = None
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError()
        
        df = pd.read_csv(path, header=None)

        if not min_length is None:
            df = df[df.apply(lambda row: min_length < len(row[-1]), axis=1)]
        if not max_length is None:
            df = df[df.apply(lambda row: max_length > len(row[-1]), axis=1)]

        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return self.df.shape[0]
    
    def __getitem__(
        self, 
        index: int
    ) -> str:
        return self.df.iloc[index, -1]

