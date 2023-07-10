import os

import typing

import pandas as pd

from torch.utils.data import Dataset as __Dataset__

class Dataset(__Dataset__):
    def __init__(
        self,
        path: typing.Union[str, os.PathLike],
        min_length: typing.Optional[int] = None,
        max_length: typing.Optional[int] = None
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError()
        
        df = pd.read_csv(path)

        if not min_length is None:
            df = df[df.iloc[:,-1].apply(lambda seq: min_length < len(seq))]
        if not max_length is None:
            df = df[df.iloc[:,-1].apply(lambda seq: max_length > len(seq))]

        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return self.df.shape[0]
    
    def __getitem__(
        self, 
        index: int
    ) -> str:
        return self.df.iloc[index, -1]

