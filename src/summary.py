import sys
import os
import typing
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import UserList
from functools import cache, cached_property
import glob
import re
import base64
from io import BytesIO
import pandas as pd
import numpy as np
import scipy as sp
import torch
from transformers import AutoModelForMaskedLM
import matplotlib.pyplot as plt
import seaborn as sns
from airium import Airium
import tqdm
from perplexity import perplexity
from getters import get_model, get_dataloaders

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BOOTSTRAP_CSS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css"
BOOTSTRAP_JS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/js/bootstrap.bundle.min.js"

def _prune_dict(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}

@dataclass
class BertAttributes():
    learning_rate: float=None
    batch_size: int=None
    mask_rate: float=None
    model_selection: str=None

    def __hash__(self) -> int:
        return hash((
            self.learning_rate,
            self.batch_size,
            self.masking_rate,
            self.model_selection
        ))

    @cached_property
    def dataframe(self):
        return pd.DataFrame({
            "Masking rate": [self.mask_rate],
            "Model selection": [self.model_selection],
            "Learning rate": [self.learning_rate],
            "Batch size": [self.batch_size]
        })

    def to_html(self, a: Airium) -> None:
        with a.ul(klss="list-group list-group-flush"):
            with a.li(klass="list_group-item"): a(f"Model selection: {self.model_selection}")
            with a.li(klass="list_group-item"): a(f"Learning rate: {self.learning_rate}")
            with a.li(klass="list_group-item"): a(f"Batch size: {self.batch_size}")
            with a.li(klass="list_group-item"): a(f"Mask rate: {self.mask_rate}")

    def get_output_file_name(
        self,
        name: str,
        out_dir: Union[str, os.PathLike]="",
        attributes_order: List[str]=[
            "mask_rate",
            "model_selection",
            "learning_rate",
            "batch_size"]
    ):
        return os.path.join(
            out_dir,
            *[str(self.__getattribute__(attribute)) for attribute in attributes_order],
            name
        )

    @classmethod
    def from_file_name(
        cls,
        file_name: Union[str, os.PathLike],
        pattern: str = r"(.*\/)?LR(?P<learning_rate>[\d.]*)_BS(?P<batch_size>[\d.]*)_P(?P<mask_rate>[\d.]*)" + \
                       r"\/(?P<model_selection>(best)|(final)|(validation))"
    ):
        attributes = _prune_dict(re.match(pattern, file_name).groupdict())
        if attributes["model_selection"] == "best": attributes["model_selection"] = "validation"
        return cls(**attributes)

class BertAttributesList(UserList):
    def __hash__(self) -> int:
        return hash(tuple(hash(el) for el in self))
    
    @cached_property
    def model_selection(self) -> List[str]:
        return [el.model_selection for el in self]

    @cached_property
    def learning_rate(self) -> List[float]:
        return [el.learning_rate for el in self]

    @cached_property
    def batch_size(self) -> List[int]:
        return [el.batch_size for el in self]

    @cached_property
    def mask_rate(self) -> List[float]:
        return [el.mask_rate for el in self]
    
    @cached_property
    def dataframe(self) -> pd.DataFrame:
        dataframe = pd.DataFrame({
            "Mask rate": self.mask_rate,
            "Model selection": self.model_selection,
            "Learning rate": self.learning_rate,
            "Batch size": self.batch_size,
        })
        dataframe = dataframe.astype({
            "Mask rate":float,
            "Model selection":str,
            "Learning rate":float,
            "Batch size":int,
        })
        return dataframe
    
    #@cache
    def repeated_dataframe(self, n_repeats:int=2):
        dataframe = pd.DataFrame({
            "Mask rate": [el for el in self.mask_rate for _ in range(n_repeats)],
            "Model selection": [el for el in self.model_selection for _ in range(n_repeats)],
            "Learning rate": [el for el in self.learning_rate for _ in range(n_repeats)],
            "Batch size": [el for el in self.batch_size for _ in range(n_repeats)]
        })
        dataframe = dataframe.astype({
            "Mask rate":float,
            "Model selection":str,
            "Learning rate":float,
            "Batch size":int,
        })
        return dataframe
    
    def get_output_file_names(
        self,
        name: str,
        out_dir: Union[str, os.PathLike]="",
        attributes_order: List[str]=[
            "mask_rate",
            "model_selection",
            "learning_rate",
            "batch_size"]
    ):
        return [el.get_output_file_name(name, out_dir, attributes_order) for el in self]
    
    @classmethod
    def from_file_names(
        cls,
        file_names: List[Union[str, os.PathLike]],
        pattern: str = r"(.*\/)?LR(?P<learning_rate>[\d.]*)_BS(?P<batch_size>[\d.]*)_P(?P<mask_rate>[\d.]*)" + \
                       r"\/(?P<model_selection>(best)|(final)|(validation))"
    ):
        if not isinstance(file_names, typing.Iterable):
            file_names = [file_names]
    
        bert_inference_attributes_list = cls()    
        for el in file_names:
            bert_inference_attributes_list.append(BertAttributes.from_file_name(el, pattern))

        return bert_inference_attributes_list

    @classmethod
    def from_adapters(cls, *args, **kwargs):
        return cls.from_file_names(*args, **kwargs)
    
@dataclass
class InferenceAttributes():
    n_iter: str="0"
    top_k_sampling: str="3"
    model_selection: str=None
    learning_rate: str=None
    batch_size: str=None
    mask_rate: str=None

    def __hash__(self) -> int:
        return hash((
            self.n_iter,
            self.top_k_sampling,
            self.model_selection,
            self.learning_rate,
            self.batch_size,
            self.mask_rate
        ))

    def to_html(self, a: Airium) -> None:
        with a.ul(klss="list-group list-group-flush"):
            with a.li(klass="list_group-item"): a(f"Model selection: {self.model_selection}")
            with a.li(klass="list_group-item"): a(f"Learning rate: {self.learning_rate}")
            with a.li(klass="list_group-item"): a(f"Batch size: {self.batch_size}")
            with a.li(klass="list_group-item"): a(f"Mask rate: {self.mask_rate}")
            if self.n_iter != 0:
                with a.li(klass="list_group-item"): a(f"Random inference with {self.n_iter} iterations")
            with a.li(klass="list_group-item"): a(f"Top K Sampling: {self.top_k_sampling}")

    def get_output_file_name(
        self,
        name: str,
        out_dir: Union[str, os.PathLike]="",
        attributes_order: List[str]=[
            "mask_rate",
            "model_selection",
            "learning_rate",
            "batch_size",
            "n_iter",
            "top_k_sampling"]
    ):
        return os.path.join(
            out_dir,
            *[str(self.__getattribute__(attribute)) for attribute in attributes_order],
            name
        )

    @classmethod
    def from_file_name(
        cls,
        file_name: Union[str, os.PathLike],
        pattern: str = r"(.*\/)?inference(\.random\.(?P<n_iter>\d+)(\.(?P<top_k_sampling>\d+)?)?)?\." +
                       r"(?P<model_selection>(final)|(validation))\." + 
                       r"LR(?P<learning_rate>.*)_BS(?P<batch_size>.*)_P(?P<mask_rate>.*)\.csv"
    ):
        return cls(**_prune_dict(re.match(pattern, file_name).groupdict()))

class InferenceAttributesList(UserList):
    def __hash__(self) -> int:
        return hash(tuple(hash(el) for el in self))

    @cached_property
    def n_iter(self) -> List[str]:
        return [el.n_iter for el in self]

    @cached_property
    def top_k_sampling(self) -> List[str]:
        return [el.top_k_sampling for el in self]

    @cached_property
    def model_selection(self) -> List[str]:
        return [el.model_selection for el in self]

    @cached_property
    def learning_rate(self) -> List[str]:
        return [el.learning_rate for el in self]

    @cached_property
    def batch_size(self) -> List[str]:
        return [el.batch_size for el in self]

    @cached_property
    def mask_rate(self) -> List[str]:
        return [el.mask_rate for el in self]

    @cached_property
    def _dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Mask rate": self.mask_rate,
            "Model selection": self.model_selection,
            "Learning rate": self.learning_rate,
            "Batch size": self.batch_size,
            "N. iterations": self.n_iter,
            "Top K Sampling": self.top_k_sampling
        })

    @cached_property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe.astype({
            "Mask rate":float,
            "Model selection":str,
            "Learning rate":float,
            "Batch size":int,
            "N. iterations":int,
            "Top K Sampling": int
        })

    def get_levels(
        self,
        attributes_order: List[str]=[
            "Mask rate",
            "Model selection",
            "Learning rate",
            "Batch size",
            "N. iterations",
            "Top K Sampling"]
    ) -> List[int]:
        levels = []
        for i in range(len(attributes_order) - 1):
            levels.append(
                self._dataframe.groupby(attributes_order[0:i+1]).groups
            )
        levels = {k:v.to_list() for level in levels for k, v in level.items()}
        return levels

    def get_output_file_names(
        self,
        name: str,
        out_dir: Union[str, os.PathLike]="",
        attributes_order: List[str]=[
            "mask_rate",
            "model_selection",
            "learning_rate",
            "batch_size",
            "n_iter",
            "top_k_sampling"
        ],
        skip=None
    ):
        if isinstance(skip, int):
            attributes_order = attributes_order[skip:]
        return [el.get_output_file_name(name, out_dir, attributes_order) for el in self]

@dataclass
class PerplexityStatsOutput():
    perplexity: pd.Series
    base_perplexity: pd.Series

    def __hash__(self) -> int:
        return hash(tuple(
            pd.util.hash_pandas_object(self.perplexity).to_list() + \
            pd.util.hash_pandas_object(self.base_perplexity).to_list()
        ))

    @cached_property
    def pcc(self) -> float:
        return sp.stats.pearsonr(self.base_perplexity, self.perplexity).statistic

    @cached_property
    def d_perplexity(self) -> pd.Series:
        return self.base_perplexity - self.perplexity

    #@cache
    def d_quantile(self, q: float) -> float:
        if isinstance(q, typing.Iterable):
            return [self.d_quantile(el) for el in q]
        return self.d_perplexity.quantile(q)

    @cached_property
    def d_mean(self) -> float:
        return self.d_perplexity.mean()

    @cached_property
    def perplexity_distributions(self) -> str:
        sns.jointplot(x=self.base_perplexity, y=self.perplexity, kind="kde")
        tmpfile = BytesIO()
        plt.gcf().set_size_inches(10, 10)
        plt.savefig(tmpfile, format='png', dpi=200)
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf8')
        plt.close()
        return encoded

    @cached_property
    def d_perplexity_distribution(self) -> str:
        sns.displot(x=self.d_perplexity, kind="kde", fill=True)
        tmpfile = BytesIO()
        plt.gcf().set_size_inches(10, 10)
        plt.savefig(tmpfile, format='png', dpi=200)
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf8')
        plt.close()
        return encoded

    def to_html(self, a: Airium) -> None:
        with a.table(klass="table"):
            with a.thead():
                with a.tr():
                    with a.th(scope="col"): a("Mean &#916;Perplexity")
                    with a.th(scope="col"): a("Q1 &#916;Perplexity")
                    with a.th(scope="col"): a("Q2 &#916;Perplexity")
                    with a.th(scope="col"): a("Q3 &#916;Perplexity")
                    with a.th(scope="col"): a("PCC")
            with a.tbody():
                with a.tr():
                    with a.td(scope="col"): a(f"{self.d_mean:.3f}")
                    with a.td(scope="col"): a(f"{self.d_quantile(0.25):.3f}")
                    with a.td(scope="col"): a(f"{self.d_quantile(0.50):.3f}")
                    with a.td(scope="col"): a(f"{self.d_quantile(0.75):.3f}")
                    with a.td(scope="col"): a(f"{self.pcc:.3f}")

        with a.div(klass="container"):
            with a.div(klass="row"):
                with a.div(klass="col-sm"):
                    with a.h4(): a("&#916;Perplexity Distribution")
                    a.img(
                        src=f"data:image/png;base64,{self.d_perplexity_distribution}",
                        alt="",
                        klass="img-fluid"
                    )
                with a.div(klass="col-sm"):
                    with a.h4(): a("Perplexity Distribution")
                    a.img(
                        src=f"data:image/png;base64,{self.perplexity_distributions}",
                        alt="",
                        klass="img-fluid"
                    )

    @classmethod
    def from_file(cls, file_name: Union[str, os.PathLike]):
        dataframe = pd.read_csv(file_name)
        return cls(
            dataframe["perplexity_mut"],
            dataframe["perplexity_src"]
        )

class PerplexityStatsOutputList(UserList):
    def __hash__(self) -> int:
        return hash(tuple(hash(el) for el in self))

    @cached_property
    def pcc(self) -> List[float]:
        return [el.pcc for el in self]

    #@cache
    def d_quantile(self, q: float) -> List[float]:
        return [el.d_quantile(q) for el in self]

    @cached_property
    def d_mean(self) -> List[float]:
        return [el.d_mean for el in self]

    @cached_property
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Mean &#916;Perplexity": self.d_mean,
            "Q1 &#916;Perplexity": self.d_quantile(0.25),
            "Q2 &#916;Perplexity": self.d_quantile(0.50),
            "Q3 &#916;Perplexity": self.d_quantile(0.75),
            "PCC": self.pcc
        }, dtype=float)

    #@cache
    def join_attributes(self, attributes_list:InferenceAttributesList) -> pd.DataFrame:
        dataframe = pd.concat([attributes_list.dataframe, self.dataframe], axis=1)
        dataframe = dataframe.sort_values([
            "Mask rate",
            "Model selection",
            "Learning rate",
            "Batch size",
            "N. iterations",
            "Top K Sampling"
        ])
        return dataframe

    def to_html(
        self,
        a: Airium,
        attributes_list: InferenceAttributesList,
        links: List[Union[str, os.PathLike]]
    ) -> None:
        dataframe = self.join_attributes(attributes_list)
        links = [links[i] for i in dataframe.index.to_list()]

        with a.table(klass="table table-sm table-striped"):
            with a.thead():
                with a.tr():
                    with a.th(scope="col"): a("#")
                    with a.th(scope="col"): a("Model selection")
                    with a.th(scope="col"): a("Learning rate")
                    with a.th(scope="col"): a("Batch size")
                    with a.th(scope="col"): a("Mask rate")
                    with a.th(scope="col"): a("N. iterations")
                    with a.th(scope="col"): a("Top K Sampling")
                    with a.th(scope="col"): a("Mean &#916;Perplexity")
                    with a.th(scope="col"): a("Q1 &#916;Perplexity")
                    with a.th(scope="col"): a("Q2 &#916;Perplexity")
                    with a.th(scope="col"): a("Q3 &#916;Perplexity")
                    with a.th(scope="col"): a("PCC")
            with a.tbody():
                for i, (_, row) in enumerate(dataframe.iterrows()):
                    with a.tr():
                        with a.th(scope="col"):
                            with a.a(href=f"{links[i]}"): a(f"{i}")
                        with a.td(scope="col"): a(f"{row['Model selection']}")
                        with a.td(scope="col"): a(f"{row['Learning rate']:.4f}")
                        with a.td(scope="col"): a(f"{row['Batch size']}")
                        with a.td(scope="col"): a(f"{row['Mask rate']:.3f}")
                        with a.td(scope="col"): a(f"{row['N. iterations']}")
                        with a.td(scope="col"): a(f"{row['Top K Sampling']}")
                        with a.td(scope="col"): a(f"{row['Mean &#916;Perplexity']:.3f}")
                        with a.td(scope="col"): a(f"{row['Q1 &#916;Perplexity']:.3f}")
                        with a.td(scope="col"): a(f"{row['Q2 &#916;Perplexity']:.3f}")
                        with a.td(scope="col"): a(f"{row['Q3 &#916;Perplexity']:.3f}")
                        with a.td(scope="col"): a(f"{row['PCC']:.3f}")

    def to_latex(
        self,
        attributes_list: InferenceAttributesList,
        index: Optional[bool]=False,
        float_format: Optional[typing.Callable]="{:.4f}".format,
        *args: Optional[typing.Any],
        **kwargs: Optional[typing.Any]
    ) -> str:
        dataframe = self.join_attributes(attributes_list)
        dataframe = dataframe.rename(
            {
                "Mean &#916;Perplexity": "$\Bar{\Delta_\text{mes,mut}}$",
                "Q1 &#916;Perplexity": "$\Delta_\text{mes,mut}_0.25$",
                "Q2 &#916;Perplexity": "$\Delta_\text{mes,mut}_0.50$",
                "Q3 &#916;Perplexity": "$\Delta_\text{mes,mut}_0.75$",
                "PCC": "$\rho_\text{mes,mut}$"
            },
            axis=1
        )
        return dataframe.to_latex(index=index, float_format=float_format, *args, **kwargs)

@dataclass
class MutationStatsOutput():
    sequences: pd.Series
    base_sequences: pd.Series

    def __hash__(self) -> int:
        return hash(tuple(
            pd.util.hash_pandas_object(self.sequences).to_list() + \
            pd.util.hash_pandas_object(self.base_sequences).to_list()
        ))

    @cached_property
    def paired_sequences(self) -> List[Tuple[str, str]]:
        return [
            (base_sequence, sequence) \
                for base_sequence, sequence in zip(self.base_sequences, self.sequences) \
                if len(base_sequence) == len(sequence)
        ]

    @cached_property
    def n_unpaired_sequences(self) -> int:
        return len(self.base_sequences)  - len(self.paired_sequences)

    @cached_property
    def _nodes(self) -> List[str]:
        return list(set(
            [el for seq in self.sequences for el in seq] + \
            [el for seq in self.base_sequences for el in seq]
        ))

    @cached_property
    def _edges(self) -> List[str]:
        return [
            (base_sequence[i], sequence[i]) \
                for base_sequence, sequence in self.paired_sequences \
                for i in range(len(sequence))
        ]

    @cached_property
    def _adjacency_matrix(self) -> np.ndarray:
        adjacency_matrix = {
            from_node: {to_node: 0 for to_node in self._nodes} \
                for from_node in self._nodes
        }
        for from_node, to_node in self._edges: adjacency_matrix[from_node][to_node] += 1
        adjacency_matrix = np.array([list(row.values()) for row in adjacency_matrix.values()])
        adjacency_matrix = adjacency_matrix / adjacency_matrix.sum(axis=1)[:, None]
        np.fill_diagonal(adjacency_matrix, 0)
        return adjacency_matrix

    #@cache
    def top_from(self, n_top:int=10) -> List[str]:
        return [
            self._nodes[i] \
                for i in np.argsort(self._adjacency_matrix.sum(axis=1))[-n_top:]
        ][::-1]

    #@cache
    def top_to(self, n_top:int=10) -> List[str]:
        return [
            self._nodes[i] \
                for i in np.argsort(self._adjacency_matrix.sum(axis=0))[-n_top:]
        ][::-1]

    #@cache
    def top_edges(self, n_top:int=20) -> List[Tuple[str, str, float]]:
        top_edges = list(zip(
            list(self._adjacency_matrix.argsort(axis=None)[-n_top:] // len(self._adjacency_matrix)),
            list(self._adjacency_matrix.argsort(axis=None)[-n_top:] % len(self._adjacency_matrix))
        ))[::-1]
        top_edges = [
            (self._nodes[from_node],
             self._nodes[to_node],
             self._adjacency_matrix[from_node, to_node]) for from_node, to_node in top_edges]
        return top_edges

    @cached_property
    def clustermap(self) -> str:
        sns.clustermap(self._adjacency_matrix, xticklabels=self._nodes, yticklabels=self._nodes)
        tmpfile = BytesIO()
        plt.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf8')
        plt.close()
        return encoded

    def to_html(self, a: Airium) -> None:
        with a.div(klass="alert alert-warning", role="alert"):
            a(f"{self.n_unpaired_sequences} sequences were not successfully paired...")

        with a.h4(): a("Top Mutated Amino Acids")
        with a.table(klass="table"):
            with a.thead():
                with a.tr():
                    with a.th(scope="col"): a("#")
                    for i in range(len(self.top_from())):
                        with a.th(scope="col"): a(str(i))
            with a.tbody():
                with a.tr():
                    with a.th(scope="col"): a("from")
                    for i in range(len(self.top_from())):
                        with a.td(scope="col"): a(self.top_from()[i])
                with a.tr():
                    with a.th(scope="col"): a("to")
                    for i in range(len(self.top_to())):
                        with a.td(scope="col"): a(self.top_to()[i])

        with a.h4(): a("Top Mutations")
        with a.table(klass="table table-sm table-striped"):
            with a.thead():
                with a.tr():
                    with a.th(scope="col"): a("from")
                    with a.th(scope="col"): a("to")
                    with a.th(scope="col"): a("P(t|f)")
            with a.tbody():
                for from_node, to_node, p in self.top_edges():
                    with a.tr():
                        with a.td(scope="col"): a(from_node)
                        with a.td(scope="col"): a(to_node)
                        with a.td(scope="col"): a(f"{p:.3f}")

        with a.h4(): a("Mutations Clustermap")
        a.img(src=f"data:image/png;base64,{self.clustermap}", alt="", klass="img-fluid")

    @classmethod
    def from_files(
        cls,
        file_name: Union[str, os.PathLike],
        base_file_name: Union[str, os.PathLike]
    ):
        dataframe = pd.read_csv(file_name)
        base_dataframe = pd.read_csv(base_file_name)
        return MutationStatsOutput(
            dataframe.iloc[:, -1],
            base_dataframe.iloc[:, -1]
        )

class MutationStatsOutputList(UserList):
    def __hash__(self) -> int:
        return hash(tuple(hash(el) for el in self))

    @staticmethod
    def _top_list(
        top_lists: List[typing.Any],
        fun: typing.Callable=lambda x: x
    ) -> List[typing.Any]:
        scores = list(range(len(top_lists[0]), 0, -1)) * len(top_lists)
        top_list = [el for top_list in top_lists for el in top_list]

        cum_score = {el:0 for el in set(top_list)}
        for el, score in zip(top_list, scores): cum_score[el] += score

        top_list = sorted(cum_score.items(), key=lambda x:x[1], reverse=True)
        top_list = [fun(el) for el, _ in top_list[:len(top_lists[0])]]

        return top_list

    #@cache
    def top_from(
        self,
        n_top: int=10,
        process: typing.Callable=lambda x: x
    ) -> List[str]:
        return self._top_list([process(el.top_from(n_top)) for el in self])

    #@cache
    def top_to(
        self,
        n_top: int=10,
        process: typing.Callable=lambda x: x
    ) -> List[str]:
        return self._top_list([process(el.top_to(n_top)) for el in self])

    #@cache
    def top_edges(
        self,
        n_top: int=20,
        process: typing.Callable=lambda x: [f"{f}{t}" for f, t, _ in x]
    ) -> List[Tuple[str, str]]:
        return self._top_list(
            [process(el.top_edges(n_top)) for el in self],
            fun=lambda x: (x[0], x[1])
        )

    def to_html(self, a:Airium) -> None:
        top_from = self.top_from()
        top_to = self.top_to()
        top_edges = self.top_edges()

        with a.h4(): a("Top Mutated Amino Acids")
        with a.table(klass="table"):
            with a.thead():
                with a.tr():
                    with a.th(scope="col"): a("#")
                    for i in range(len(top_from)):
                        with a.th(scope="col"): a(str(i))
            with a.tbody():
                with a.tr():
                    with a.th(scope="col"): a("from")
                    for i in range(len(top_from)):
                        with a.td(scope="col"): a(top_from[i])
                with a.tr():
                    with a.th(scope="col"): a("to")
                    for i in range(len(top_to)):
                        with a.td(scope="col"): a(top_to[i])

        with a.h4(): a("Top Mutations")
        with a.table(klass="table table-sm table-striped"):
            with a.thead():
                with a.tr():
                    with a.th(scope="col"): a("from")
                    with a.th(scope="col"): a("to")

            with a.tbody():
                for from_node, to_node in top_edges:
                    with a.tr():
                        with a.td(scope="col"): a(from_node)
                        with a.td(scope="col"): a(to_node)

@dataclass
class ThermophilicityStatsOutput():
    dataframe: pd.DataFrame

    def __hash__(self) -> int:
        return hash(tuple(
            pd.util.hash_pandas_object(self.dataframe).to_list()
        ))

    @cached_property
    def description(self) -> pd.DataFrame:
        return self.dataframe.describe().loc[["mean", "25%", "50%", "75%"],:].T

    @cached_property
    def distributions(self) -> str:
        sns.pairplot(self.dataframe)
        tmpfile = BytesIO()
        plt.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf8')
        plt.close()
        return encoded

    def to_html(self, a:Airium) -> None:
        with a.table(klass="table"):
            with a.thead():
                with a.tr():
                    with a.th(scope="col"): a("#")
                    with a.th(scope="col"): a("Mean")
                    with a.th(scope="col"): a("Q1")
                    with a.th(scope="col"): a("Q2")
                    with a.th(scope="col"): a("Q3")
            with a.tbody():
                for index, row in self.description.iterrows():
                    with a.tr():
                        with a.th(scope="col"): a(f"{index}")
                        with a.td(scope="col"): a(f"{row['mean']:.3f}")
                        with a.td(scope="col"): a(f"{row['25%']:.3f}")
                        with a.td(scope="col"): a(f"{row['50%']:.3f}")
                        with a.td(scope="col"): a(f"{row['75%']:.3f}")

        a.img(
            src=f"data:image/png;base64,{self.distributions}",
            alt="",
            klass="img-fluid"
        )

    @classmethod
    def from_file(cls, file_name:Union[str, os.PathLike]):
        dataframe = pd.read_csv(file_name)
        return cls(dataframe)

class ThermophilicityStatsOutputList(UserList):
    def __hash__(self) -> int:
        return hash(tuple(hash(el) for el in self))

@dataclass
class ComparativePerplexityOutput():
    adapter_name: str
    model_perplexity: pd.Series
    base_model_perplexity: pd.Series
    hue: pd.Series

    def __hash__(self) -> int:
        return hash((
            self.adapter_name,
            hash(
                tuple(
                    self.model_perplexity.to_list() + \
                    self.base_model_perplexity.to_list()
                )
            )
        ))

    @cached_property
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "model_perplexity": self.model_perplexity, 
            "base_model_perplexity": self.base_model_perplexity, 
            "d_perplexity": self.base_model_perplexity - self.model_perplexity, 
            "hue": self.hue
        })

    @cached_property
    def perplexity_distributions(self) -> str:
        sns.jointplot(
            self.dataframe,
            x="model_perplexity",
            y="base_model_perplexity",
            hue="hue",
            kind="kde"
        )
        tmpfile = BytesIO()
        plt.xlim(0.5, 5)
        plt.ylim(0.5, 5)
        plt.gcf().set_size_inches(10, 10)
        plt.savefig(tmpfile, format='png', dpi=200)
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf8')
        plt.close()
        return encoded

    @cached_property
    def d_perplexity_distribution(self) -> str:
        sns.displot(self.dataframe, x="d_perplexity", hue="hue", kind="kde", fill=True)
        tmpfile = BytesIO()
        plt.xlim(-2.5, 2.5)
        plt.ylim(0, 2)
        plt.gcf().set_size_inches(10, 10)
        plt.savefig(tmpfile, format='png', dpi=200)
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf8')
        plt.close()
        return encoded

    @cached_property
    def description(self) -> pd.DataFrame:
        _prefix = ["model_perplexity", "d_perplexity"]
        _stats = ["mean", "25%", "50%", "75%"]
        columns = [(a, b) for a in _prefix for b in _stats]
        description = self.dataframe \
            .groupby("hue") \
            .describe() \
            .loc[:,columns] \
            .reset_index() \
            .rename({"index": "Thermophilicity"}, axis=1)
        description.columns = [
            "Thermophilicity",
            "Model Mean",
            "Model Q1",
            "Model Q2",
            "Model Q3",
            "&#916 Mean",
            "&#916 Q1",
            "&#916 Q2",
            "&#916 Q3"
        ]
        return description

    def to_html(self, a:Airium) -> None:
        description = self.description

        with a.div(klass="container"):
            with a.table(klass="table table-sm"):
                with a.thead():
                    with a.tr():
                        with a.th(scope="col"): a("Thermophilicity")
                        with a.th(scope="col"): a("Model Mean")
                        with a.th(scope="col"): a("Model Q1")
                        with a.th(scope="col"): a("Model Q2")
                        with a.th(scope="col"): a("Model Q3")
                        with a.th(scope="col"): a("&#916 Mean")
                        with a.th(scope="col"): a("&#916 Q1")
                        with a.th(scope="col"): a("&#916 Q2")
                        with a.th(scope="col"): a("&#916 Q3")
                with a.tbody():
                    for _, row in self.description.iterrows():
                        with a.tr():
                            with a.th(scope="col"): a(f"{row['Thermophilicity']}")
                            with a.td(scope="col"): a(f"{row['Model Mean']:.3f}")
                            with a.td(scope="col"): a(f"{row['Model Q1']:.3f}")
                            with a.td(scope="col"): a(f"{row['Model Q2']:.3f}")
                            with a.td(scope="col"): a(f"{row['Model Q3']:.3f}")
                            with a.td(scope="col"): a(f"{row['&#916 Mean']:.3f}")
                            with a.td(scope="col"): a(f"{row['&#916 Q1']:.3f}")
                            with a.td(scope="col"): a(f"{row['&#916 Q2']:.3f}")
                            with a.td(scope="col"): a(f"{row['&#916 Q3']:.3f}")

            with a.div(klass="row d-print-none"):
                with a.div(klass="col"):
                    with a.h5(): a("&#916;Perplexity distribution")
                    a.img(
                        src=f"data:image/png;base64,{self.d_perplexity_distribution}",
                        alt="",
                        klass="img-fluid"
                    )

                with a.div(klass="col"):
                    with a.h5(): a("Perplexity distributions")
                    a.img(
                        src=f"data:image/png;base64,{self.perplexity_distributions}",
                        alt="",
                        klass="img-fluid"
                    )

class ComparativePerplexityOutputList(UserList):
    def __hash__(self) -> int:
        return hash(tuple(hash(el) for el in self))

    @cached_property
    def description(self) -> pd.DataFrame:
        return pd.concat([el.description for el in self], axis=0, ignore_index=True)

    def join_attributes_to_description(self, attributes_list:BertAttributesList) -> pd.DataFrame:
        description = pd.concat([attributes_list.repeated_dataframe(), self.description], axis=1)
        description = description.sort_values([
            "Mask rate",
            "Model selection",
            "Learning rate",
            "Batch size",
            "Thermophilicity"
        ])
        return description

    def to_html(
        self,
        a:Airium,
        attributes_list:BertAttributesList,
        links: List[Union[str, os.PathLike]],
        n_repeats:int=2
    ) -> None:
        description = self.join_attributes_to_description(attributes_list)
        links = [link for link in links for _ in range(n_repeats)]
        links = [links[i] for i in description.index.to_list()]

        with a.table(klass="table table-sm table-striped"):
            with a.thead():
                with a.tr():
                    with a.th(scope="col"): a("#")
                    with a.th(scope="col"): a("Mask rate")
                    with a.th(scope="col"): a("Model selection")
                    with a.th(scope="col"): a("Learning rate")
                    with a.th(scope="col"): a("Batch size")
                    with a.th(scope="col"): a("Thermophilicity")
                    with a.th(scope="col"): a("Model Mean")
                    with a.th(scope="col"): a("Model Q1")
                    with a.th(scope="col"): a("Model Q2")
                    with a.th(scope="col"): a("Model Q3")
                    with a.th(scope="col"): a("&#916 Mean")
                    with a.th(scope="col"): a("&#916 Q1")
                    with a.th(scope="col"): a("&#916 Q2")
                    with a.th(scope="col"): a("&#916 Q3")
            with a.tbody():
                for i, (_, row) in enumerate(description.iterrows()):
                    with a.tr():
                        with a.th(scope="col"):
                            with a.a(href=f"{links[i]}"): a(f"{i}")
                        with a.td(scope="col"): a(f"{row['Mask rate']}")
                        with a.td(scope="col"): a(f"{row['Model selection']}")
                        with a.td(scope="col"): a(f"{row['Learning rate']:.4f}")
                        with a.td(scope="col"): a(f"{row['Batch size']}")
                        with a.td(scope="col"): a(f"{row['Thermophilicity']}")
                        with a.td(scope="col"): a(f"{row['Model Mean']:.3f}")
                        with a.td(scope="col"): a(f"{row['Model Q1']:.3f}")
                        with a.td(scope="col"): a(f"{row['Model Q2']:.3f}")
                        with a.td(scope="col"): a(f"{row['Model Q3']:.3f}")
                        with a.td(scope="col"): a(f"{row['&#916 Mean']:.3f}")
                        with a.td(scope="col"): a(f"{row['&#916 Q1']:.3f}")
                        with a.td(scope="col"): a(f"{row['&#916 Q2']:.3f}")
                        with a.td(scope="col"): a(f"{row['&#916 Q3']:.3f}")

    def to_latex(
        self,
        attributes_list: BertAttributesList,
        index: Optional[bool]=False,
        float_format: Optional[typing.Callable]="{:.4f}".format,
        *args: Optional[typing.Any],
        **kwargs: Optional[typing.Any]
    ) -> str:
        description = self.join_attributes_to_description(attributes_list)
        description = description.rename(
            {
                "&#916 Mean": "$\Bar{\Delta_\text{base,ft}}$",
                "&#916 Q1": "$\Delta_\text{base,ft}_0.25$",
                "&#916 Q2": "$\Delta_\text{base,ft}_0.50$",
                "&#916 Q3": "$\Delta_\text{base,ft}_0.75$",
            },
            axis=1
        )
        return description.to_latex(index=index, float_format=float_format, *args, **kwargs)

    @classmethod
    def from_adapters(
        cls,
        from_adapters: List[Union[str, os.PathLike]],
        thermophilics_set: Union[str, os.PathLike],
        mesophilics_set: Union[str, os.PathLike],
        from_model: Optional[Union[str, os.PathLike]]="Rostlab/prot_bert_bfd",
        from_tokenizer: Optional[Union[str, os.PathLike]]="Rostlab/prot_bert_bfd",
        min_length: Optional[int]=None,
        max_length: Optional[int]=None,
        mask: Optional[bool]=False,
        p: Optional[float]=0,
        local_batch_size: Optional[int]=8,
        num_workers: Optional[int]=2
    ):
        if not isinstance(from_adapters, typing.Iterable):
            from_adapters = [from_adapters]

        def get_model_perplexity(model, thermophilics_dataloader, mesophilics_dataloader):
            model_perplexity = torch.empty(0)

            for batch in thermophilics_dataloader:
                input_ids = batch.input_ids
                attention_mask = batch.attention_mask
                perp = perplexity(
                    model=model,
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device)
                )
                model_perplexity = torch.concat([model_perplexity, perp.cpu()])

            for batch in mesophilics_dataloader:
                input_ids = batch.input_ids
                attention_mask = batch.attention_mask
                perp = perplexity(
                    model=model,
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device)
                )
                model_perplexity = torch.concat([model_perplexity, perp.cpu()])

            model_perplexity = pd.Series(model_perplexity)
            return model_perplexity

        base_model = AutoModelForMaskedLM.from_pretrained(from_model)
        base_model.to(device).to(torch.bfloat16)

        set_args = {
            "from_tokenizer": from_tokenizer,
            "min_length": min_length,
            "max_length": max_length,
            "mask": mask,
            "p": p,
            "local_batch_size":
            local_batch_size,
            "num_workers": num_workers
        }
        thermophilics_dataloader = get_dataloaders(
            {"training_set": thermophilics_set, **set_args},
            shuffle=False,
            return_validation=False
        )
        mesophilics_dataloader = get_dataloaders(
            {"training_set": mesophilics_set, **set_args},
            shuffle=False,
            return_validation=False
        )

        hue = pd.Series(
            ["thermophilics"]*len(thermophilics_dataloader.dataset) + \
            ["mesophilics"]*len(mesophilics_dataloader.dataset)
        )

        base_model_perplexity = get_model_perplexity(
            base_model,
            thermophilics_dataloader,
            mesophilics_dataloader
        )

        comparative_perplexity_list = cls()
        for el in tqdm.tqdm(from_adapters):
            model = get_model({"from_model": from_model, "from_adapters": el})
            model.to(device).to(torch.bfloat16)

            model_perplexity = get_model_perplexity(
                model,
                thermophilics_dataloader,
                mesophilics_dataloader
            )
            comparative_perplexity_list.append(
                ComparativePerplexityOutput(el, model_perplexity, base_model_perplexity, hue)
            )

        return comparative_perplexity_list

def formatter(fun:typing.Callable) -> typing.Callable:
    def wrapper(*args:typing.Any, **kwargs:typing.Any) -> str:
        a = Airium()
        a('<!DOCTYPE html>')

        with a.html(lang="en"):
            with a.head():
                a.meta(charset="utf-8")
                a.meta(name="viewport", content="width=device-width, initial-scale=1")
                a.title(_t="Mutator Analysis")
                a.link(
                    href=BOOTSTRAP_CSS,
                    rel="stylesheet"
                )

            with a.body():
                with a.h1(): a("Mutator Analysis")

                fun(a, *args, **kwargs)

                a.script(
                    src=BOOTSTRAP_JS
                )

        html = str(a)
        html = html.replace("__", "-")
        return html

    return wrapper

def _accordion_item_formatter(
    a: Airium,
    id: str,
    title: str,
    content: typing.Any,
    *args: typing.Any,
    **kwargs: typing.Any
) -> None:
    with a.div(klass="accordion-item"):
        with a.h2(klass="accordion-header"):
            with a.button(
                klass="accordion-button",
                type="button",
                data__bs__toggle="collapse",
                data__bs__target=f"#panelsStayOpen-{id}",
                aria__expanded="true",
                aria__controls=f"panelsStayOpen-{id}"
            ):
                a(title)
        with a.div(id=f"panelsStayOpen-{id}", klass="accordion-collapse collapse show"):
            with a.div(klass="accordion-body"):
                content.to_html(a, *args, **kwargs)

@formatter
def bert_summary_formatter(
    a: Airium,
    comparative_perplexity: ComparativePerplexityOutput, 
    bert_attributes: BertAttributes 
) -> None:
    with a.div(klass="container"):
        with a.div(klass="accordion"):
            _accordion_item_formatter(
                a,
                "bert_attributes",
                "Bert Attributes",
                bert_attributes
            )
            _accordion_item_formatter(
                a,
                "comparative_perplexity",
                "Comparative Perplexity",
                comparative_perplexity
            )

@formatter
def summary_formatter(
    a: Airium,
    p_stats: PerplexityStatsOutput,
    m_stats: MutationStatsOutput,
    t_stats: ThermophilicityStatsOutput,
    attributes: InferenceAttributes
) -> None:
    with a.ul(klass="nav justify-content-end"):
        with a.li(klass="nav-item"):
            with a.a(klass="nav-link", href="/".join([".."]*6) + "/index.html"): a("Index")
        with a.li(klass="nav-item"):
            with a.a(klass="nav-link", href="/".join([".."]*5) + "/index.html"): a("Mask rate")
        with a.li(klass="nav-item"):
            with a.a(klass="nav-link", href="/".join([".."]*4) + "/index.html"): a("Model selection")
        with a.li(klass="nav-item"):
            with a.a(klass="nav-link", href="/".join([".."]*3) + "/index.html"): a("Learning rate")
        with a.li(klass="nav-item"):
            with a.a(klass="nav-link", href="/".join([".."]*2) + "/index.html"): a("Batch size")
        with a.li(klass="nav-item"):
            with a.a(klass="nav-link", href="/".join([".."]*1) + "/index.html"): a("N. iterations")

    with a.div(klass="container"):
        with a.div(klass="accordion"):
            _accordion_item_formatter(
                a,
                "inference_attributes",
                "Inference Attributes",
                attributes
            )
            _accordion_item_formatter(
                a,
                "perplexity_stats",
                "Perplexity Statistics",
                p_stats
            )
            _accordion_item_formatter(
                a,
                "mutations_stats",
                "Mutation Statistics",
                m_stats
            )

            if t_stats is not None:
                _accordion_item_formatter(
                    a,
                    "thermophilicity_stats",
                    "Thermophilicity Statistics",
                    t_stats
                )

@formatter
def index_formatter(
    a: Airium,
    comparative_perplexity_list: ComparativePerplexityOutputList=None,
    p_stats_list: PerplexityStatsOutputList=None,
    m_stats_list: MutationStatsOutputList=None,
    bert_attributes_list: BertAttributesList=None,
    attributes_list: InferenceAttributesList=None,
    bert_links: List[Union[str, os.PathLike]]=None,
    links: List[Union[str, os.PathLike]]=None
) -> None:
    with a.div(klass="container"):
        with a.div(klass="accordion"):
            if comparative_perplexity_list is not None:
                if bert_attributes_list is None:
                    raise ValueError()
                if bert_links is None:
                    raise ValueError()

                _accordion_item_formatter(
                    a,
                    "comparative_perplexity",
                    "Comparative Perplexity",
                    comparative_perplexity_list,
                    bert_attributes_list,
                    bert_links
                )
            if p_stats_list is not None:
                if attributes_list is None:
                    raise ValueError()
                if links is None:
                    raise ValueError()

                _accordion_item_formatter(
                    a,
                    "perplexity_stats",
                    "Perplexity Statistics",
                    p_stats_list,
                    attributes_list,
                    links
                )
            if m_stats_list is not None:
                _accordion_item_formatter(
                    a,
                    "mutations_stats",
                    "Mutation Statistics",
                    m_stats_list
                )

def safe_writer(content:str, path: Union[str, os.PathLike]) -> None:
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    with open(path, "w", encoding="UTF-8") as f:
        f.write(content)

def bert_summary(
    bert_adapters_template: Union[str, os.PathLike],
    thermophilics_set: Union[str, os.PathLike],
    mesophilics_set: Union[str, os.PathLike],
    out_dir: Optional[Union[str, os.PathLike]]="./html"   
) -> Tuple[
        ComparativePerplexityOutputList,
        BertAttributesList
    ]:
    adapters = [el for el in glob.glob(bert_adapters_template) if os.path.isdir(el)]
    bert_attributes_list = BertAttributesList.from_adapters(adapters)
    comparative_perplexity_list = ComparativePerplexityOutputList.from_adapters(
        from_adapters=adapters,
        thermophilics_set=thermophilics_set,
        mesophilics_set=mesophilics_set
    )

    for comparative_perplexity, bert_attributes in zip(comparative_perplexity_list, bert_attributes_list):
        safe_writer(
            bert_summary_formatter(comparative_perplexity, bert_attributes),
            bert_attributes.get_output_file_name("bert_summary.html", out_dir)
        )

    return comparative_perplexity_list, bert_attributes_list

def summary(
    file_name: Union[str, os.PathLike],
    base_file_name: Union[str, os.PathLike],
    thermophilicity_stats_file: Optional[Union[str, os.PathLike]]=None,
    out_dir="./html"
) -> Tuple[
        PerplexityStatsOutput,
        MutationStatsOutput,
        ThermophilicityStatsOutput,
        InferenceAttributes
    ]:

    attributes = InferenceAttributes.from_file_name(file_name)
    p_stats = PerplexityStatsOutput.from_file(file_name)
    m_stats = MutationStatsOutput.from_files(file_name, base_file_name)

    t_stats = None
    if thermophilicity_stats_file is not None:
        t_stats = ThermophilicityStatsOutput.from_file(thermophilicity_stats_file)

    safe_writer(
        summary_formatter(p_stats, m_stats, t_stats, attributes),
        attributes.get_output_file_name("summary.html", out_dir)
    )

    return p_stats, m_stats, t_stats, attributes

def multiple_summary(
    file_name_template: Union[str, os.PathLike],
    base_file_name: Union[str, os.PathLike],
    bert_adapters_template: Union[str, os.PathLike],
    thermophilics_set: Union[str, os.PathLike],
    mesophilics_set: Union[str, os.PathLike],
    out_dir: Optional[Union[str, os.PathLike]]="./html"
) -> None:
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    attributes_list = InferenceAttributesList()
    p_stats_list = PerplexityStatsOutputList()
    m_stats_list = MutationStatsOutputList()
    t_stats_list = ThermophilicityStatsOutputList()

    for file_name in tqdm.tqdm(glob.glob(file_name_template)):
        thermophilicity_stats_file = file_name[:-4] + ".stats.csv"
        if not os.path.isfile(thermophilicity_stats_file):
            thermophilicity_stats_file = None

        p_stats, m_stats, t_stats, attributes = summary(
            file_name,
            base_file_name,
            thermophilicity_stats_file,
            out_dir
        )

        attributes_list.append(attributes)
        p_stats_list.append(p_stats)
        m_stats_list.append(m_stats)
        t_stats_list.append(t_stats)

    for level, indices in attributes_list.get_levels().items():
        attributes_subset = InferenceAttributesList(
            [attributes_list[i] for i in indices]
        )
        p_stats_subset = PerplexityStatsOutputList(
            [p_stats_list[i] for i in indices]
        )
        m_stats_subset = MutationStatsOutputList(
            [m_stats_list[i] for i in indices]
        )

        skip = None
        if isinstance(level, tuple):
            skip = len(level)
            level = "/".join([str(el) for el in level])
        else:
            skip = 1
            level = str(level)

        safe_writer(
            index_formatter(
                p_stats_list = p_stats_subset,
                m_stats_list = m_stats_subset,
                attributes_list = attributes_subset,
                links = attributes_subset.get_output_file_names("summary.html", skip=skip)
            ),
            os.path.join(out_dir, level, "index.html")
        )

    (comparative_perplexity_list,
     bert_attributes_list) = bert_summary(bert_adapters_template,
                                                 thermophilics_set,
                                                 mesophilics_set,
                                                 out_dir) 

    safe_writer(
        index_formatter(
            comparative_perplexity_list,
            p_stats_list,
            m_stats_list,
            bert_attributes_list,
            attributes_list,
            bert_attributes_list.get_output_file_names("bert_summary.html"),
            attributes_list.get_output_file_names("summary.html")
        ),
        os.path.join(out_dir, "index.html")
    )

    safe_writer(
        comparative_perplexity_list.to_latex(bert_attributes_list),
        os.path.join(out_dir, "comparative_perplexity.tex")
    )

    safe_writer(
        p_stats_list.to_latex(attributes_list),
        os.path.join(out_dir, "perplexity_statistics.tex")
    )

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    multiple_summary(
        sys.argv[1],
        sys.argv[2],
        bert_adapters_template=sys.argv[3],
        thermophilics_set=sys.argv[4],
        mesophilics_set=sys.argv[5]
    )
