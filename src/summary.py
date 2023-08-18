import sys
import os
import typing
from dataclasses import dataclass
from collections import UserList
import tqdm
import glob
import re

import pandas as pd
import numpy as np
import scipy as sp
import torch

import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

from transformers import AutoModelForMaskedLM

from airium import Airium

from statistics import perplexity
from getters import get_model, get_dataloaders

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@dataclass
class InferenceAttributes():
    n_iter: int=0
    mode: str=None
    lr: float=None
    bs: int=None
    p: float=None

    def to_html(self, a):
        with a.ul(klss="list-group list-group-flush"):
            with a.li(klass="list_group-item"):
                a(f"Model selection: {self.mode}")
            with a.li(klass="list_group-item"):
                a(f"Learning rate: {self.lr}")
            with a.li(klass="list_group-item"):
                a(f"Batch size: {self.bs}")
            with a.li(klass="list_group-item"):
                a(f"Mask rate: {self.p}")
            if self.n_iter != 0:
                with a.li(klass="list_group-item"):
                    a(f"Random inference  with {self.n_iter} iterations")
    
    @classmethod
    def from_file_name(
        cls, 
        file_name,
        random_pattern = r"\.\/out\/inference.random.(?P<n_iter>\d+).(?P<mode>(final)|(validation)).LR(?P<lr>.*)_BS(?P<bs>.*)_P(?P<p>.*).csv",
        deterministic_pattern = r"\.\/out\/inference.(?P<mode>(final)|(validation)).LR(?P<lr>.*)_BS(?P<bs>.*)_P(?P<p>.*).csv"
    ):
        if attributes := re.match(random_pattern, file_name):
            attributes = attributes.groupdict()
        elif attributes := re.match(deterministic_pattern, file_name):
            attributes = attributes.groupdict()
            attributes["n_iter"] = 0
        else:
            raise ValueError()

        return cls(**attributes)

class InferenceAttributesList(UserList):
    @property
    def n_iter(self):
        return [el.n_iter for el in self]

    @property
    def mode(self):
        return [el.mode for el in self]

    @property
    def lr(self):
        return [el.lr for el in self]

    @property
    def bs(self):
        return [el.bs for el in self]

    @property
    def p(self):
        return [el.p for el in self]

    @property
    def dataframe(self):
        dataframe = pd.DataFrame({
            "Model selection": self.mode,
            "Learning rate": self.lr,
            "Batch size": self.bs,
            "Mask rate": self.p,
            "N. iterations": self.n_iter
        })
        dataframe = dataframe.astype({"Model selection":str, "Learning rate":float, "Batch size":int, "Mask rate":float, "N. iterations":int})
        return dataframe

@dataclass
class PerplexityStatsOutput():
    perplexity: pd.Series
    base_perplexity: pd.Series

    @property
    def pcc(self):
        return sp.stats.pearsonr(self.base_perplexity, self.perplexity).statistic
    
    @property
    def d_perplexity(self):
        return self.base_perplexity - self.perplexity
    
    def d_quantile(self, q):
        if isinstance(q, typing.Iterable):
            return [self.d_quantile(el) for el in q]
        return self.d_perplexity.quantile(q)
    
    @property
    def d_mean(self):
        return self.d_perplexity.mean()

    @property
    def perplexity_distributions(self):
        g = sns.jointplot(x=self.base_perplexity, y=self.perplexity, kind="kde")
        tmpfile = BytesIO()
        plt.gcf().set_size_inches(10, 10)
        plt.savefig(tmpfile, format='png', dpi=200)
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf8')
        plt.close()
        return encoded

    @property
    def d_perplexity_distribution(self):
        g = sns.displot(x=self.d_perplexity, kind="kde", fill=True)
        tmpfile = BytesIO()
        plt.gcf().set_size_inches(10, 10)
        plt.savefig(tmpfile, format='png', dpi=200)
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf8')
        plt.close()
        return encoded
    
    def to_html(self, a):
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
                    a.img(src=f"data:image/png;base64,{self.d_perplexity_distribution}", alt="", klass="img-fluid")
                with a.div(klass="col-sm"):
                    with a.h4(): a("Perplexity Distribution")
                    a.img(src=f"data:image/png;base64,{self.perplexity_distributions}", alt="", klass="img-fluid")

    @classmethod
    def from_file(cls, file_name):
        df = pd.read_csv(file_name)
        return cls(df["perplexity_src"], df["perplexity_mut"])

class PerplexityStatsOutputList(UserList):
    @property
    def pcc(self):
        return [el.pcc for el in self]
    
    def d_quantile(self, q):
        return [el.d_quantile(q) for el in self]
    
    @property
    def d_mean(self):
        return [el.d_mean for el in self]
    
    @property
    def dataframe(self):
        return pd.DataFrame({
            "Mean &#916;Perplexity": self.d_mean,
            "Q1 &#916;Perplexity": self.d_quantile(0.25),
            "Q2 &#916;Perplexity": self.d_quantile(0.50),
            "Q3 &#916;Perplexity": self.d_quantile(0.75),
            "PCC": self.pcc
        }, dtype=float)
    
    def join_attributes(self, attributes_list):
        dataframe = pd.concat([attributes_list.dataframe, self.dataframe], axis=1)
        dataframe = dataframe.sort_values(["Mask rate", "Model selection", "Learning rate", "Batch size", "N. iterations"])
        return dataframe
    
    def to_html(self, a, attributes_list, links):
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
                    with a.th(scope="col"): a("Mean &#916;Perplexity")
                    with a.th(scope="col"): a("Q1 &#916;Perplexity")
                    with a.th(scope="col"): a("Q2 &#916;Perplexity")
                    with a.th(scope="col"): a("Q3 &#916;Perplexity")
                    with a.th(scope="col"): a("PCC")
            with a.tbody():
                for i, (_, row) in enumerate(self.join_attributes(attributes_list).iterrows()):
                    with a.tr():
                        with a.th(scope="col"): 
                            with a.a(href=f"{links[i]}"): a(f"{i}")
                        with a.td(scope="col"): a(f"{row['Model selection']}")
                        with a.td(scope="col"): a(f"{row['Learning rate']:.4f}")
                        with a.td(scope="col"): a(f"{row['Batch size']}")
                        with a.td(scope="col"): a(f"{row['Mask rate']:.3f}")
                        with a.td(scope="col"): a(f"{row['N. iterations']}")
                        with a.td(scope="col"): a(f"{row['Mean &#916;Perplexity']:.3f}")
                        with a.td(scope="col"): a(f"{row['Q1 &#916;Perplexity']:.3f}")
                        with a.td(scope="col"): a(f"{row['Q1 &#916;Perplexity']:.3f}")
                        with a.td(scope="col"): a(f"{row['Q1 &#916;Perplexity']:.3f}")
                        with a.td(scope="col"): a(f"{row['PCC']:.3f}")

@dataclass
class MutationStatsOutput():
    sequences: pd.Series
    base_sequences: pd.Series

    @property
    def paired_sequences(self):
        return [(base_sequence, sequence) for base_sequence, sequence in zip(self.base_sequences, self.sequences) if len(base_sequence) == len(sequence)]
    
    @property
    def n_unpaired_sequences(self):
        return len(self.base_sequences)  - len(self.paired_sequences)
    
    @property
    def _nodes(self):
        return list(set([el for seq in self.sequences for el in seq] + [el for seq in self.base_sequences for el in seq]))

    @property
    def _edges(self):
        return [(base_sequence[i], sequence[i]) for base_sequence, sequence in self.paired_sequences for i in range(len(sequence))]
    
    @property
    def _adjacency_matrix(self):
        adjacency_matrix = {from_node: {to_node: 0 for to_node in self._nodes} for from_node in self._nodes}
        for from_node, to_node in self._edges: adjacency_matrix[from_node][to_node] += 1
        adjacency_matrix = np.array([list(row.values()) for row in adjacency_matrix.values()])
        adjacency_matrix = adjacency_matrix / adjacency_matrix.sum(axis=1)[:, None]
        np.fill_diagonal(adjacency_matrix, 0)
        return adjacency_matrix

    def top_from(self, n_top=10):
        return [self._nodes[i] for i in np.argsort(self._adjacency_matrix.sum(axis=1))[-n_top:]][::-1]

    def top_to(self, n_top=10):
        return [self._nodes[i] for i in np.argsort(self._adjacency_matrix.sum(axis=0))[-n_top:]][::-1]

    def top_edges(self, n_top=20):
        top_edges = list(zip(list(self._adjacency_matrix.flatten().argsort()[-n_top:] // len(self._adjacency_matrix)), list(self._adjacency_matrix.flatten().argsort()[-n_top:] % len(self._adjacency_matrix))))[::-1]
        top_edges = [(self._nodes[from_node], self._nodes[to_node], self._adjacency_matrix[from_node, to_node ]) for from_node, to_node in top_edges]
        return top_edges

    @property
    def clustermap(self):
        g = sns.clustermap(self._adjacency_matrix, xticklabels=self._nodes, yticklabels=self._nodes)
        tmpfile = BytesIO()
        plt.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf8')
        plt.close()
        return encoded
    
    def to_html(self, a):
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
    def from_files(cls, file_name, base_file_name):
        df = pd.read_csv(file_name)
        base_df = pd.read_csv(base_file_name)
        return MutationStatsOutput(df.iloc[:, -1], base_df.iloc[:, -1])

class MutationStatsOutputList(UserList):
    @staticmethod
    def _top_list(top_lists, fun=lambda x: x):
        scores = list(range(len(top_lists[0]), 0, -1)) * len(top_lists)
        top_list = [el for top_list in top_lists for el in top_list]

        cum_score = {el:0 for el in set(top_list)}
        for el, score in zip(top_list, scores): cum_score[el] += score

        top_list = sorted(cum_score.items(), key=lambda x:x[1], reverse=True)
        top_list = [fun(el) for el, _ in top_list[:len(top_lists[0])]]

        return top_list

    def top_from(self, n_top=10, process=lambda x: x):
        return self._top_list([process(el.top_from(n_top)) for el in self])

    def top_to(self, n_top=10, process=lambda x: x):
        return self._top_list([process(el.top_to(n_top)) for el in self])

    def top_edges(self, n_top=20, process=lambda x: [f"{f}{t}" for f, t, _ in x]):
        return self._top_list([process(el.top_edges(n_top)) for el in self], fun=lambda x: (x[0], x[1]))

    def to_html(self, a):
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
    df: pd.DataFrame

    @property
    def description(self):
        return self.df.describe().loc[["mean", "25%", "50%", "75%"],:].T
    
    @property
    def distributions(self):
        g = sns.pairplot(self.df)
        tmpfile = BytesIO()
        plt.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf8')
        plt.close()
        return encoded
    
    def to_html(self, a):
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
        
        a.img(src=f"data:image/png;base64,{self.distributions}", alt="", klass="img-fluid")

    @classmethod
    def from_file(cls, file_name):
        df = pd.read_csv(file_name)
        return cls(df)

class ThermophilicityStatsOutputList(UserList):
    pass

@dataclass
class ComparativePerplexityOutput():
    adapter_name: str
    model_perplexity: pd.Series
    base_model_perplexity: pd.Series
    hue: pd.Series

    @property
    def df(self):
        return pd.DataFrame({
            "model_perplexity": self.model_perplexity, 
            "base_model_perplexity": self.base_model_perplexity, 
            "d_perplexity": self.base_model_perplexity - self.model_perplexity, 
            "hue": self.hue
        })
    
    @property
    def perplexity_distributions(self):
        g = sns.jointplot(self.df, x="model_perplexity", y="base_model_perplexity", hue="hue", kind="kde")
        tmpfile = BytesIO()
        plt.xlim(0.5, 5)
        plt.ylim(0.5, 5)
        plt.gcf().set_size_inches(10, 10)
        plt.savefig(tmpfile, format='png', dpi=200)
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf8')
        plt.close()
        return encoded

    @property
    def d_perplexity_distribution(self):
        g = sns.displot(self.df, x="d_perplexity", kind="kde", fill=True)
        tmpfile = BytesIO()
        plt.xlim(-2.5, 2.5)
        plt.ylim(0, 2)
        plt.gcf().set_size_inches(10, 10)
        plt.savefig(tmpfile, format='png', dpi=200)
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf8')
        plt.close()
        return encoded
    
    @property
    def description(self):
        columns = [(a, b) for a in ["model_perplexity", "d_perplexity"] for b in ["mean", "25%", "50%", "75%"]]
        return self.df.groupby("hue").describe().loc[:,columns]
    
    def to_html(self, a):
        with a.div(klass="container"):
            with a.h4(): a(self.adapter_name)
    
            with a.table(klass="table table-sm"):
                with a.thead():
                    with a.tr():
                        with a.th(scope="col"): a("#")
                        with a.th(scope="col"): a("Model Mean")
                        with a.th(scope="col"): a("Model Q1")
                        with a.th(scope="col"): a("Model Q2")
                        with a.th(scope="col"): a("Model Q3")
                        with a.th(scope="col"): a("&#916 Mean")
                        with a.th(scope="col"): a("&#916 Q1")
                        with a.th(scope="col"): a("&#916 Q2")
                        with a.th(scope="col"): a("&#916 Q3")
                with a.tbody():
                    for i, row in self.description.iterrows():
                        with a.tr():
                            with a.th(scope="col"): a(i)
                            for value in row.values:
                                with a.td(scope="col"): a(f"{value:.3f}")

            with a.div(klass="row"):
                with a.div(klass="col"):
                    with a.h5(): a("&#916;Perplexity distribution")
                    a.img(src=f"data:image/png;base64,{self.d_perplexity_distribution}", alt="", klass="img-fluid")

                with a.div(klass="col"):
                    with a.h5(): a("Perplexity distributions")
                    a.img(src=f"data:image/png;base64,{self.perplexity_distributions}", alt="", klass="img-fluid")


class ComparativePerplexityOutputList(UserList):
    def to_html(self, a):
        with a.div(klass="container"):
            n_items = len(self)

            for i in range(0, n_items + 1, 2):
                with a.div(klass="row"):
                    if i < n_items:
                        with a.div(klass="col"):
                            self[i].to_html(a)
                    if i+1 < n_items:
                        with a.div(klass="col"):
                            self[i+1].to_html(a)

    @classmethod
    def from_adapters(
        cls,
        from_adapters, 
        thermophilics_set, 
        mesophilics_set, 
        from_model="Rostlab/prot_bert_bfd", 
        from_tokenizer="Rostlab/prot_bert_bfd",
        min_length=None,
        max_length=None,
        mask=False,
        p=0,
        local_batch_size=8,
        num_workers=2
    ):
        if not isinstance(from_adapters, typing.Iterable):
            from_adapters = [from_adapters]

        def get_model_perplexity(model, thermophilics_dataloader, mesophilics_dataloader):
            model_perplexity = torch.empty(0)

            for batch in thermophilics_dataloader:
                input_ids = batch.input_ids
                attention_mask = batch.attention_mask
                perp = perplexity(model=model, input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
                model_perplexity = torch.concat([model_perplexity, perp.cpu()])

            for batch in mesophilics_dataloader:
                input_ids = batch.input_ids
                attention_mask = batch.attention_mask
                perp = perplexity(model=model, input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
                model_perplexity = torch.concat([model_perplexity, perp.cpu()])

            model_perplexity = pd.Series(model_perplexity)
            return model_perplexity

        base_model = AutoModelForMaskedLM.from_pretrained(from_model)
        base_model.to(device).to(torch.bfloat16)

        set_args = {"from_tokenizer": from_tokenizer, "min_length": min_length, "max_length": max_length, "mask": mask, "p": p, "local_batch_size": local_batch_size, "num_workers": num_workers}
        thermophilics_dataloader = get_dataloaders({"training_set": thermophilics_set, **set_args}, shuffle=False, return_validation=False)
        mesophilics_dataloader = get_dataloaders({"training_set": mesophilics_set, **set_args}, shuffle=False, return_validation=False)

        hue = pd.Series(["thermophilics"]*len(thermophilics_dataloader.dataset) + ["mesophilics"]*len(mesophilics_dataloader.dataset))

        base_model_perplexity = get_model_perplexity(base_model, thermophilics_dataloader, mesophilics_dataloader)

        comparative_perplexity_list = cls()
        for el in tqdm.tqdm(from_adapters):
            model = get_model({"from_model": from_model, "from_adapters": el})
            model.to(device).to(torch.bfloat16)

            model_perplexity = get_model_perplexity(model, thermophilics_dataloader, mesophilics_dataloader)
            comparative_perplexity_list.append(ComparativePerplexityOutput(el, model_perplexity, base_model_perplexity, hue))
        
        return comparative_perplexity_list

def formatter(fun):
    def wrapper(*args, **kwargs):
        a = Airium()
        a('<!DOCTYPE html>')

        with a.html(lang="en"):
            with a.head():
                a.meta(charset="utf-8")
                a.meta(name="viewport", content="width=device-width, initial-scale=1")
                a.title(_t="Mutator Analysis")
                a.link(href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css", rel="stylesheet", integrity="sha384-4bw+/aepP/YC94hEpVNVgiZdgIC5+VKNBQNGCHeKRQN+PtmoHDEXuppvnDJzQIu9", crossorigin="anonymous")
            
            with a.body():
                with a.h1(): a(f"Mutator Analysis")

                fun(a, *args, **kwargs)

                a.script(src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/js/bootstrap.bundle.min.js", integrity="sha384-HwwvtgBNo3bZJJLYd8oVXjrBZt8cqVSpeBNS5n7C8IVInixGAoxmnlMuBnhbgrkm", crossorigin="anonymous")

        html = str(a)
        html = html.replace("__", "-")
        return html
    
    return wrapper

def _accordion_item_formatter(a, id, title, content, *args, **kwargs):
    with a.div(klass="accordion-item"):
        with a.h2(klass="accordion-header"):
            with a.button(klass="accordion-button", type="button", data__bs__toggle="collapse", data__bs__target=f"#panelsStayOpen-{id}", aria__expanded="true", aria__controls=f"panelsStayOpen-{id}"):
                a(title)
        with a.div(id=f"panelsStayOpen-{id}", klass="accordion-collapse collapse show"):
            with a.div(klass="accordion-body"):
                content.to_html(a, *args, **kwargs)

@formatter
def summary_formatter(a, p_stats, m_stats, t_stats, attributes):
    with a.div(klass="container"):
        with a.div(klass="accordion"):
            _accordion_item_formatter(a, "inference_attributes", "Inference Attributes", attributes)
            _accordion_item_formatter(a, "perplexity_stats", "Perplexity Statistics", p_stats)
            _accordion_item_formatter(a, "mutations_stats", "Mutation Statistics", m_stats)

            if t_stats is not None:
                _accordion_item_formatter(a, "thermophilicity_stats", "Thermophilicity Statistics", t_stats)

@formatter
def index_formatter(a, comparative_perplexity_list, p_stats_list, m_stats_list, attributes_list, links):
    with a.div(klass="container"):
        with a.div(klass="accordion"):
            _accordion_item_formatter(a, "comparative_perplexity", "Comparative Perplexity", comparative_perplexity_list)
            _accordion_item_formatter(a, "perplexity_stats", "Perplexity Statistics", p_stats_list, attributes_list, links)
            _accordion_item_formatter(a, "mutations_stats", "Mutation Statistics", m_stats_list)

def summary(file_name, base_file_name, thermophilicity_stats_file=None):
    attributes = InferenceAttributes.from_file_name(file_name)
    p_stats = PerplexityStatsOutput.from_file(file_name)
    m_stats = MutationStatsOutput.from_files(file_name, base_file_name)

    t_stats = None
    if thermophilicity_stats_file is not None:
        t_stats = ThermophilicityStatsOutput.from_file(thermophilicity_stats_file)

    html = summary_formatter(p_stats, m_stats, t_stats, attributes)

    return html, p_stats, m_stats, t_stats, attributes

def multiple_summary(file_name_template, base_file_name, bert_adapters_template, thermophilics_set, mesophilics_set, out_dir="./html"):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    links = []
    attributes_list = InferenceAttributesList()
    p_stats_list = PerplexityStatsOutputList()
    m_stats_list = MutationStatsOutputList()
    t_stats_list = ThermophilicityStatsOutputList()

    for file_name in tqdm.tqdm(glob.glob(file_name_template)):
        thermophilicity_stats_file = file_name[:-4] + ".stats.csv"
        if not os.path.isfile(thermophilicity_stats_file):
            thermophilicity_stats_file = None
    
        html, p_stats, m_stats, t_stats, attributes = summary(file_name, base_file_name, thermophilicity_stats_file)
    
        links.append(os.path.join("./", os.path.basename(file_name)[:-4] + ".html"))
        attributes_list.append(attributes)
        p_stats_list.append(p_stats)
        m_stats_list.append(m_stats)
        t_stats_list.append(t_stats)
    
        with open(os.path.join(out_dir, os.path.basename(file_name)[:-4] + ".html"), "w") as f:
            f.write(html)

    comparative_perplexity_list = ComparativePerplexityOutputList.from_adapters(
        from_adapters=[el for el in glob.glob(bert_adapters_template) if os.path.isdir(el)],
        thermophilics_set=thermophilics_set, 
        mesophilics_set=mesophilics_set
    )
   
    html = index_formatter(
        comparative_perplexity_list, 
        p_stats_list, 
        m_stats_list, 
        attributes_list, 
        links
    )
    
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(html)
    
    with open(os.path.join(out_dir, "index.tex"), "w") as f:
        f.write(p_stats_list.join_attributes(attributes_list).to_latex(index=False, float_format="{:.4f}".format))

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    multiple_summary(sys.argv[1], sys.argv[2], bert_adapters_template=sys.argv[3], thermophilics_set=sys.argv[4], mesophilics_set=sys.argv[5])