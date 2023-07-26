import typing
from dataclasses import dataclass
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from transformers import AutoTokenizer, AutoModelForMaskedLM

from .genetic_algorithm import GeneticModel
from .statistics import perplexity
from .utils import no_grad

@dataclass
class MutatorOutput():
    input_ids: typing.Optional[torch.Tensor]=None,
    input_embeddings: typing.Optional[torch.Tensor]=None,
    mutated_ids: typing.Optional[torch.Tensor]=None,
    mutated_embeddings: typing.Optional[torch.Tensor]=None,
    attention_mask: typing.Optional[torch.Tensor]=None,
    mutation_mask: typing.Optional[torch.Tensor]=None,
    mutated_perplexity: typing.Optional[torch.Tensor]=None,
    input_cls: typing.Optional[torch.Tensor]=None,
    mutated_cls: typing.Optional[torch.Tensor]=None,
    cls_distance: typing.Optional[torch.Tensor]=None
    
    @classmethod
    def to_fitness(cls, obj, p_coef=1, d_coef=0, *args, **kwargs):
        if isinstance(obj, Iterable):
            return torch.tensor([cls.to_fitness(el, p_coef, d_coef, *args, **kwargs) for el in obj])
        
        mutated_perplexity = obj.mutated_perplexity.mean()
        cls_distance = obj.cls_distance.mean()
        return p_coef * mutated_perplexity + d_coef * cls_distance

class MutatorBase():
    model = None
    tokenizer = None
    n_mutations = None
    k = None

    def __init__(self):
        if self.model is None:
            raise RuntimeError("Mutator model is undefined, please define a model using Mutator.set_model(model).")
        if self.tokenizer is None:
            raise RuntimeError("Mutator tokenizer is undefined, please define a tokenizer using Mutator.set_tokenizer(tokenizer).")
        if self.n_mutations is None:
            raise RuntimeError("Mutator n_mutations is undefined, please define the number of mutations using Mutator.set_n_mutations(n).")
        if self.n_mutations is None:
            raise RuntimeError("Mutator k is undefined, please define the number of mutations using Mutator.set_k(k).")

    @no_grad
    def forward(
        self, 
        input_ids: typing.Optional[torch.Tensor]=None,
        input_embeddings: typing.Optional[torch.Tensor]=None,
        input_cls: typing.Optional[torch.Tensor]=None,
        attention_mask: typing.Optional[torch.Tensor]=None,
        mutator_output: typing.Optional[MutatorOutput]=None,
        *args: typing.Any,
        **kwargs: typing.Any
    ):
        if mutator_output is None:            
            if attention_mask is None:
                raise ValueError("Specify attention_mask")

            if input_embeddings is None:
                if input_ids is None:
                    raise ValueError("Specify either input_embeddings either input_ids")
                input_embeddings = Mutator.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]

            if input_cls is None:
                input_cls = input_embeddings[:,0,:]
        else:
            input_ids = mutator_output.mutated_ids
            input_embeddings = mutator_output.mutated_embeddings
            input_cls = mutator_output.input_cls
            attention_mask = mutator_output.attention_mask

        # Compute the sequences mask from the embeddings
        logits = self.logits_fn(input_ids, input_embeddings, input_cls, attention_mask)
        probs = F.softmax(logits, dim=-1)
        mutation = Categorical(probs=probs).sample(sample_shape=torch.Size([1, Mutator.n_mutations]))[0,:,:].T
        mutation_mask = (F.one_hot(mutation, num_classes=logits.shape[-1]).sum(axis=1) > 0).long()

        # Mask the input the input ids using the mutations mask
        masked_input_ids = ((1-mutation_mask)*input_ids) + (mutation_mask*Mutator.tokenizer.mask_token_id)

        # Predict the masked ids
        out = Mutator.model(input_ids=masked_input_ids, attention_mask=attention_mask)
        top_k_ids = (-out.logits).argsort(dim=-1)[:,:,:Mutator.k]
        is_in_top_k_ids = ((input_ids[:,:,None] - top_k_ids == 0).sum(axis=-1) > 0).long()
        mutated_ids = (1-mutation_mask)*input_ids + mutation_mask*(is_in_top_k_ids*input_ids + (1-is_in_top_k_ids)*top_k_ids[:,:,0])
        
        # Compute the mutated sequence pseudo-perplexity
        out = Mutator.model(input_ids=mutated_ids, attention_mask=attention_mask, output_hidden_states=True)
        mutated_embeddings = out.hidden_states[-1]
        mutated_perplexity = perplexity(
            model=Mutator.model, 
            input_ids=input_ids,
            logits=out.logits,
            attention_mask=attention_mask
        )

        # CLS token distance
        mutated_cls = out.hidden_states[-1][:,0,:]
        cls_distance = F.mse_loss(mutated_cls, input_cls, reduction='none').mean(axis=-1)

        out = MutatorOutput(
            input_ids=input_ids,
            input_embeddings=input_embeddings,
            mutated_ids=mutated_ids,
            mutated_embeddings=mutated_embeddings,
            attention_mask=attention_mask,
            mutation_mask=mutation_mask,
            mutated_perplexity=mutated_perplexity,
            input_cls=input_cls,
            mutated_cls=mutated_cls,
            cls_distance=cls_distance,
        )
        return out

    def logits_fn(self, input_ids, input_embeddings, input_cls, attention_mask):
        raise NotImplementedError()

    @classmethod
    def configure(cls, model, tokenizer, n_mutations, k, mutation_rate):
        cls.set_model(model)
        cls.set_tokenizer(tokenizer)
        cls.set_n_mutations(n_mutations)
        cls.set_k(k)
        cls.set_mutation_rate(mutation_rate)

    @classmethod
    def set_model(cls, model:AutoModelForMaskedLM):
        cls.model = model

    @classmethod
    def set_tokenizer(cls, tokenizer:AutoTokenizer):
        cls.tokenizer = tokenizer

    @classmethod
    def set_n_mutations(cls, n_mutations: int):
        cls.n_mutations = n_mutations

    @classmethod
    def set_k(cls, k: int):
        cls.k = k

class Mutator(MutatorBase, GeneticModel):
    model = None
    tokenizer = None
    n_mutations = None
    k = None

    def __init__(self, *args:typing.Any, **kwargs:typing.Any):
        super(MutatorBase, self).__init__()
        super(GeneticModel, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(Mutator.model.config.hidden_size, Mutator.model.config.hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(Mutator.model.config.hidden_size, 1, bias=False)
        )

    def logits_fn(self, input_ids, input_embeddings, input_cls, attention_mask):
        logits = self.classifier(input_embeddings)[:,:,0]
        logits = attention_mask * logits + (1-attention_mask) * -1e6
        return logits

class MutatorRandom(MutatorBase):
    def __init__(self, *args:typing.Any, **kwargs:typing.Any):
        super(MutatorBase, self).__init__()

    def logits_fn(self, input_ids, input_embeddings, input_cls, attention_mask):
        logits = attention_mask + (1-attention_mask) * -1e6
        return logits