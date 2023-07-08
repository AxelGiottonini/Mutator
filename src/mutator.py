import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from transformers import AutoTokenizer, AutoModelForMaskedLM

from .genetic_algorithm import GeneticModel
from .utils import no_grad

class Mutator(GeneticModel):
    model = None
    tokenizer = None

    def __init__(self):
        super().__init__()

        if Mutator.model is None:
            raise RuntimeError("Mutator model is undefined, please define a model using Mutator.set_model(model).")

        if Mutator.tokenizer is None:
            raise RuntimeError("Mutator tokenizer is undefined, please define a tokenizer using Mutator.set_tokenizer(tokenizer).")

        self.classifier = nn.Sequential(
            nn.Linear(Mutator.model.config.hidden_size, Mutator.model.config.hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(Mutator.model.config.hidden_size, 1, bias=False)
        )

    @no_grad
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        n_mutations: int, 
        k: int,
        cls_input: typing.Optional[torch.Tensor] = None
    ):
        # Get current sequence embeddings
        out = Mutator.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        embeddings = out.hidden_states[-1]

        # Compute the sequences mask from the embeddings
        logits = self.classifier(embeddings)[:,:,0]
        logits = attention_mask * logits + (1-attention_mask) * -1e6
        probs = F.softmax(logits, dim=-1)
        mutations = Categorical(probs=probs).sample(sample_shape=torch.Size([1, n_mutations]))[0,:,:].T
        mutations_mask = (F.one_hot(mutations, num_classes=logits.shape[-1]).sum(axis=1) > 0).long()

        # Mask the input the input ids using the mutations mask
        masked_input_ids = ((1-mutations_mask)*input_ids) + (mutations_mask*Mutator.tokenizer.mask_token_id)

        # Predict the masked ids
        out = Mutator.model(input_ids=masked_input_ids, attention_mask=attention_mask)
        top_k_ids = (-out.logits).argsort(dim=-1)[:,:,:k]
        is_in_top_k_ids = ((input_ids[:,:,None] - top_k_ids == 0).sum(axis=-1) > 0).long()
        mutated_ids = (1-mutations_mask)*input_ids + mutations_mask*(is_in_top_k_ids*input_ids + (1-is_in_top_k_ids)*top_k_ids[:,:,0])
        
        # Compute the mutated sequence pseudo-perplexity
        out = Mutator.model(input_ids=mutated_ids, attention_mask=attention_mask, output_hidden_states=True)
        log_probs = F.log_softmax(out.logits, dim=-1)
        perplexity = (-((F.one_hot(mutated_ids, num_classes=30) * log_probs).sum(axis=-1) * attention_mask).sum(axis=-1) / attention_mask.sum(axis=-1)).exp()

        # CLS token distance
        cls_input = embeddings[:,0,:] if cls_input is None else cls_input
        cls_output = out.hidden_states[-1][:,0,:]
        cls_distance = F.mse_loss(cls_output, cls_input, reduction='none').mean(axis=-1)

        out = {
            "mutated_ids": mutated_ids,
            "perplexity": perplexity,
            "cls_distance": cls_distance
        }
        out = type('',(object,), out)()

        return out

    @classmethod
    def set_model(cls, model:AutoModelForMaskedLM):
        cls.model = model

    @classmethod
    def set_tokenizer(cls, tokenizer:AutoTokenizer):
        cls.tokenizer = tokenizer
