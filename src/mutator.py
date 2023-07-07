import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from .genetic_algorithm import GeneticAlgorithm

class Mutator(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024, bias=False),
            nn.ReLU(),
            nn.Linear(1024, 1, bias=False)
        )

    def forward(
        self, 
        model, 
        tokenizer, 
        input_ids, 
        attention_mask, 
        n_mutations, 
        k,
        cls_input = None
    ):
        # Get current sequence embeddings
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        embeddings = out.hidden_states[-1]

        # Compute the sequences mask from the embeddings
        logits = self.classifier(embeddings)[:,:,0]
        logits = attention_mask * logits + (1-attention_mask) * -1e6
        probs = F.softmax(logits, dim=-1)
        mutations = Categorical(probs=probs).sample(sample_shape=torch.Size([1, n_mutations]))[0,:,:].T
        mutations_mask = (F.one_hot(mutations, num_classes=logits.shape[-1]).sum(axis=1) > 0).long()

        # Mask the input the input ids using the mutations mask
        masked_input_ids = ((1-mutations_mask)*input_ids) + (mutations_mask*tokenizer.mask_token_id)

        # Predict the masked ids
        out = model(input_ids=masked_input_ids, attention_mask=attention_mask)
        top_k_ids = (-out.logits).argsort(dim=-1)[:,:,:k]
        is_in_top_k_ids = ((input_ids[:,:,None] - top_k_ids == 0).sum(axis=-1) > 0).long()
        mutated_ids = (1-mutations_mask)*input_ids + mutations_mask*(is_in_top_k_ids*input_ids + (1-is_in_top_k_ids)*top_k_ids[:,:,0])
        
        # Compute the mutated sequence pseudo-perplexity
        out = model(input_ids=mutated_ids, attention_mask=attention_mask, output_hidden_states=True)
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
    
class MutatorGA(GeneticAlgorithm):
    pass