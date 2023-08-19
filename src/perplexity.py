import typing
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import no_grad

@no_grad
def perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    logits: typing.Optional[torch.Tensor]=None,
    attention_mask: typing.Optional[torch.Tensor]=None,
):
    if logits is None:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    
    log_probs = F.log_softmax(logits, dim=-1)
    perplexity = (-((F.one_hot(input_ids, num_classes=model.config.vocab_size) * log_probs).sum(axis=-1) * attention_mask).sum(axis=-1) / attention_mask.sum(axis=-1)).exp()
    
    return perplexity