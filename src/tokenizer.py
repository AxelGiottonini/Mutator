import typing
import torch

from transformers import AutoTokenizer

class Tokens():
    def __init__(
        self,
        input_ids: typing.Optional[torch.Tensor]=None,
        attention_mask: typing.Optional[torch.Tensor]=None
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, *args, **kwargs):
        self.input_ids = self.input_ids.to(*args, **kwargs)
        self.attention_mask = self.attention_mask.to(*args, **kwargs)

        return self

class MaskedTokens(Tokens):
    def __init__(
        self,
        masked_input_ids: typing.Optional[torch.Tensor]=None,
        masked_mask: typing.Optional[torch.Tensor]=None,
        *args: typing.Any,
        **kwargs: typing.Any
    ):
        super().__init__(*args, **kwargs)
        self.masked_input_ids = masked_input_ids
        self.masked_mask = masked_mask

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.masked_input_ids = self.masked_input_ids.to(*args, **kwargs)
        self.masked_mask = self.masked_mask.to(*args, **kwargs)
        return self

def __get_collate_fn__(
    tokenizer: AutoTokenizer,
    mask: typing.Optional[bool]=False,
    mask_rate: typing.Optional[float]=None
) -> typing.Callable:

    def collate_fn(seqs: typing.List[str]) -> Tokens:
        seqs = [' '.join(list(el)) for el in seqs]
        tokens = tokenizer(seqs, return_tensors="pt", padding=True)
        input_ids, attention_mask = tokens.input_ids, tokens.attention_mask

        if mask:
            masked_mask = (torch.rand_like(input_ids, dtype=torch.float32) < mask_rate).int()
            masked_mask = attention_mask * masked_mask
            masked_input_ids = ((1-masked_mask)*input_ids) + masked_mask*tokenizer.mask_token_id

            tokens = MaskedTokens(
                input_ids=input_ids,
                masked_input_ids=masked_input_ids,
                attention_mask=attention_mask,
                masked_mask=masked_mask
            )
            return tokens
        
        tokens = Tokens(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return tokens

    return collate_fn