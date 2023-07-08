import typing
import torch

from transformers import AutoTokenizer

class Tokens():
    def __init__(
        self,
        input_ids: typing.Optional[torch.Tensor]=None,,
        attention_mask: typing.Optional[torch.Tensor]=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

class MaskedTokens(Tokens):
    def __init__(
        self,
        input_ids: typing.Optional[torch.Tensor]=None,
        masked_input_ids: typing.Optional[torch.Tensor]=None,
        attention_mask: typing.Optional[torch.Tensor]=None,
        masked_mask: typing.Optional[torch.Tensor]=None
    ):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)
        self.masked_input_ids = masked_input_ids
        self.masked_mask = masked_mask

def get_collate_fn(
        tokenizer: AutoTokenizer,
        mask: typing.Optional[bool]=False
    ) -> typing.Callable:

    def collate_fn(seqs: typing.List[str]) -> Tokens:
        seqs = [' '.join(list(el)) for el in seqs]
        tokens = tokenizer(seqs, return_tensors="pt", padding=True)
        input_ids, attention_mask = tokens.input_ids, tokens.attention_mask

        if mask:
            masked_mask = (torch.rand_like(input_ids, dtype=torch.float32) < args["p"]).int()
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