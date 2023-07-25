import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.adapters import PfeifferInvConfig

from .dataset import Dataset
from .mutator import Mutator
from .tokenizer import __get_collate_fn__

__all__ = ["get_model", "get_dataloaders", "get_mutator"]

def __get_tokenizer(args):
    return AutoTokenizer.from_pretrained(args["from_tokenizer"])

def __get_collate_fn(tokenizer, args):
    return __get_collate_fn__(tokenizer, mask=args["mask"], mask_rate=args["p"])

def get_model(args, return_config=False, return_optimizer=False):
    model = AutoModelForMaskedLM.from_pretrained(args["from_model"])
    config = PfeifferInvConfig()
    if args["from_adapters"] is None:
        model.add_adapter("thermo", config=config)
    else:
        model.load_adapter(args["from_adapters"])
    model.set_active_adapters("thermo")
    model.train_adapter("thermo")

    optimizer = None
    if return_optimizer:
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args["learning_rate"], 
            betas=args["betas"], 
            eps=args["eps"], 
            weight_decay=args["weight_decay"]
        )

    out =  (
        model,
        optimizer,
        config if return_config else None, 
    )
    out = tuple(el for el in out if el is not None)
    out = out[0] if len(out) == 1 else out
    return out

def get_dataloaders(args, shuffle=True, return_validation=True, return_tokenizer=False, return_collate_fn=False):

    tokenizer = __get_tokenizer(args)
    collate_fn = __get_collate_fn(tokenizer, args)

    training_set = Dataset(args["training_set"], min_length=args["min_length"], max_length=args["max_length"])
    training_dataloader = DataLoader(dataset=training_set, batch_size=args["local_batch_size"], shuffle=shuffle, num_workers=args["num_workers"], collate_fn=collate_fn)

    if return_validation:
        validation_set = Dataset(args["validation_set"], min_length=args["min_length"], max_length=args["max_length"])
        validation_dataloader = DataLoader(dataset=validation_set, batch_size=args["local_batch_size"], shuffle=False, num_workers=args["num_workers"], collate_fn=collate_fn)

    out =  (
        training_dataloader,
        validation_dataloader if return_validation else None,
        tokenizer if return_tokenizer else None,
        collate_fn if return_collate_fn else None
    ) 
    out = tuple(el for el in out if el is not None)
    out = out[0] if len(out) == 1 else out
    return out

def get_mutator(args, model=None, tokenizer=None, return_model=False, return_tokenizer=False):
    if model is None:
        model = get_model(args)

    if tokenizer is None:
        tokenizer = __get_tokenizer(args)

    Mutator.configure(model, tokenizer, args["n_mutations"], args["k"], args["mutation_rate"])
    mutator = Mutator()
    mutator.load_state_dict(torch.load(args["from_mutator"]))

    out = (
        mutator,
        model if return_model else None,
        tokenizer if return_tokenizer else None
    )
    out = tuple(el for el in out if el is not None)
    out = out[0] if len(out) == 1 else out
    return out