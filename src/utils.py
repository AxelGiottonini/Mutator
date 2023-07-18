import sys
import os

import typing
import logging

import time

import torch
import torch.nn as nn

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.adapters import PfeifferInvConfig

from .dataset import Dataset
from .tokenizer import __get_collate_fn__

from .cli import summary

__all__ = ["no_grad", "train_loop", "get_model", "get_dataloaders"]

def no_grad(fun: typing.Callable)->typing.Callable:
    """
    Decorator function to disable gradients.

    Usage:
        @no_grad
        def fun(*args, **kwargs):
            ...
    """
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return fun(*args, **kwargs)

    return wrapper

def train_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: dict,
    device: torch.device=torch.device("cuda:0" if torch.cuda.is_available else "cpu"), 
    precision: torch.memory_format=torch.bfloat16
)->typing.Callable:
    """
    Decorator function for training function

    Usage:
        @train_loop(
            model=model, 
            optimizer=optimizer,
            global_batch_size=64,
            n_epochs=50
        )
        def fun(*args, **kwargs):
            ...
    """
    n_epochs = args["n_epochs"]
    global_batch_size = args["global_batch_size"]
    local_batch_size = args["local_batch_size"]
    save_each = args["save_each"]

    model.to(device)
    model.to(precision)

    if local_batch_size is None:
        local_batch_size = global_batch_size
    accumulation_steps = global_batch_size // local_batch_size

    metrics = {
        "best_loss": float("inf"),
        "training/loss/step": [],
        "training/loss/mean": [],
        "validation/loss/mean": []
    }

    def decorator(step: typing.Callable)->typing.Callable:
        def wrapper(
            training_dataloader, 
            validation_dataloader=None, 
        ):
            summary(model, training_dataloader, validation_dataloader)
            for i_epoch in range(1, n_epochs+1):
                try:
                    epoch_metrics = {
                        "start_time": time.time(),
                        "training/loss": [],
                        "validation/loss": [],
                    }

                    # Training
                    model.train()
                    torch.cuda.empty_cache()
                    for i_batch, batch in enumerate(training_dataloader):
                        loss = step(model, batch.to(device))
                        (loss / accumulation_steps).backward()

                        if (
                            (i_batch + 1) % accumulation_steps == 0 or 
                            (i_batch + 1) == len(training_dataloader)
                        ):
                            optimizer.step()
                            optimizer.zero_grad()

                        metrics["training/loss/step"].append(loss.item())
                        epoch_metrics["training/loss"].append(loss.item())

                        if (
                            (i_batch + 1) % (save_each * accumulation_steps) == 0 or
                            (i_batch + 1) == len(training_dataloader)
                        ):
                            torch.save(
                                metrics, 
                                os.path.join(args["model_dir"], args["model_name"], args["model_version"], "metrics.bin")
                            )

                    # Validation
                    model.eval()
                    torch.cuda.empty_cache()
                    for i_batch, batch in enumerate(validation_dataloader):
                        loss = step(model, batch.to(device))

                        epoch_metrics["validation/loss"].append(loss.item())

                    metrics["training/loss/mean"].append(torch.tensor(epoch_metrics["training/loss"]).mean())
                    metrics["validation/loss/mean"].append(torch.tensor(epoch_metrics["validation/loss"]).mean())

                    # Logging
                    str_metrics = f"Training Loss: {metrics['training/loss/mean'][-1]:.4f}" + " | " + f"Validation Loss:{metrics['validation/loss/mean'][-1]:.4f}"
                    str_duration = f"Duration: {time.strftime('%H:%M:%S', time.gmtime(time.time() - epoch_metrics['start_time']))}"
                    str_epoch = f"EPOCH[{i_epoch}]" + "\n\t" + str_metrics + "\n\t" + str_duration
            
                    logging.info(str_epoch)

                    # Saving
                    if metrics['validation/loss/mean'][-1] < metrics["best_loss"]*0.98:
                        model.save_adapter(
                            os.path.join(args["model_dir"], args["model_name"], args["model_version"], "best"), 
                            "thermo"
                        )
                        torch.save(
                            optimizer.state_dict(), 
                            os.path.join(args["model_dir"], args["model_name"], args["model_version"], "best", "optimizer.bin")
                        )
                        
                        metrics["best_loss"] = metrics['validation/loss/mean'][-1]

                except (KeyboardInterrupt, RuntimeError):
                    model.save_adapter(
                        os.path.join(args["model_dir"], args["model_name"], args["model_version"], "crash"), 
                        "thermo"
                    )
                    torch.save(
                        optimizer.state_dict(), 
                        os.path.join(args["model_dir"], args["model_name"], args["model_version"], "crash", "optimizer.bin")
                    )
                    sys.exit(0)

            model.save_adapter(
                os.path.join(args["model_dir"], args["model_name"], args["model_version"], "final"), 
                "thermo"
            )
            torch.save(
                optimizer.state_dict(), 
                os.path.join(args["model_dir"], args["model_name"], args["model_version"], "final", "optimizer.bin")
            )

        return wrapper
    return decorator

def __get_tokenizer(args):
    return AutoTokenizer.from_pretrained(args["from_tokenizer"])

def __get_collate_fn(tokenizer, args):
    return __get_collate_fn__(tokenizer, mask=args["mask"], mask_rate=args["p"])

def get_model(args, get_config=False, get_optimizer=False):
    model = AutoModelForMaskedLM.from_pretrained(args["from_model"])
    config = PfeifferInvConfig()
    if args["from_adapters"] is None:
        model.add_adapter("thermo", config=config)
    else:
        model.load_adapter(args["from_adapters"])
    model.set_active_adapters("thermo")
    model.train_adapter("thermo")

    if get_optimizer:
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args["learning_rate"], 
            betas=args["betas"], 
            eps=args["eps"], 
            weight_decay=args["weight_decay"]
        )

    out =  (
        model,
        config if get_config else None, 
        optimizer if get_optimizer else None
    )

    out = tuple(el for el in out if el is not None)
    out = out[0] if len(out) == 1 else out
    return out

def get_dataloaders(args, get_training_and_validation=True, get_tokenizer=False, get_collate_fn=False):

    tokenizer = __get_tokenizer(args)
    collate_fn = __get_collate_fn(tokenizer, args)

    if get_training_and_validation:
        training_set = Dataset(args["training_set"], min_length=args["min_length"], max_length=args["max_length"])
        validation_set = Dataset(args["validation_set"], min_length=args["min_length"], max_length=args["max_length"])

        training_dataloader = DataLoader(dataset=training_set, batch_size=args["local_batch_size"], shuffle=True, num_workers=args["num_workers"], collate_fn=collate_fn)
        validation_dataloader = DataLoader(dataset=validation_set, batch_size=args["local_batch_size"], shuffle=False, num_workers=args["num_workers"], collate_fn=collate_fn)

        out =  (
            training_dataloader,
            validation_dataloader,
            tokenizer if get_tokenizer else None,
            collate_fn if get_collate_fn else None
        )
    
    else:
        dataset = Dataset(args["set"], min_length=args["min_length"], max_length=args["max_length"])
        dataloader = DataLoader(dataset=dataset, batch_size=args["local_batch_size"], shuffle=False, num_workers=10, collate_fn=collate_fn)

        out =  (
            dataloader,
            tokenizer if get_tokenizer else None,
            collate_fn if get_collate_fn else None
        )
    
    out = tuple(el for el in out if el is not None)
    out = out[0] if len(out) == 1 else out
    return out