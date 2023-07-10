import sys
import os

import typing
import logging

import time

import torch
import torch.nn as nn

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
    global_batch_size: int, 
    n_epochs: int,
    logging: logging,
    local_batch_size: typing.Optional[int]=None,
    device: torch.device=torch.device("cuda:0" if torch.cuda.is_available else "cpu"), 
    precision: torch.memory_format=torch.bfloat16,
    save_each: typing.Optional[int]=10
)->typing.Callable:
    """
    Decorator function for training function

    Usage:
        @train_loop(
            model=model, 
            optimizer=optimizer,
            global_batch_size=64,
            n_epochs=50,
            logging=logging
        )
        def fun(*args, **kwargs):
            ...
    """
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
        def wrapper(training_dataloader, validation_dataloader, args):
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
                        loss = step(model, batch)
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
                        loss = step(model, batch)

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
                os.path.join(args["model_dir"], args["model_name"], args["model_version"], "crash"), 
                "thermo"
            )
            torch.save(
                optimizer.state_dict(), 
                os.path.join(args["model_dir"], args["model_name"], args["model_version"], "crash", "optimizer.bin")
            )


        return wrapper
    return decorator
