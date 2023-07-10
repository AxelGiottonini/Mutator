import sys
import os
import typing
import time

import argparse
import logging
import json

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.adapters import PfeifferInvConfig

from src import Dataset, get_collate_fn

torch.manual_seed(42)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--log_dir", type=str, default="./logs")

    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--model_version", type=str, required=True, help="Movel version")

    parser.add_argument("--from_tokenizer", type=str, default="Rostlab/prot_bert_bfd", help="Path or Huggingface's repository of the model's tokenizer")
    parser.add_argument("--from_model", type=str, default="Rostlab/prot_bert_bfd", help="Path to repository containing the model's encoder and decoder")
    parser.add_argument("--from_adapters", type=str, default=None, help="Path to repository containing the model's adapter, if None, the adapters are initialized")
    
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate")
    parser.add_argument("--betas", type=str, default="(0.9, 0.999)", help="betas")
    parser.add_argument("--eps", type=float, default=1e-08, help="eps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    
    parser.add_argument("--training_set", type=str, required=True, help="Path to training set")
    parser.add_argument("--validation_set", type=str, required=True, help="Path to validation set")
    parser.add_argument("--min_length", type=int, default=None, help="Minimum sequence length")
    parser.add_argument("--max_length", type=int, default=None, help="Maximum sequence length")

    parser.add_argument("--p", type=float, default=0.15, help="masking probability")

    parser.add_argument("--n_epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--local_batch_size", type=int, default=1, help="Mini-Batch size")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of sub-processes to use for data loading.")
    args = vars(parser.parse_args())

    betas = args["betas"][1:-1].replace(" ", "").split(",")
    if not len(betas) == 2:
        raise ValueError()
    args["betas"] = tuple(float(el) for el in betas)

    if not args["batch_size"] % args["local_batch_size"] == 0:
        raise ValueError(f"--batch_size ({args['batch_size']}) should be a multiple of --local_batch_size ({args['local_batch_size']})")

    if not os.path.isdir(args["model_dir"]):
        os.mkdir(args["model_dir"])

    if not os.path.isdir(os.path.join(args["model_dir"], args["model_name"])):
        os.mkdir(os.path.join(args["model_dir"], args["model_name"]))

    if os.path.isdir(os.path.join(args["model_dir"], args["model_name"], args["model_version"])):
        raise FileExistsError("The same version of the model exists, please choose a new version")

    os.mkdir(os.path.join(args["model_dir"], args["model_name"], args["model_version"]))
    os.mkdir(os.path.join(args["model_dir"], args["model_name"], args["model_version"], "best"))
    os.mkdir(os.path.join(args["model_dir"], args["model_name"], args["model_version"], "final"))

    if not os.path.isdir(args["log_dir"]):
        os.mkdir(args["log_dir"])

    if not os.path.isdir(os.path.join(args["log_dir"], args["model_name"])):
        os.mkdir(os.path.join(args["log_dir"], args["model_name"]))

    logging.basicConfig(filename=os.path.join(args["log_dir"], args["model_name"], args["model_version"] + ".log"), level=logging.INFO)

    with open(os.path.join(args["model_dir"], args["model_name"], args["model_version"], "args.json"), 'w', encoding='utf8') as f:
        f.write(json.dumps(args, indent=4, sort_keys=False, separators=(',', ': '), ensure_ascii=False))

    return args

def train(
    tokenizer,
    model,
    collate_fn,
    training_dataloader,
    validation_dataloader,
    optimizer,
    device,
    precision,
    args
):
    model.to(device)
    model.to(precision)

    accumulation_steps = args["batch_size"] // args["local_batch_size"]

    metrics = {
        "best_loss": float("inf"),
        "training/loss/step": [],
        "training/loss/mean": [],
        "validation/loss/mean": []
    }

    def step(input_ids, attention_mask, output_ids, masked_mask, model):
        out = model(
            input_ids = input_ids,
            attention_mask = attention_mask
        )

        logits = out.logits.view(-1, out.logits.shape[-1])
        preds = F.log_softmax(logits, dim=-1)
        loss_fn = NLLLoss(reduction='none')
        mlkp_loss = loss_fn(preds, output_ids.flatten())
        mlkp_loss = ((masked_mask.flatten() * mlkp_loss)).sum() / masked_mask.sum()
        return mlkp_loss

    for i_epoch in range(1, args["n_epochs"]+1):

        epoch_metrics = {
            "start_time": time.time(),
            "training/loss": [],
            "validation/loss": [],
        }

        model.train()
        torch.cuda.empty_cache()
        
        for i_batch, batch in enumerate(training_dataloader):
            mlkp_loss = step(
                input_ids=batch.masked_input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
                output_ids=batch.input_ids.to(device),
                masked_mask=batch.masked_mask.to(device),
                model=model
            )

            (mlkp_loss / accumulation_steps).backward()
            if (i_batch + 1) % accumulation_steps == 0 or i_batch + 1 == len(training_dataloader):
                optimizer.step()
                optimizer.zero_grad()

            metrics["training/loss/step"].append(mlkp_loss.item())
            epoch_metrics["training/loss"].append(mlkp_loss.item())
            if i_batch % (10*accumulation_steps) == 0:
                torch.save(metrics, os.path.join(args["model_dir"], args["model_name"], args["model_version"], "metrics.bin"))

        model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            
            for i_batch, batch in enumerate(validation_dataloader):
                mlkp_loss = step(
                    input_ids=batch.masked_input_ids.to(device),
                    attention_mask=batch.attention_mask.to(device),
                    output_ids=batch.input_ids.to(device),
                    masked_mask=batch.masked_mask.to(device),
                    model=model
                )

                epoch_metrics["validation/loss"].append(mlkp_loss.item())

        metrics["training/loss/mean"].append(torch.tensor(epoch_metrics["training/loss"]).mean())
        metrics["validation/loss/mean"].append(torch.tensor(epoch_metrics["validation/loss"]).mean())

        str_metrics = f"Training Loss: {metrics['training/loss/mean'][-1]:.4f}" + " | " + f"Validation Loss:{metrics['validation/loss/mean'][-1]:.4f}"
        str_duration = f"Duration: {time.strftime('%H:%M:%S', time.gmtime(time.time() - epoch_metrics['start_time']))}"
        str_epoch = f"EPOCH[{i_epoch}]" + "\n\t" + str_metrics + "\n\t" + str_duration
        logging.info(str_epoch)

        if metrics['validation/loss/mean'][-1] < metrics["best_loss"]*0.98:
            model.save_adapter(os.path.join(args["model_dir"], args["model_name"], args["model_version"], "best"), "thermo")

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args["from_tokenizer"])
    model = AutoModelForMaskedLM.from_pretrained(args["from_model"])

    config = PfeifferInvConfig()
    if args["from_adapters"] is None:
        model.add_adapter("thermo", config=config)
    else:
        model.load_adapter(args["from_adapters"])
    model.set_active_adapters("thermo")
    model.train_adapter("thermo")
    
    n_total_params = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"#total: {n_total_params}; #trainable: {n_train_params}")

    collate_fn = get_collate_fn(tokenizer, mask=True, mask_rate=args["p"])

    training_set = Dataset(args["training_set"], min_length=args["min_length"], max_length=args["max_length"])
    validation_set = Dataset(args["validation_set"], min_length=args["min_length"], max_length=args["max_length"])

    logging.info(f"#train: {len(training_set)}; #val: {len(validation_set)}")

    training_dataloader = DataLoader(dataset=training_set, batch_size=args["local_batch_size"], shuffle=True, num_workers=args["num_workers"], collate_fn=collate_fn)
    validation_dataloader = DataLoader(dataset=validation_set, batch_size=args["local_batch_size"], shuffle=False, num_workers=args["num_workers"], collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args["learning_rate"], 
        betas=args["betas"], 
        eps=args["eps"], 
        weight_decay=args["weight_decay"]
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16

    try:
        train(
            tokenizer,
            model,
            collate_fn,
            training_dataloader,
            validation_dataloader,
            optimizer,
            device,
            precision,
            args
        )
    except KeyboardInterrupt:
        pass

    model.save_adapter(os.path.join(args["model_dir"], args["model_name"], args["model_version"], "final"), "thermo")
    sys.exit(0)

if __name__ == "__main__":
    args = parse_args()
    main(args)