import os

import logging

import torch

from src import Mutator, MutatorOutput, GeneticAlgorithm as GA, get_model, get_dataloaders, configure

if __name__ == "__main__":
    #args = {
    #    "from_tokenizer": "Rostlab/prot_bert_bfd",
    #    "from_model": "Rostlab/prot_bert_bfd",
    #    "from_adapters": "./models/hps/LR0.001_BS256_P0.05/best/",
    #    "training_set": "./data/non_thermo.csv",
    #    "min_length": None,
    #    "max_length": None,
    #    "global_batch_size": 128,
    #    "local_batch_size": 128,
    #    "num_workers": 8,
    #    "mask": False,
    #    "p": 0,
    #    "n_epochs": 10
    #}

    args = configure()
    dataloader, tokenizer = get_dataloaders(args, return_validation=False, return_tokenizer=True)
    model = get_model(args)
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.to(torch.bfloat16)

    GA.configure(
        Mutator, None, None,
        model,
        tokenizer=tokenizer,
        n_mutations=args["n_mutations"],
        k=args["k"],
        mutation_rate=args["mutation_rate"]
    )
    ga = GA(args["population_size"], args["offspring_size"])

    accumulation_steps = args["global_batch_size"] // args["local_batch_size"]

    for i_epoch in range(args["n_epochs"]):
        for i_batch, batch in enumerate(dataloader):
            out = ga(
                MutatorOutput.to_fitness, 
                input_ids=batch.input_ids.to(ga.device), 
                attention_mask=batch.attention_mask.to(ga.device), 
                p_coef=1, 
                d_coef=1
            )

            if (
                (i_batch + 1) % accumulation_steps == 0 or 
                (i_batch + 1) == len(dataloader)
            ):
                print(ga.fitness)
                ga.step()
                ga.zero_fitness()

    ga.save(os.path.join(args["model_dir"], args["model_name"], args["model_version"], "best" + ".bin"))