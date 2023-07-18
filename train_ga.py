import torch

from src import Mutator, MutatorOutput, GeneticAlgorithm as GA, get_model, get_dataloaders

if __name__ == "__main__":
    args = {
        "from_tokenizer": "Rostlab/prot_bert_bfd",
        "from_model": "Rostlab/prot_bert_bfd",
        "from_adapters": "./models/hps/LR0.001_BS256_P0.05/best/",
        "set": "./data/non_thermo.csv",
        "min_length": None,
        "max_length": None,
        "global_batch_size": 128,
        "local_batch_size": 128,
        "num_workers": 8,
        "mask": False,
        "p": 0,
        "n_epochs": 10
    }

    dataloader, tokenizer = get_dataloaders(args, get_training_and_validation=False, get_tokenizer=True)
    model = get_model(args)
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.to(torch.bfloat16)

    GA.configure(
        Mutator, None, None,
        model,
        tokenizer=tokenizer,
        n_mutations=5,
        k=3,
        mutation_rate=0.1
    )
    ga = GA(4, 2)

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

    ga.save("best_mutator_2")