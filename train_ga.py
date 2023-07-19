import random
import torch

from src import Mutator, MutatorOutput, GeneticAlgorithm as GA, get_model, get_dataloaders, configure, train_loop

random.seed(42)
torch.manual_seed(42)

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16

    args = configure()
    dataloader, tokenizer = get_dataloaders(args, return_validation=False, return_tokenizer=True)
    model = get_model(args)
    model.to(device).to(precision)

    GA.configure(
        Mutator, device, precision,
        model,
        tokenizer=tokenizer,
        n_mutations=args["n_mutations"],
        k=args["k"],
        mutation_rate=args["mutation_rate"]
    )
    ga = GA(args["population_size"], args["offspring_size"])

    @train_loop(
        model = ga,
        optimizer = ga,
        args = args,
        backpropagation = False,
        save_model = True,
        save_adapter = False,
        save_optimizer = False
    )
    def train(model, batch):
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        p_coef = args["p_coef"]
        d_coef = args["d_coef"]

        model(
            MutatorOutput.to_fitness,
            input_ids=input_ids,
            attention_mask=attention_mask,
            p_coef=p_coef,
            d_coef=d_coef
        )

        return model.fitness.min()
    
    train(dataloader)
