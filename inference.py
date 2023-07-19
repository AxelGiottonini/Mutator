import torch
import pandas as pd
from src import Mutator, get_model, get_dataloaders, get_mutator, perplexity, configure

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16

    args = configure()

    model = get_model(args)
    model.to(device).to(precision)
    dataloader, tokenizer = get_dataloaders(args, shuffle=False, return_validation=False, return_tokenizer=True)
    mutator = get_mutator(args, model, tokenizer)
    mutator.to(device).to(precision)

    results = {
        "perplexity_src": torch.empty([0]),
        "perplexity_mut": torch.empty([0]),
        "sequences_mut": []
    }

    for batch in dataloader:
        batch = batch.to(device)

        out = mutator(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        perplexity_src = perplexity(model, out.input_ids, attention_mask=out.attention_mask)
        perplexity_mut = perplexity(model, out.mutated_ids, attention_mask=out.attention_mask)
        sequences_mut = tokenizer.batch_decode(out.mutated_ids, skip_special_tokens=True)
        sequences_mut = [el.replace(" ", "") for el in sequences_mut]

        results["perplexity_src"] = torch.concat([results["perplexity_src"], perplexity_src.cpu()])
        results["perplexity_mut"] = torch.concat([results["perplexity_mut"], perplexity_mut.cpu()])
        results["sequences_mut"].extend(sequences_mut)

    results = pd.DataFrame(results)
    results.to_csv("results.csv", index=None)