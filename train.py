import random
import torch
import torch.nn.functional as F
from torch.nn import NLLLoss

from src import train_loop, configure, get_model, get_dataloaders

random.seed(42)
torch.manual_seed(42)

if __name__ == "__main__":

    args = configure()
    training_dataloader, validation_dataloader, tokenizer, collate_fn = get_dataloaders(args, get_tokenizer=True, get_collate_fn=True)
    model, _, optimizer = get_model(args, get_config=False, get_optimizer=True)

    @train_loop(
        model = model,
        optimizer = optimizer,
        n_epochs = args["n_epochs"],
        global_batch_size = args["global_batch_size"],
        local_batch_size = args["local_batch_size"]
    )
    def train(model, batch):

        input_ids=batch.masked_input_ids
        attention_mask=batch.attention_mask
        output_ids=batch.input_ids
        masked_mask=batch.masked_mask

        out = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = out.logits.view(-1, out.logits.shape[-1])
        preds = F.log_softmax(logits, dim=-1)
        loss_fn = NLLLoss(reduction='none')
        loss = loss_fn(preds, output_ids.flatten())
        loss = ((masked_mask.flatten() * loss)).sum() / masked_mask.sum()
        return loss

    train(training_dataloader, validation_dataloader, args)