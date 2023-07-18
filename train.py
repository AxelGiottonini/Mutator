import random
import torch
import torch.nn.functional as F
from torch.nn import NLLLoss

from src import train_loop, configure, get_model, get_dataloaders

random.seed(42)
torch.manual_seed(42)

if __name__ == "__main__":

    args = configure()
    training_dataloader, validation_dataloader = get_dataloaders(args)
    model, optimizer = get_model(args, return_config=False, return_optimizer=True)

    @train_loop(
        model = model,
        optimizer = optimizer,
        args = args
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

    train(training_dataloader, validation_dataloader)