import os
import argparse
import logging
import json


__all__ = ["configure", "summary"]

def __parse_args__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--log_dir", type=str, default="./logs")

    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--model_version", type=str, help="Movel version")

    parser.add_argument("--from_tokenizer", type=str, default="Rostlab/prot_bert_bfd", help="Path or Huggingface's repository of the model's tokenizer")
    parser.add_argument("--from_model", type=str, default="Rostlab/prot_bert_bfd", help="Path to repository containing the model's encoder and decoder")
    parser.add_argument("--from_adapters", type=str, default=None, help="Path to repository containing the model's adapter, if None, the adapters are initialized")
    parser.add_argument("--from_mutator", type=str, default=None)
    
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--betas", type=str, default="(0.9, 0.999)", help="betas")
    parser.add_argument("--eps", type=float, default=1e-08, help="eps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    
    parser.add_argument("--training_set", type=str, required=True, help="Path to training set")
    parser.add_argument("--validation_set", type=str, default=None, help="Path to validation set")
    parser.add_argument("--min_length", type=int, default=None, help="Minimum sequence length")
    parser.add_argument("--max_length", type=int, default=None, help="Maximum sequence length")

    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--p", type=float, default=0.15, help="masking probability")

    parser.add_argument("--n_epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--global_batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--local_batch_size", type=int, default=1, help="Mini-Batch size")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of sub-processes to use for data loading.")

    parser.add_argument("--n_mutations", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--mutation_rate", type=float, default=0.1)
    parser.add_argument("--population_size", type=int, default=100)
    parser.add_argument("--offspring_size", type=int, default=60)
    parser.add_argument("--p_coef", type=float, default=1.0)
    parser.add_argument("--d_coef", type=float, default=1.0)

    parser.add_argument("--n_iter", type=int, default=1)

    parser.add_argument("--output", type=str, default=None)

    parser.add_argument("--save_each", type=int, default=10)

    args = vars(parser.parse_args())

    betas = args["betas"][1:-1].replace(" ", "").split(",")
    if not len(betas) == 2:
        raise ValueError()
    args["betas"] = tuple(float(el) for el in betas)

    if not args["global_batch_size"] % args["local_batch_size"] == 0:
        raise ValueError(f"--global_batch_size ({args['global_batch_size']}) should be a multiple of --local_batch_size ({args['local_batch_size']})")

    return args

def __safe_makedirs__(model_dir, model_name, model_version):
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    if not os.path.isdir(current:=(os.path.join(model_dir, model_name))):
        os.mkdir(current)

    if not os.path.isdir(current:=(os.path.join(model_dir, model_name, model_version))):
        os.mkdir(current)
        os.mkdir(os.path.join(current, "best"))
        os.mkdir(os.path.join(current, "final"))
        os.mkdir(os.path.join(current, "crash"))
    else:
        raise FileExistsError("The same version of the model exists, please choose a new version")

def __safe_logging__(log_dir, model_name, model_version):
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    if not os.path.isdir(current:=(os.path.join(log_dir, model_name))):
        os.mkdir(current)

    logging.basicConfig(filename=os.path.join(log_dir, model_name, model_version + ".log"), level=logging.INFO, format='%(message)s')

def __save_args__(model_dir, model_name, model_version, args):
    with open(os.path.join(model_dir, model_name, model_version, "args.json"), 'w', encoding='utf8') as f:
        f.write(json.dumps(args, indent=4, sort_keys=False, separators=(',', ': '), ensure_ascii=False))

def configure(training=True):
    args = __parse_args__()
    if training:
        __safe_makedirs__(args["model_dir"], args["model_name"], args["model_version"])
        __safe_logging__(args["log_dir"], args["model_name"], args["model_version"])
        __save_args__(args["model_dir"], args["model_name"], args["model_version"], args)
    return args

def summary(model, training_dataloader, validation_dataloader):
    n_total_params = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    len_training_dataloader = len(training_dataloader.dataset) if training_dataloader is not None else 0
    len_validation_dataloader = len(validation_dataloader.dataset) if validation_dataloader is not None else 0
    logging.info(
        f"\n"
        f"Parameters:\n" +
        f"\tTotal: {n_total_params}\n" +
        f"\tTraining: {n_train_params}\n" +
        f"="*80 + "\n" +
        f"Datasets:\n" +
        f"\tTraining: {len_training_dataloader}\n" +
        f"\tValidation: {len_validation_dataloader}\n" +
        f"="*80 + "\n"
    )