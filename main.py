import logging
import uuid
from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, AutoModelForTokenClassification

import wandb
from src.args_utils import init_args
from src.config import BEST_MODEL_FOLDER, init_logging, WANDB_PROJECT_NAME, WANDB_ENTITY
from src.data_utils import prepare_data_loaders, prepare_datasets_task2
from src.evaluate import get_predictions_and_write_results
from src.task import Task
from src.train import train_model


def load_model_and_tokenizer(
        model_path: str,
        num_labels: int,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    return model, tokenizer


def main():
    init_logging()
    args = init_args()

    if not args.no_wandb:
        wandb.init(
            project=WANDB_PROJECT_NAME,
            entity=WANDB_ENTITY,
            config=vars(args),
        )

    device = torch.device("cpu" if args.use_cpu else "cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Using device: %s", device)

    logging.info("Loading datasets")
    datasets, test_data = prepare_datasets_task2(args)
    logging.info("Datasets loaded")

    logging.info("Loading model and tokenizer")
    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model,
        num_labels=2,
    )
    logging.info("Model and tokenizer loaded")

    dev_data_loader, test_data_loader, train_data_loader = prepare_data_loaders(
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        datasets=datasets,
        max_seq_length=args.max_seq_length,
    )

    unique_id = uuid.uuid4() if args.id is None else args.id

    if not args.no_wandb:
        wandb.log({"unique_id": str(unique_id)})

    if not args.inference_only:
        logging.info("Training model")

        best_model_dir = Path(BEST_MODEL_FOLDER)

        model_path = best_model_dir / f"best_model-{unique_id}"

        model_path = str(model_path)

        train_model(
            model=model,
            tokenizer=tokenizer,
            train_data_loader=train_data_loader,
            dev_data_loader=dev_data_loader,
            args=args,
            device=device,
            model_path=model_path,
            no_wandb=args.no_wandb,
        )

        best_model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=2)
    else:
        logging.info("Inference only")
        best_model = model

    get_predictions_and_write_results(
        best_model=best_model,
        device=device,
        test_data=test_data,
        test_data_loader=test_data_loader,
        tokenizer=tokenizer,
        unique_id=unique_id,
        task=Task.TASK2,
        unique_labels=None,
    )

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
