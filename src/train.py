import argparse
import logging
from math import ceil

import torch
import wandb
from peft import PeftModel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ConstantLR
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase, Adafactor, get_linear_schedule_with_warmup

from src.evaluate import evaluate_model_token_classification


def get_optimizer(model: PreTrainedModel, optimizer_name: str, learning_rate: float) -> Optimizer:
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == "adafactor":
        return Adafactor(model.parameters(), lr=learning_rate, scale_parameter=False, relative_step=False)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(
        scheduler_name: str,
        optimizer: Optimizer,
        epochs: int,
        total_steps: int,
) -> LRScheduler:
    if scheduler_name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0.03 * total_steps,
            num_training_steps=total_steps,
        )
    elif scheduler_name == "warmup_linear_decay":
        # count steps for one epoch
        one_epoch_steps = total_steps // epochs
        return WarmupLinearDecayScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=total_steps,
            decay_steps=one_epoch_steps * 3,
            start_lr=1e-6,
            end_lr=2e-4,
        )
    return ConstantLR(optimizer, factor=1.0, total_iters=total_steps, last_epoch=-1)


class WarmupLinearDecayScheduler(LRScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            warmup_steps: int = 100,
            total_steps: int = 1000,
            decay_steps: int = 900,
            start_lr: float = 1e-6,
            end_lr: float = 2e-4,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.start_lr = start_lr
        self.decay_steps = decay_steps
        self.end_lr = end_lr
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            alpha = self.last_epoch / self.warmup_steps
            lr = self.start_lr + alpha * (self.end_lr - self.start_lr)
        elif self.last_epoch < self.decay_steps:
            # keep the end_lr constant
            lr = self.end_lr
        else:
            # linear decay for rest of the steps to start_lr
            alpha = (self.last_epoch - self.decay_steps) / (self.total_steps - self.decay_steps)
            lr = self.end_lr - alpha * (self.end_lr - self.start_lr)
        return [lr for _ in self.optimizer.param_groups]


def train_model(
        model: PreTrainedModel | PeftModel,
        tokenizer: PreTrainedTokenizerBase,
        train_data_loader: DataLoader,
        dev_data_loader: DataLoader,
        args: argparse.Namespace,
        device: torch.device,
        model_path: str,
        no_wandb: bool,
) -> None:
    model = model.to(device)

    optimizer = get_optimizer(model, args.optimizer, args.learning_rate)

    scheduler = get_scheduler(
        scheduler_name=args.scheduler,
        optimizer=optimizer,
        epochs=args.epochs,
        total_steps=ceil(len(train_data_loader) / args.accumulate_grad_batches) * args.epochs,
    )

    model_saved = False
    best_epoch = 0
    best_f1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        logging.info("Epoch %d/%d", epoch + 1, args.epochs)
        for i, batch in enumerate(train_data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if "labels_attention_mask" in batch:
                labels = batch["labels_ids"].to(device)
                labels_attention_mask = batch["labels_attention_mask"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    decoder_attention_mask=labels_attention_mask,
                )
            else:
                labels = batch["labels"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            loss = outputs.loss
            loss = loss / args.accumulate_grad_batches
            loss.backward()

            if ((i + 1) % args.accumulate_grad_batches) == 0 or i == len(train_data_loader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scheduler.step()
                # get learning rate
                lr = scheduler.get_lr()[0]
                if not no_wandb:
                    wandb.log({"lr": lr})
                    wandb.log({"loss": loss.item()})
                optimizer.step()
                optimizer.zero_grad()

            if i % 10 == 0:
                logging.info("Step: %d Loss: %f", i, loss.item())

        torch.cuda.empty_cache()

        logging.info("Epoch finished, evaluating model")

        metrics = evaluate_model_token_classification(
            model=model,
            data_loader=dev_data_loader,
            device=device,
        )

        f1 = metrics["f1"]
        dev_loss = metrics.get("loss", 0.0)

        if not no_wandb:
            wandb.log({"f1": f1})
            if dev_loss > 0.0:
                wandb.log({"dev_loss": dev_loss})

        logging.info("Dev metrics: %s", str(metrics))
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            model_saved = True
            best_epoch = epoch + 1

    if not model_saved:
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

    logging.info("Best epoch: %d", best_epoch)
    logging.info("Best F1: %f", best_f1)
    if not no_wandb:
        wandb.log({"best_f1": best_f1})
        wandb.log({"best_epoch": best_epoch})
