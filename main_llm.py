import argparse
import functools
import logging
import sys
import uuid
from math import ceil
from pathlib import Path

import bitsandbytes as bnb
import torch
import wandb
from datasets import DatasetDict
from peft import AutoPeftModelForCausalLM, LoraConfig
from torch.utils.data import DataLoader
from transformers import (BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
                          PreTrainedTokenizerBase)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from src.args_utils import init_args
from src.config import init_logging, BEST_MODEL_FOLDER, WANDB_PROJECT_NAME, WANDB_ENTITY
from src.data_utils import (prepare_datasets, preprocess_dataset_llms_instruction_tuning,
                            preprocess_dataset_llms_testing, data_collate_llm_dataset)
from src.evaluate import get_predictions_and_write_results
from src.task import Task


def find_target_modules(model) -> list[str]:
    # Initialize a Set to Store Unique Layers
    unique_layers = set()

    # Iterate Over All Named Modules in the Model
    for name, module in model.named_modules():
        # Check if the Module Type Contains 'Linear4bit'
        if "Linear4bit" in str(type(module)):
            # Extract the Type of the Layer
            layer_type = name.split('.')[-1]

            # Add the Layer Type to the Set of Unique Layers
            unique_layers.add(layer_type)

    # Return the Set of Unique Layers Converted to a List
    return list(unique_layers)


def main():
    init_logging()
    args = init_args()

    if not args.no_wandb:
        wandb.init(
            project=WANDB_PROJECT_NAME,
            entity=WANDB_ENTITY,
            config=vars(args),
        )

    use_cpu = True if args.use_cpu else True if not torch.cuda.is_available() else False
    device = torch.device("cpu" if use_cpu else "cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Using device: %s", device)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # load model in 4-bit precision
        bnb_4bit_quant_type="nf4",  # pre-trained model should be quantized in 4-bit NF format
        bnb_4bit_use_double_quant=True,  # Using double quantization as mentioned in QLoRA paper
        bnb_4bit_compute_dtype=torch.bfloat16,  # During computation, pre-trained model should be loaded in BF16 format
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    if tokenizer.chat_template is None:
        if "microsoft/Orca" in args.model:
            tokenizer.chat_template = "{{ bos_token }} {% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        else:
            raise ValueError("Chat template not defined for this model.")

    datasets, unique_labels, test_data = prepare_datasets(args)

    response_template = tokenizer.encode("\n<|im_start|>assistant\n", add_special_tokens=False)[2:]
    assistant_text = "<|im_start|> assistant\n"

    unique_id = uuid.uuid4() if args.id is None else args.id

    if not args.no_wandb:
        wandb.log({"unique_id": str(unique_id)})

    if not args.inference_only:

        model_path = instruction_tuning(
            args=args,
            bnb_config=bnb_config,
            datasets=datasets,
            response_template=response_template,
            tokenizer=tokenizer,
            unique_id=unique_id,
            use_cpu=use_cpu,
        )
    else:
        model_path = args.model

    datasets["test"] = datasets["test"].map(
        functools.partial(preprocess_dataset_llms_testing, tokenizer=tokenizer),
        batched=False,
    )

    # load best model
    best_model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )

    tokenizer.padding_side = "left"

    test_data_loader = DataLoader(
        datasets["test"],
        batch_size=1,
        collate_fn=functools.partial(data_collate_llm_dataset, tokenizer=tokenizer),
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )

    get_predictions_and_write_results(
        best_model=best_model,
        device=device,
        test_data=test_data,
        test_data_loader=test_data_loader,
        tokenizer=tokenizer,
        unique_id=unique_id,
        assistant_text=assistant_text,
        task=Task.TASK1,
        unique_labels=unique_labels,
    )

    if not args.no_wandb:
        wandb.finish()


def instruction_tuning(
        args: argparse.Namespace,
        bnb_config: BitsAndBytesConfig,
        datasets: DatasetDict,
        response_template: list[int],
        tokenizer: PreTrainedTokenizerBase,
        unique_id: uuid.UUID,
        use_cpu: bool,
):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config if not use_cpu else None,
        device_map="auto" if not use_cpu else "cpu",
        use_cache=False,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16 if not use_cpu else torch.float32,
    )
    model.config.pretraining_tp = 1
    tokenizer.padding_side = "right"
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        target_modules=find_target_modules(model),
        modules_to_save=None,
        bias="none",
        task_type="CAUSAL_LM",
    )
    datasets["train"] = datasets["train"].map(
        functools.partial(preprocess_dataset_llms_instruction_tuning, tokenizer=tokenizer),
        batched=False,
    )
    datasets["dev"] = datasets["dev"].map(
        functools.partial(preprocess_dataset_llms_instruction_tuning, tokenizer=tokenizer),
        batched=False,
    ) if datasets["dev"] is not None else None

    output_dir = "output-model"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulate_grad_batches,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=args.epochs,
        optim="paged_adamw_32bit",
        report_to=["wandb"] if not args.no_wandb else [],
        lr_scheduler_type="constant_with_warmup" if args.scheduler == "constant" else "cosine" if args.scheduler == "cosine" else "linear",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        weight_decay=0.001,
        bf16=True if not use_cpu else False,
        tf32=True if not use_cpu else False,
        save_strategy="steps",
        save_steps=100,
        eval_steps=100,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        evaluation_strategy="steps",
        use_cpu=use_cpu,
        remove_unused_columns=True,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        disable_tqdm=True,
        group_by_length=True,
        dataloader_drop_last=False,
    )
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    trainer = SFTTrainer(
        train_dataset=datasets["train"],
        eval_dataset=datasets["dev"],
        model=model,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
        dataset_text_field="input_ids",
        max_seq_length=1024,
        data_collator=collator,
    )

    best_model_dir = Path(BEST_MODEL_FOLDER)
    model_path = best_model_dir / f"best_model-{unique_id}"
    model_path = str(model_path)

    logging.info("Training...")
    trainer.train()
    trainer.save_model(model_path)
    logging.info("Training finished")

    return model_path


if __name__ == '__main__':
    # sys.argv.extend(['--model', 'microsoft/Orca-2-7b'])
    # sys.argv.extend(['--model', 'models/best_model-f5a305b7-a90a-4eb7-af2d-6d695c13db8c'])
    # # sys.argv.extend(['--model', 'microsoft/Orca-2-7b'])
    # # sys.argv.extend(['--max_test_data', '10'])
    # # sys.argv.extend(['--max_train_data', '200'])
    # # sys.argv.extend(['--max_dev_data', '20'])
    # sys.argv.extend(['--batch_size', '2'])
    # sys.argv.extend(['--accumulate_grad_batches', '8'])
    # sys.argv.extend(['--no_wandb'])
    # # sys.argv.extend(['--scheduler', "cosine"])
    # # # sys.argv.extend(['--use_cpu'])
    # sys.argv.extend(['--epochs', '1'])
    # sys.argv.extend(['--inference_only'])

    main()
