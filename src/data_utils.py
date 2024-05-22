import argparse
import ast
import functools
import logging
from pathlib import Path

import pandas as pd
import torch
from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from src.config import SYSTEM_PROMPT, PROMPT, LLM_LABEL_TEXT, RANDOM_SEED, LANGUAGES


def create_dataset_dict(
        args: argparse.Namespace,
        train_data: pd.DataFrame,
        dev_data: pd.DataFrame | None,
        test_data: pd.DataFrame,
) -> tuple[DatasetDict, pd.DataFrame]:
    # Limit the number of data points
    if args.max_train_data is not None:
        train_data = train_data.head(args.max_train_data)
    if dev_data is not None and args.max_dev_data is not None:
        dev_data = dev_data.head(args.max_dev_data)
    if args.max_test_data is not None:
        test_data = test_data.head(args.max_test_data)
    datasets = DatasetDict(
        {
            "train": Dataset.from_pandas(train_data),
            "dev": Dataset.from_pandas(dev_data) if dev_data is not None else None,
            "test": Dataset.from_pandas(test_data),
        }
    )
    return datasets, test_data


def prepare_datasets(args: argparse.Namespace) -> tuple[DatasetDict, set[str], pd.DataFrame]:
    data_path = Path("data")
    train_file_path = data_path / "exalt_emotion_train.tsv"
    dev_file_path = data_path / "exalt_emotion_dev_participants.tsv"
    train_file = pd.read_csv(train_file_path, sep="\t", encoding="utf-8")
    dev_file = pd.read_csv(dev_file_path, sep="\t", encoding="utf-8")
    unique_labels = train_file["Labels"].unique().tolist()
    logging.info("Unique labels: %s", str(unique_labels))
    unique_labels = set(unique_labels)
    train_data, dev_data = train_test_split(train_file, test_size=0.1, random_state=RANDOM_SEED)

    if args.load_test_data:
        test_data = pd.read_csv(data_path / "exalt_emotion_test_participants.tsv", sep="\t", encoding="utf-8")
    else:
        test_data = dev_file.copy()

    if args.train_translated:
        for lang in LANGUAGES:
            translated_file_part = data_path / f"exalt_emotion_train_{lang}.tsv"
            translated_file = pd.read_csv(translated_file_part, sep="\t", encoding="utf-8")
            train_data = pd.concat([train_data, translated_file])

        # shuffle the data
        train_data = train_data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    datasets, test_data = create_dataset_dict(args=args, train_data=train_data, dev_data=dev_data, test_data=test_data)
    return datasets, unique_labels, test_data


def prepare_datasets_task2(args: argparse.Namespace) -> tuple[DatasetDict, pd.DataFrame]:
    data_path = Path("data")
    train_file_path = data_path / "exalt_triggers_train.tsv"
    dev_file_path = data_path / "exalt_triggers_dev_participants.tsv"
    train_file = pd.read_csv(train_file_path, sep="\t", encoding="utf-8")
    dev_file = pd.read_csv(dev_file_path, sep="\t", encoding="utf-8")
    train_data, dev_data = train_test_split(train_file, test_size=0.1, random_state=RANDOM_SEED)


    if args.load_test_data:
        test_data = pd.read_csv(data_path / "exalt_triggers_test_participants.tsv", sep="\t", encoding="utf-8")
    else:
        test_data = dev_file.copy()

    if args.train_translated:
        for lang in LANGUAGES:
            translated_file_pat = data_path / f"exalt_triggers_train_{lang}.tsv"
            translated_file = pd.read_csv(translated_file_pat, sep="\t", encoding="utf-8")
            train_data = pd.concat([train_data, translated_file])

        # shuffle the data
        train_data = train_data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    if args.train_acs:
        for lang in LANGUAGES:
            acs_file_pat = data_path / f"exalt_triggers_train_en_{lang}.tsv"
            acs_file = pd.read_csv(acs_file_pat, sep="\t", encoding="utf-8")
            train_data = pd.concat([train_data, acs_file])
            acs_file_pat = data_path / f"exalt_triggers_train_{lang}_en.tsv"
            acs_file = pd.read_csv(acs_file_pat, sep="\t", encoding="utf-8")
            train_data = pd.concat([train_data, acs_file])

        # shuffle the data
        train_data = train_data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Limit the number of data points
    datasets, test_data = create_dataset_dict(args=args, train_data=train_data, dev_data=dev_data, test_data=test_data)
    return datasets, test_data


def preprocess_dataset_llms_instruction_tuning(examples, tokenizer: PreTrainedTokenizerBase) -> dict:
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{PROMPT}\nInput: \"\"\"{examples['Texts']}\"\"\""},
    ]
    template = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    template_with_labels = f'{template}{LLM_LABEL_TEXT} {examples["Labels"]}{tokenizer.eos_token}'

    tokenized_template = tokenizer(
        template_with_labels,
        return_tensors="pt",
        add_special_tokens=False,
        return_attention_mask=True,
    )

    return {
        "input_ids": tokenized_template["input_ids"].squeeze(),
        "attention_mask": tokenized_template["attention_mask"].squeeze(),
    }


def preprocess_dataset_llms_testing(examples, tokenizer: PreTrainedTokenizerBase) -> dict:
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{PROMPT}\nInput: \"\"\"{examples['Texts']}\"\"\""},
    ]

    tokenized_template = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )

    return {
        "input_ids": tokenized_template["input_ids"].squeeze(),
        "attention_mask": tokenized_template["attention_mask"].squeeze(),
        "text": examples["Texts"],
    }


def data_collate_llm_dataset(batch: list[dict], tokenizer: PreTrainedTokenizerBase) -> dict:
    """
    Collate function for DataLoader.

    :param batch: batch of data
    :param tokenizer: tokenizer
    :return: batch of data
    """
    texts = []
    examples = [{"input_ids": sample["input_ids"], "attention_mask": sample["attention_mask"]} for sample in batch]
    padded = tokenizer.pad(examples, return_tensors="pt")
    for sample in batch:
        texts.append(sample["text"])

    return {
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"],
        "text": texts,
    }


def data_collate_token_classification(
        features: list[dict],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int,
) -> dict:

    batch = {}
    tokenized_texts = tokenizer(
        [feature["Texts"] for feature in features],
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True,
    )

    batch["input_ids"] = tokenized_texts["input_ids"]
    batch["attention_mask"] = tokenized_texts["attention_mask"]

    if "Labels" in features[0]:
        batch["labels"] = []
        tokenized_length = tokenized_texts["input_ids"].shape[1]
        for j, feature in enumerate(features):
            # token classification, labels are in the form of list of labels, where 1 is the label and 0 is not, add -100 for special tokens and also for tokens that are not start of the word
            labels = ast.literal_eval(feature["Labels"])
            text = feature["Texts"]
            tokens = tokenizer.tokenize(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_seq_len,
                padding=False,
            )

            labels_index = 0
            label = [-100 for _ in range(tokenized_length)]

            for i, token in enumerate(tokens):
                if (not token.startswith("▁")
                    or token == tokenizer.cls_token or token == tokenizer.sep_token or token == tokenizer.pad_token):
                    continue
                if labels[labels_index] == 1:
                    label[i] = 1
                else:
                    label[i] = 0
                labels_index += 1
            batch["labels"].append(label)
        batch["labels"] = torch.tensor(batch["labels"])
    else:
        batch["text"] = [feature["Texts"] for feature in features]
        batch["labels"] = []
        tokenized_length = tokenized_texts["input_ids"].shape[1]
        for j, feature in enumerate(features):
            # token classification, labels are in the form of list of labels, where 1 is the label and 0 is not, add -100 for special tokens and also for tokens that are not start of the word
            text = feature["Texts"]
            tokens = tokenizer.tokenize(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_seq_len,
                padding=False,
            )

            labels_index = 0
            label = [-100 for _ in range(tokenized_length)]

            for i, token in enumerate(tokens):
                if (not token.startswith("▁")
                        or token == tokenizer.cls_token or token == tokenizer.sep_token or token == tokenizer.pad_token):
                    continue
                label[i] = 0
                labels_index += 1
            batch["labels"].append(label)
        batch["labels"] = torch.tensor(batch["labels"])

    return batch


def prepare_data_loaders(
        batch_size: int,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        datasets: DatasetDict,
):
    data_collator = functools.partial(
        data_collate_token_classification,
        tokenizer=tokenizer,
        max_seq_len=max_seq_length,
    )

    train_data_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=True,
        drop_last=False,
    )
    dev_data_loader = DataLoader(
        datasets["dev"],
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False,
        drop_last=False,
    )
    test_data_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False,
        drop_last=False,
    )
    return dev_data_loader, test_data_loader, train_data_loader
