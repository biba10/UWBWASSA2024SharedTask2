import logging
import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.config import (
    BINARY_TRIGGERS_FILE, EMOTIONS_FILE, FILE_EXTENSION, LLM_LABEL_TEXT, NUMERICAL_TRIGGERS_FILE, OUTPUT_FOLDER,
)
from src.task import Task


def get_predictions_generative(
        model: PreTrainedModel,
        data_loader: DataLoader,
        tokenizer: PreTrainedTokenizerBase,
        device: torch,
        assistant_text: str,
        unique_labels: set[str],
) -> list[str]:
    model = model.to(device)
    model.eval()

    predictions = []
    all_texts = []

    data_loader_len = len(data_loader)

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
            )
            prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            texts = batch["text"]
            all_texts.extend(texts)

            for pred, text in zip(prediction, texts):
                pred = pred.split(assistant_text)[-1].strip()
                if LLM_LABEL_TEXT in pred:
                    pred = pred.split(LLM_LABEL_TEXT)[-1].strip()
                if pred not in unique_labels:
                    # try if some label is not substring of pred
                    found = False
                    for label in unique_labels:
                        if label in pred:
                            pred = label
                            found = True
                            break
                    if not found:
                        pred = "Neutral"
                predictions.append(pred)

            if i % 100 == 0:
                logging.info("Predicted %d/%d batches", i, data_loader_len)

    logging.info("Predictions:")
    for text, prediction in zip(all_texts, predictions):
        logging.info("%s – %s", text, prediction)

    return predictions


def evaluate_model_token_classification(
        model: PreTrainedModel | PeftModel,
        data_loader: DataLoader,
        device: torch,
):
    model.eval()

    predictions = []
    labels = []

    dev_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels_batch)
            dev_loss += outputs.loss.item()
            argmax_predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
            for argmax_prediction, batch_labels in zip(argmax_predictions, batch["labels"].cpu().numpy()):
                preds = []
                labs = []
                for prediction, label in zip(argmax_prediction, batch_labels):
                    if label != -100:
                        preds.append(prediction)
                        labs.append(label)
                predictions.append(preds)
                labels.append(labs)

    dev_loss /= len(data_loader)
    # log predictions
    logging.info("Predictions:")
    for label, prediction in zip(labels, predictions):
        logging.info("Label: %s Prediction: %s", label, prediction)

    # do it as a multilabel classification
    tp = 0
    fp = 0
    fn = 0

    for prediction, label in zip(predictions, labels):
        for pred, lab in zip(prediction, label):
            if pred == 1 and lab == 1:
                tp += 1
            elif pred == 1 and lab == 0:
                fp += 1
            elif pred == 0 and lab == 1:
                fn += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "loss": dev_loss}


def get_predictions_token_classification(
        model: PreTrainedModel | PeftModel,
        data_loader: DataLoader,
        device: torch.device,
):
    model = model.to(device)
    model.eval()

    predictions = []
    predictions_scores = []
    all_texts = []

    data_loader_len = len(data_loader)

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
            )
            texts = batch["text"]
            all_texts.extend(texts)

            argmax_predictions = outputs.logits.argmax(dim=-1).cpu().numpy()

            batch_scores = outputs.logits[:, :, 1].cpu().numpy()
            # filter out -100 labels
            labels = batch["labels"]
            for prediction, scores, label, text in zip(argmax_predictions, batch_scores, labels, texts):
                tokens = text.split(" ")
                # get only indexes where label != -100
                eval_positions = np.where(label != -100)[0]
                filtered_prediction = prediction[eval_positions].tolist()
                filtered_scores = scores[eval_positions].tolist()

                if not len(tokens) == len(filtered_prediction):
                    for i, token in enumerate(tokens):
                        if not token or token == "\u200d":
                            # add new 0 to the filtered prediction and scores at this position, new value, not overwrite
                            filtered_prediction.insert(i, 0)
                            filtered_scores.insert(i, 0.0)

                # do softmax on scores
                filtered_scores = (np.exp(filtered_scores) / np.sum(np.exp(filtered_scores))).tolist()

                predictions.append(filtered_prediction)
                predictions_scores.append(filtered_scores)

            if i % 10 == 0:
                logging.info("Predicted %d/%d batches", i, data_loader_len)

    logging.info("Predictions:")
    for text, prediction in zip(all_texts, predictions):
        logging.info("%s – %s", text, prediction)

    return predictions, predictions_scores


def get_predictions_and_write_results(
        best_model: PreTrainedModel,
        device: torch.device,
        test_data: pd.DataFrame,
        test_data_loader: DataLoader,
        tokenizer: PreTrainedTokenizerBase,
        unique_id: uuid.UUID,
        task: Task,
        assistant_text: str | None = None,
        unique_labels: set[str] | None = None,
):
    if task == Task.TASK1:
        predictions = get_predictions_generative(
            model=best_model,
            data_loader=test_data_loader,
            tokenizer=tokenizer,
            device=device,
            assistant_text=assistant_text,
            unique_labels=unique_labels,
        )
        predictions_scores = None
    else:
        predictions, predictions_scores = get_predictions_token_classification(
            model=best_model,
            data_loader=test_data_loader,
            device=device,
        )

    output_folder = Path(OUTPUT_FOLDER)
    output_folder.mkdir(parents=True, exist_ok=True)
    if task == Task.TASK1:
        output_file = output_folder / f"{EMOTIONS_FILE}-{unique_id}{FILE_EXTENSION}"
    else:
        output_file = output_folder / f"{BINARY_TRIGGERS_FILE}-{unique_id}{FILE_EXTENSION}"
        scores_output_file = output_folder / f"{NUMERICAL_TRIGGERS_FILE}-{unique_id}{FILE_EXTENSION}"
        # create deep copy of the test data
        test_data_scores = test_data.copy()
        test_data_scores["Labels"] = predictions_scores
        test_data_scores.to_csv(scores_output_file, sep="\t", index=False, encoding="utf-8")

    test_data["Labels"] = predictions
    test_data.to_csv(output_file, sep="\t", index=False, encoding="utf-8")
