from helper import read_json, split, create_data_loader
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch import nn
from transformers import TrainingArguments
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments, Trainer


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":

    BATCH_SIZE = 16
    MAX_LEN = 128

    ds = read_json(
        "./Datsets/news_dataset.nosync/Sarcasm_Headlines_Dataset_v2.json")

    df_train, df_test, df_val = split(ds[:100], 0.2, True, True, 0.2)

    class_names = ['Sarcastic', 'Genuine']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=2)

    train_data_loader = create_data_loader(
        df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(
        df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(
        df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    training_args = TrainingArguments(output_dir="test_trainer")

    metric = load_metric("accuracy")

    training_args = TrainingArguments(
        output_dir="test_trainer", evaluation_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data_loader,
        eval_dataset=val_data_loader,
        compute_metrics=compute_metrics,
    )
    trainer.train()
