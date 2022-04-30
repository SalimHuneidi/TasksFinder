from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import json
import torch


class DataSet(Dataset):

    def __init__(self, text, label, tokenizer, max_len):
        self.text = text
        self.label = label
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = dict()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = str(self.text[idx])
        label = self.label[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding=False,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def split(data, test_percentage, transpose=True, val_dataset=True, val_percentage=0.2):
    df_train, df_test = train_test_split(
        data, test_size=test_percentage, random_state=42)

    if val_dataset == True:
        df_train, df_val = train_test_split(
            df_train, test_size=val_percentage, random_state=42)
        df_val = np.array(df_val, dtype=object)
    else:
        df_val = []

    df_train = np.array(df_train, dtype=object)
    df_test = np.array(df_test, dtype=object)

    if(transpose == True):
        df_train = np.transpose(df_train)
        df_test = np.transpose(df_test)
        df_val = np.transpose(df_val)

    print(
        f" Training shape={df_train.shape} | Test shape={df_test.shape} | Validation Shape={df_val.shape}")
    return df_train, df_test, df_val


def read_csv(file_path):
    import pandas as pd
    data = pd.read_csv(file_path)
    print(data)


def read_json(file_path):
    import json
    ds = []
   # urls = []

    for line in open(file_path, 'r'):
        data = json.loads(line)
        # urls.append(data["article_link"])
        ds.append([data["headline"], int(data["is_sarcastic"])])
    return ds


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = DataSet(text=df[0], label=df[1], tokenizer=tokenizer, max_len=max_len)
    return ds


if __name__ == "__main__":
    read_csv("./Datsets/train-balanced-sarcasm.csv.nosync.csv")
    print("done")
