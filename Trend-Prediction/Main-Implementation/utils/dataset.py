import os
import torch
import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings('ignore')
os.environ['WANDB_DISABLED'] = 'true'

class Dataset:
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels if len(labels) > 0 else []

        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index] if len(self.labels) > 0 else []

        inputs = self.tokenizer.__call__(str(text),
                                        None,
                                        add_special_tokens=True,
                                        max_length=self.max_len,
                                        padding="max_length",
                                        truncation=True,
                                        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        # token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long) if len(self.labels) > 0 else []
        }


class MyDataset:
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels

        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]

        if self.labels is not None:
            label = self.labels[index]

        inputs = self.tokenizer.__call__(text,
                                        None,
                                        add_special_tokens=True,
                                        max_length=self.max_len,
                                        padding="max_length",
                                        truncation=True,
                                        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        # print(inputs.keys())

        if self.labels is not None:
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                "labels": torch.tensor(label, dtype=torch.long)
            }
        return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
            }


#--------------------------------
def get_dataset(data_set, data_dir):
    # Reading the 3 different datasets from the CSV files
    if data_set == 1:
        dataset = pd.read_csv(os.path.join(data_dir, 'tweet_emotions.csv'))
        dataset = dataset.drop(columns='tweet_id')
        return {"dataset1": dataset}
    elif data_set == 2:
        dataset = pd.read_csv(os.path.join(data_dir, 'tweet_emotions_1.csv'))
        return {"dataset2": dataset}
    else:
        dataset1 = pd.read_csv(os.path.join(data_dir, 'tweet_emotions.csv'))
        dataset1 = dataset1.drop(columns='tweet_id')
        dataset2 = pd.read_csv(os.path.join(data_dir, 'tweet_emotions_1.csv'))
        dataset3 = pd.read_csv(os.path.join(data_dir, 'reddit_emotions.csv'))
    
    return {"dataset1": dataset1, "dataset2": dataset2, "dataset3": dataset3}

def get_class_map():
    class_names = []
    with open('/home/siu856533724/code/source-code/Social-Networks/Trend-Prediction/DataSet/emotions.txt') as f:
        while True:
            line = f.readline()
            if not line:
                break
            # print(line.strip())
            class_names.append(line.strip())

    mapping = {}
    for i, line in enumerate(class_names):
        mapping[line] = i
    print(mapping)
    return mapping

if __name__ == '__main__':
     print("#Dataset")
    #  print(get_class_map())