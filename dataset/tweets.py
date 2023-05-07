import pandas as pd
import re
import torch
import random
from torch.utils.data import DataLoader, Dataset


class Tweets(Dataset):
    def __init__(
            self,
            df,
            label='label',
            subset='train',
            options_name='bert-base-uncased',
            trans_augment=None,
    ):
        df = pd.read_csv(df)
        df = df[df['subset'] == subset]
        df = df[df[label].notna()]

        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', options_name)
        self.df = df
        self.subset = subset
        self.label = label
        self.trans_augment = trans_augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        item = self.df.iloc[i]

        if self.trans_augment:
            mid = random.sample(self.trans_augment, 1)[0]
            text = item['text'+'_'+mid]
        else:
            text = item['text']

        label = item[self.label]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")

        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), int(label)


class Tweets_apply(Dataset):
    def __init__(
            self,
            df,
            options_name = 'bert-base-uncased',
    ):
        df = pd.read_csv(df, low_memory=False, lineterminator="\n")

        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', options_name)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        item = self.df.iloc[i]
        text = item['full_text']

        try:
            text = text_url(text)
            inputs = self.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")
        except:
            print('empty text')
            inputs = self.tokenizer('For Friday, only 39 out of 670 occupied ICU beds were open to new patients.',
                                    padding='max_length', truncation=True, return_tensors="pt")

        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0)


def text_url(x):
    new_x = re.sub(r'http\S+', '', x)
    return new_x