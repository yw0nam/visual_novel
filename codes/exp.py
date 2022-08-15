# %%
import argparse
import random
from custom_dataset import *
from transformers import Trainer
from transformers import TrainingArguments
import torch
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# %%
tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v2')
model = AutoModelForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-v2', num_labels=2)
# %%
import pandas as pd 
# %%
csv = pd.read_csv('../data/data_for_lm.csv')
# %%
tokens = tokenizer(
    csv['text'][:10].to_list(),
    padding=True,
    truncation=True,
    return_tensors="pt",
    max_length=512
)
# %%
tokens
# %%
from sklearn.model_selection import train_test_split
# %%
train_test_split(csv, test_size=0.2, random_state=1004, stratify=csv['use'])
# %%
