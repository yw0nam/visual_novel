# %%
from tqdm import tqdm
from custom_dataset import *
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd 
from transformers import pipeline
# %%
tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v2')
model = AutoModelForSequenceClassification.from_pretrained('./model/checkpoints/checkpoint-80/', num_labels=2)

# %%
train_data = pd.read_csv('../data/data_for_lm.csv')
csv = pd.read_csv('../../my_coqui_TTS/data/data_voice.path_exist.csv')
names = csv['name'].value_counts().loc[csv['name'].value_counts()>900].index
csv = csv.query("name in @names")
csv['normalized_text'] = csv['normalized_text'].map(lambda x: x.replace('__', '_'))
# csv['normalized_text'] = csv['normalized_text'].map(lambda x: x.replace("芦花", "ロカ"))
# %%
dataset = MyDataset(csv)

loader = DataLoader(dataset, collate_fn=dataCollator(tokenizer,
                                512,
                                with_text=False), batch_size=16)
# %%

# %%
text_pipeline = pipeline(task='text-classification', model=model, tokenizer=tokenizer)
# %%
csv['predict'] = csv['text'].map(lambda x: text_pipeline([x]))
# %%
batch_size = 64
out_ls = []
for i in tqdm(range(int(len(csv) / batch_size)-1)):
    texts = csv['text'].iloc[batch_size*i:batch_size*(i+1)].to_list()
    out_ls = out_ls + text_pipeline(texts)
# %%
len(out_ls)
# %%
out = text_pipeline(csv['text'].iloc[batch_size*(i+1):].to_list())
# %%
final_out = out_ls + out
# %%
csv['pred'] = final_out
# %%
csv.to_csv('./../data/data_predict_nonuse.csv',index=False)
# %%
