# %%
import re
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
data = pd.read_csv('/mnt/hdd/spow12/visual_novel/data.csv')
# %%
comp_1 = re.compile("[[][\s0-9ぁ-ゔァ-ヴ々〆〤一-龥ー,\s]*[]]")
comp_2 = re.compile("[[][・][]]")
data['text_remove_yomigana'] = data['text'].map(lambda x: re.sub(comp_1, '', x))
data['text_remove_yomigana'] = data['text_remove_yomigana'].map(lambda x: re.sub(comp_2, '', x)) 
# %%
comp_3 = re.compile("[『]|[』]")
data['text_remove_yomigana'] = data['text_remove_yomigana'].map(lambda x: re.sub(comp_3, '', x)) 
data = data.query("dialog_type == 'conversation'")
data = data[data['text_remove_yomigana'] != "………"]
# %%
person = list(data['name'].value_counts().index[data['name'].value_counts() > 1700])
data['name_spt'] = data['name'].map(lambda x: '<%s>'%x if x in person else '<某>')
# %%
temp = []
context = []
texts = ""
max_context = 5
txt_index_lim = 4
for i in tqdm(range(1, len(data)-1)):
    texts += data['text_remove_yomigana'].iloc[i]
    if data['name'].iloc[i] == data['name'].iloc[i+1] and len(context) <= max_context and (data['text_idx'].iloc[i+1] - data['text_idx'].iloc[i]) <= txt_index_lim:        
        texts += ' '
    elif data['name'].iloc[i] != data['name'].iloc[i+1] and len(context) <= max_context and (data['text_idx'].iloc[i+1] - data['text_idx'].iloc[i]) <= txt_index_lim:
        context.append(data['name_spt'].iloc[i]+texts)
        texts = ""
    elif (data['text_idx'].iloc[i+1] - data['text_idx'].iloc[i]) > txt_index_lim:
        if len(context) != 0:
            temp.append(['</s>'.join(context), data['name_spt'].iloc[i] +texts])
        context = []
        texts = ""
    elif len(context) > max_context:
        temp.append(['</s>'.join(context), data['name_spt'].iloc[i] +texts])
        context = []
        texts = ""
# %%
temp.append(['</s>'.join(context), data['name_spt'].iloc[i+1] +data['text_remove_yomigana'].iloc[-1]])
# %%
t = pd.DataFrame(temp, columns=['context', 'response'])
# %%
t.to_csv('./data/QA.context.tsv', sep='\t', index=False, header=None)
# %%
temp = []
texts = ""
prev_texts = None
txt_index_lim = 4
for i in tqdm(range(1, len(data)-1)):
    texts += data['text_remove_yomigana'].iloc[i]
    if data['name'].iloc[i] == data['name'].iloc[i+1] and (data['text_idx'].iloc[i+1] - data['text_idx'].iloc[i]) <= txt_index_lim:        
        texts += ' '
    elif data['name'].iloc[i] != data['name'].iloc[i+1] and (data['text_idx'].iloc[i+1] - data['text_idx'].iloc[i]) <= txt_index_lim:
        if not prev_texts:
            prev_texts = data['name_spt'].iloc[i]+texts
        else:
            temp.append([prev_texts, data['name_spt'].iloc[i] +texts])
            prev_texts = data['name_spt'].iloc[i] +texts
        texts = ""
    elif (data['text_idx'].iloc[i+1] - data['text_idx'].iloc[i]) > txt_index_lim:
        texts = ""
        prev_texts = None
# %%
pd.DataFrame(temp).to_csv('./data/QA.long.tsv', sep='\t', index=False, header=None)
# %%
