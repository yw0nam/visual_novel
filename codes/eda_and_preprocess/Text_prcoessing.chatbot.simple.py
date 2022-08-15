# %%
import re
import pandas as pd
import numpy as np
from glob import glob

data = pd.read_csv('/mnt/hdd/spow12/visual_novel/data.csv')
# %%
comp_1 = re.compile("[[][\s0-9ぁ-ゔァ-ヴ々〆〤一-龥ー,\s]*[]]")
comp_2 = re.compile("[[][・][]]")
data['text_remove_yomigana'] = data['text'].map(lambda x: re.sub(comp_1, '', x))
data['text_remove_yomigana'] = data['text_remove_yomigana'].map(lambda x: re.sub(comp_2, '', x)) 
# %%
comp_3 = re.compile("[『]|[』]")
data['text_remove_yomigana'] = data['text_remove_yomigana'].map(lambda x: re.sub(comp_3, '', x)) 

# %%
data_conversation = data.query("dialog_type == 'conversation'")
# %%
data_conversation.head(20)
# %%
temp = []
for i in range(1, len(data_conversation)- 1):
    if data_conversation['name'].iloc[i] != data_conversation['name'].iloc[i-1]:
        temp.append([data_conversation['name'].iloc[i-1],
                               data_conversation['text_remove_yomigana'].iloc[i-1],
                               data_conversation['name'].iloc[i],
                               data_conversation['text_remove_yomigana'].iloc[i]])
# %%
pd.DataFrame(temp).to_csv('./data/QA.simple.tsv', sep='\t', index=False, header=None)
# %%
