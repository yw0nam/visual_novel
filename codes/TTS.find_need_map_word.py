# %%
import re
import pandas as pd
import numpy as np
from glob import glob
import os, subprocess
from tqdm import tqdm
data = pd.read_csv('/data1/spow12/datas/visual_novel/data.csv')

# %%
data_voice = data.dropna(subset=['voice'])
data_voice = data_voice[data_voice['text'].map(lambda x: '[・]' not in x)]
comp = re.compile("\u3000")
data_voice['text'] = data_voice['text'].map(lambda x: re.sub(comp, '', x))
# %%
data_to_normalize = data_voice[data_voice['text'].map(lambda x: ']' in x)]
# %%
comp = re.compile("[[][ぁ-ゔァ-ヴ\sー]*[]]*")
pronun_dicts = []
for i in tqdm(range(len(data_to_normalize))):
    text = data_to_normalize['text'].iloc[i]
    iter_obj = comp.finditer(text)
    prev_end = None
    words = ""
    pronun = ""
    for iter in iter_obj:
        if prev_end == None:
            words += text[iter.end()]
            pronun += iter[0][1:-1]
        elif iter.start() - prev_end <= 2:
            words += text[iter.end()]
            pronun += iter[0][1:-1]
        elif iter.start() - prev_end >= 3:
            pronun_dicts.append({
                "words": words,
                "pronun":pronun,
                "game" : data_to_normalize['game_name'].iloc[i],
                "index": i
            })
            words = ""
            pronun = ""
        prev_end = iter.end()
    if words == "":
        words += text[iter.end()]
        pronun += iter[0][1:-1]
    pronun_dicts.append({
        "words": words,
        "pronun":pronun,
        "game" : data_to_normalize['game_name'].iloc[i],
        "index": i
    })
# %%
text = data_to_normalize['text'].iloc[i]
t = pd.DataFrame(pronun_dicts)
# %%
df = pd.read_excel('./../data/temp_2.xlsx')

df_non_chara = df.query("need_map == 'Y' and etc != 'Chara_name'")
# %%
df_non_chara = df_non_chara.drop_duplicates(subset='words')
# %%
df_non_chara = df_non_chara.drop(89)
# %%
# %%

data_voice['normalized_text'] = data_voice['text'].map(lambda x: re.sub(comp, "", x))
# %%
expr = re.compile("[[][ぁ-ゔァ-ヴ\sー]*[]]*")
def remove_word_and_replace_pronun(text, expr=expr):
    iter_obj = expr.finditer(text)
    for obj in iter_obj:
        
# for i in tqdm(range(len(data_to_normalize))):