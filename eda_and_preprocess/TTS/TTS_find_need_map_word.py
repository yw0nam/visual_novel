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
comp = re.compile("\u3000|『|』")
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

df_non_chara = df_non_chara.drop_duplicates(subset='words')
df_non_chara = df_non_chara.drop(89)
df_chara_name = pd.read_excel('./../data/chara_names.xlsx')
# %%
def remove_word_and_replace_pronun(text):
    expr = re.compile("[[][ぁ-ゔァ-ヴ\sー]*[]]*")
    text_to_remove = []
    iter_obj = expr.finditer(text)
    
    for obj in iter_obj:
        text_to_remove.append(text[obj.end()])
        
    expr = '|'.join(text_to_remove+["[[]", "[]]"])
    regx_remove_text = re.compile(expr)
    return re.sub(regx_remove_text, "",  text)

def make_chara_mapping(df):
    chara_dicts=dict()
    df_chara = df.reset_index(drop=True)
    for i in range(len(df_chara)):
        chara_dicts[df_chara['First_name'].iloc[i]] = df_chara['First_name_pronun'].iloc[i]
        if df_chara['Last_name_present'].iloc[i] == 'Y':
            chara_dicts[df_chara['Last_name'].iloc[i]] = df_chara['Last_name_pronun'].iloc[i]
    return chara_dicts

def replace_proper_noun(text, comp):
    iter_obj = comp.finditer(text)
    match = comp.match(text)
    if match:
        for obj in iter_obj:
            text = text.replace(obj[0], mapping_table[obj[0]])
        return text
    else:
        return text
# %%

data_voice['normalized_text'] = data_voice['text'].map(lambda x: remove_word_and_replace_pronun(x))
# %%
game_ls = ['SenrenBanka', 'RiddleJoker', 'CafeStella']

# %%
series = []
for i in range(3):
    df_game_word = df_non_chara.query("game == @game_ls[%d]"%(i))
    df_game_chara = df_chara_name.query("game == @game_ls[%d]"%(i))
    mapping_table_word = dict(zip(df_game_word.words, df_game_word.pronun))
    mapping_table_chara = make_chara_mapping(df_game_chara)

    map_keys = list(mapping_table_word.keys()) + list(mapping_table_chara.keys())
    map_values = list(mapping_table_word.values()) + list(mapping_table_chara.values())
    mapping_table = dict(zip(map_keys, map_values))
    expr = '|'.join(map_keys)
    comp = re.compile(expr)
    series.append(data_voice.query("game_name == @game_ls[%d]"%(i))['normalized_text'].map(lambda x: replace_proper_noun(x, comp)))
    
# %%
data_voice['normalized_text'] = pd.concat(series)
# %%
data_voice.to_csv('./../data/data_voice.csv', index=False)
# %%
