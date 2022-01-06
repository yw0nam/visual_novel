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
def remove_word_and_replace_pronun(text):
    expr = re.compile("[[][ぁ-ゔァ-ヴ\sー]*[]]*")
    text_to_remove = []
    iter_obj = expr.finditer(text)
    
    for obj in iter_obj:
        text_to_remove.append(text[obj.end()])
        
    expr = '|'.join(text_to_remove+["[[]", "[]]"])
    regx_remove_text = re.compile(expr)
    return re.sub(regx_remove_text, "",  text)

# %%

data_voice['normalized_text'] = data_voice['text'].map(lambda x: remove_word_and_replace_pronun(x))
# %%
game_ls = ['SenrenBanka', 'RiddleJoker', 'CafeStella']
# %%
game_df = df_non_chara.query("game == @game_ls[0]")
# %%
text = data_voice[data_voice['text'].map(lambda x: '叢雨丸' in x)].iloc[0]['text']
# %%
expr = '|'.join(game_df['words'].to_list())
comp = re.compile(expr)
iter_obj = comp.finditer(text)
# %%
for obj in iter_obj:
    print(obj[0])
    
# %%
game_df.query("words == @obj[0]")['pronun'][0]
# %%
game_df['words'].to_list()
# %%
dict(zip(game_df.words, game_df.pronun))

# %%
