# %%
import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, root_dir)
import datasets
import random
from tqdm import tqdm
import re
import pandas as pd
from  matplotlib import pyplot as plt
from utils import jdump, jload
# %%
system_dict = jload('./data/system_dict_updated.json')
system_message = """This is an RP (roleplay) chat. Our characters could come from all sorts of places--like movies, video games, books, or even anime. Make sure that you follow all my instructions and setup, based on the details to follow
I'm going to give you a character's name, a background, and a {game_name}.
I want you to respond and answer like characters using the tone, manner and vocabulary characters would use. 
"""
# %%
data = pd.read_csv('~/Desktop/data/visual_novel/yuzusoft/data.csv')
# %%
comp_1 = re.compile("[[][\s0-9ぁ-ゔァ-ヴ々〆〤一-龥ー,\s]*[]]")
comp_2 = re.compile("[[][・][]]")
data['text_remove_yomigana'] = data['text'].map(lambda x: re.sub(comp_1, '', x))
data['text_remove_yomigana'] = data['text_remove_yomigana'].map(lambda x: re.sub(comp_2, '', x)) 
# %%ß
comp_3 = re.compile("[『]|[』]")
data['text_remove_yomigana'] = data['text_remove_yomigana'].map(lambda x: re.sub(comp_3, '', x)) 
# data = data[data['text_remove_yomigana'] != "………"]
temp = data['name'].value_counts()[3:]
name_ls = temp[temp > 1500].index.to_list()
# %%
data['name'] = data['name'].replace({'昂晴': 'ユーザー', '暁': 'ユーザー', '将臣': 'ユーザー'})
data['name'] = data['name'].fillna('ユーザー')
# %%
min_context_window = 8
max_context_window = 15
out_ls = []
index = 0
# %%
grouped = data.groupby(['scene_name', 'game_name'])
# %%
out = []
def apply_fn(name, text_remove_yomigana, dialog_type):
    text = ""
    if dialog_type == 'monologue':
        text += text_remove_yomigana
    else:
        text += f"{name}:" + text_remove_yomigana + '\n'
    return text
for name, group in tqdm(grouped):
    idx = 0
    while idx < len(group):
        context_size = random.randint(min_context_window, max_context_window)
        temp_df = group[idx:idx+context_size]
        temp_df['mapped_text'] = temp_df.apply(lambda x: apply_fn(x['name'], x['text_remove_yomigana'], x['dialog_type']), axis=1)
        out.append({
            'mapped_text': temp_df['mapped_text'].to_list(),
            'characters': temp_df.name.to_list(),
            'game_name': temp_df['game_name'].iloc[0]
        })
        idx += context_size
# %%
out[0]    
# %%
df = pd.DataFrame(out_ls)