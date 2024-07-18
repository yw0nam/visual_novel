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
import numpy as np
# %%
chara_bg_dicts = jload('./data/system_dict_updated.json')
system_message = """This is an RP (roleplay) chat. Our characters could come from all sorts of places--like movies, video games, books, or even anime. Make sure that you follow all my instructions and setup, based on the details to follow
I'm going to give you an character name, a background, and a {game_name}.
I want you to respond and answer like characters using the tone, manner and vocabulary characters would use. 
Here is Main Character's backgrounds.
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
min_context_window = 10
max_context_window = 20
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
        text += f"{name}:" + text_remove_yomigana
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
def make_masked_chat_prediction(mapped_text, characters, game_name, chara_bg_dicts, mask_portion=0.4):
    while True:
        flag = 0
        masked_ls =[mapped_text[0]]
        out_mapped_text = []
        mask_ = np.random.choice([0, 1], size=len(mapped_text), p=[1-mask_portion, mask_portion])
        for text, mask, chara in zip(mapped_text, mask_, characters):
            if mask and (chara in chara_bg_dicts): #Masking only main characters text
                if ':' in text:
                    index = text.find(':')
                    masked_ls.append(text[:index+1] +' [MASK]')
                else:
                    masked_ls.append('[MASK]')
                out_mapped_text.append(text)
                flag = 1
            else:
                masked_ls.append(text)
        if flag:
            break

    output = "\n".join([f"{i+1}: {out}" for i, out in enumerate(out_mapped_text)])
    chara_bgs = []
    for chara in characters:
        if chara in chara_bg_dicts:
            chara_bgs.append(chara_bg_dicts[chara])
    instruction = system_message.format(game_name=game_name) +"\n".join(chara_bgs) +"\n与えられた台詞を見て、[MASK]で省略されたキャラクターの台詞を上から順番に生成してください。"
    inputs = "\n".join(masked_ls)
    return instruction, inputs, output, 'fill_mask'

def make_novel_generate(mapped_text, characters, game_name, chara_bg_dicts, init_portion=0.3):
    
    init_index = int(len(mapped_text) * init_portion)
    init_mapped_text = mapped_text[:init_index]
    target_mapped_text = mapped_text[init_index:]
    chara_bgs = []
    for chara in characters:
        if chara in chara_bg_dicts:
            chara_bgs.append(chara_bg_dicts[chara])
    instruction = system_message.format(game_name=game_name) +"\n".join(chara_bgs) +"\n与えられたキャラクターを利用して、キャラクターの会話につながる会話を生成してください。"
    inputs = "\n".join(init_mapped_text)
    output = "\n".join(target_mapped_text)
    return instruction, inputs, output, 'novel_generate'
# %%
df = pd.DataFrame(out)
# %%
tqdm.pandas()
instruction, inputs, output, source = zip(*df.progress_apply((lambda x: 
    make_masked_chat_prediction(**x, chara_bg_dicts=chara_bg_dicts) 
    if random.randint(0,1) == 0  and any([character in chara_bg_dicts for character in list(set(x['characters']))]) else 
    make_novel_generate(**x, chara_bg_dicts=chara_bg_dicts) 
),axis=1))
    
# %%
temp = pd.DataFrame({'insturuction': instruction, 'input': inputs, 'output': output, 'source': source})
# %%
temp['source'].value_counts()
# %%
