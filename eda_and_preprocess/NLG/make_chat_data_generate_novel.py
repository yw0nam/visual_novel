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
Conversation Start:
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
max_context_window = 25
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
for name, group in grouped:
    idx = 0
    while idx < len(group):
        context_size = random.randint(min_context_window, max_context_window)
        temp_df = group[idx:idx+context_size]
        temp_df['mapped_text'] = temp_df.apply(lambda x: apply_fn(x['name'], x['text_remove_yomigana'], x['dialog_type']), axis=1)
        mapped_text = '\n'.join(temp_df['mapped_text'].to_list())
        out.append({
            'mapped_text': mapped_text,
            'characters': list(temp_df.name.unique()),
            'game_name': temp_df['game_name'].iloc[0]
        })
        idx += context_size
    break
# %%
while index < len(data):
    out = []
    context_size = random.randint(min_context_window, max_context_window)
    df = data.loc[index:index+context_size]
    if len(df['scene_name'].unique()) != 1:
        df = df.query(f"scene_name =='{df['scene_name'].iloc[0]}'")
        context_size = len(df)
    if len(df.game_name.unique()) != 1:
        index += context_size
        continue
    
    chara_ls = df['name'].unique()
    personas = []
    for chara in chara_ls:
        if chara in system_dict:
            personas.append(system_dict[chara])
    persona_setup = system_message.format({
        'game_name':df.game_name[0]
    })            
    persona_message = persona_setup + "\n".join(personas)
    for i in df.index:
        if df['dialog_type'][i] == 'monologue' and df['name'][i] == '':  # If, user's Monologue
            if out[-1]['role'] != 'user':
                out.append({
                    'role': 'user',
                    'content': f"{data.loc[i]['text_remove_yomigana']}",
                })
            else:
                out[-1]['content'] = f"{out[-1]['content']}\n{data.loc[i]['text_remove_yomigana']}"
            
        elif out[-1]['name'] == data.loc[j]['name']: # if same character saying continuously
            out[-1]['content'] = f"{out[-1]['content']}\n{data.loc[j]['name']}:{data.loc[j]['text_remove_yomigana']}"
            
        elif out[-1]['name'] != data.loc[j]['name'] and data.loc[index]['name'] != data.loc[j]['name']: # if diff character saying and is not target chara,
            if out[-1]['role'] == 'assistant':
                out.append({
                    'role': 'user',
                    'content': f"{data.loc[j]['name']}: {data.loc[j]['text_remove_yomigana']}",
                    'name':data.loc[j]['name']
                })
            else:
                out[-1]['name'] = data.loc[j]['name']
                out[-1]['content'] = out[-1]['content'] + "\n" +f"{data.loc[j]['name']}: {data.loc[j]['text_remove_yomigana']}"
            
        elif out[-1]['name'] != data.loc[j]['name'] and data.loc[index]['name'] == data.loc[j]['name']: # if diff character saying and is target chara,
            out.append({
                'role': 'assistant',
                'content': f"{data.loc[j]['name']}: {data.loc[j]['text_remove_yomigana']}",
                'name':data.loc[j]['name']
            })
        else:
            break_flag = 1
            break
    if break_flag:
        break
    prev_last_index = index
    out_ls.append({
        'chat_template': out,
        'character': data.loc[index]['name']
    })
# %%    
df = pd.DataFrame(out_ls)