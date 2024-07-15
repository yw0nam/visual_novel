# %%
import pandas as pd
import os
data = pd.read_csv("../../data/data.remove_wrongstrings.replace_noun.filter_only_dict.remove_high_pitch.trim_dur.csv")
# %%
data = data.dropna(subset='romazi_text')
data['jp_text_per_sec'] = data.apply(lambda x: len(x['normalized_text']) / x['duration'], axis=1)
data['en_text_per_sec'] = data.apply(lambda x: len(x['romazi_text']) / x['duration'], axis=1)
# %%
index = data.query("jp_text_per_sec < 2 or en_text_per_sec < 5").index
data = data.query("index not in @index")
# %%
data = data.query('pitch_std < 140')
data = data.query('pitch_mean < 480')
# %%
name_ls = ['芦花',
 '小春',
 'ムラサメ',
 '芳乃',
 '茉子',
 'レナ',
 '七海',
 'あやせ',
 '茉優',
 '羽月',
 '千咲',
 '希',
 'ナツメ',
 '栞那',
 '愛衣',
 '涼音']
# %%
df_ls = []
for name in name_ls:
    temp = data.query(f"name== '{name}'")
    df_ls.append(temp.iloc[:1000])

sampled_data = pd.concat(df_ls)

root_path = "/data2/datas/Speech/vn/visual_novel/"
sampled_data['path'] = sampled_data['path'].map(lambda x: os.path.join(root_path, "/".join(x.split('/')[5:])))
# %%
sampled_data.to_csv('./../../data/data_sampled.csv',index=False)
#%%
# %%
temp = pd.read_csv('./../../data/data.csv')
# %%
from glob import glob
# %%
pathes = glob('./../../../weights/sbv2/sample_1000/*/*.safe*')
# %%
for path in pathes:
    chara_name, basename = path.split('/')[-2], path.split('/')[-1], 
    os.rename(path, path.replace(basename, f"{chara_name}.safetensors"))
# %%
chara_name, basename = pathes[0].split('/')[-2], pathes[0].split('/')[-1], 
# %%
chara_name
# %%
data = pd.read_csv("../../../../../data/visual_novel/yuzusoft/data.csv")
# %%
# %%
