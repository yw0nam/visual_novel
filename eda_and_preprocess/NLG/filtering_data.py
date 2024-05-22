# %%
import pandas as pd
import os
data = pd.read_csv("/home/wonjong/codes/2024_upper/visual_novel/data/data.remove_wrongstrings.replace_noun.filter_only_dict.remove_high_pitch.trim_dur.csv")
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
temp = pd.read_csv("/home/wonjong/codes/2024_upper/visual_novel/data/data.remove_wrongstrings.replace_noun.filter_only_dict.remove_high_pitch.csv")
# %%
root_path = "/data/research_data/dataset/visual_novel/visual_novel/"
# %%
data['path'] = data['path'].map(lambda x: os.path.join(root_path, "/".join(x.split('/')[5:])))
# %%
#%%
data.to_csv('/home/wonjong/codes/2024_upper/visual_novel/data/data.remove_wrongstrings.replace_noun.filter_only_dict.remove_high_pitch.trim_dur.csv',index=False)
# %%
import librosa
# %%
wav = librosa.load('/home/wonjong/codes/2023_second_half/TTS/DiffGAN-TTS/noz304_099.wav')
# %%
wav[0]
# %%
