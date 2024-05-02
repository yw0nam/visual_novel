# %%
import matplotlib.pyplot as plt
import re
import pandas as pd
# %%
data = pd.read_csv('/data/research_data/dataset/visual_novel/visual_novel/data.csv')
temp = pd.read_csv("./data/data.remove_wrongstrings.replace_noun.filter_only_dict.remove_high_pitch.csv")
data = pd.merge(data, temp[['voice', 'label']], how='left', left_on=['voice'], right_on=['voice'])
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
data['name'] = data['name'].replace({'昂晴': 'ユーザー', '暁': 'ユーザー', '将臣': 'ユーザー'})
data['name'] = data['name'].fillna('non_exist')
# %%
chara_data = data.query("label == 1")
chara_data = chara_data.query("name in @name_ls")
# %%
data['chapters'] = data['scene_name'].map(lambda x:  x.split('-')[0] if 'chapter' in x else x)
# %%
grouped = data.groupby(['game_name', 'chapters', 'name'])
# %%
data = pd.DataFrame(grouped.agg('size')).reset_index()
# %%
data = data.rename({0: 'count'},axis=1)
# %%
data.to_excel('./game_chapter_chara_counts.xlsx',index=False)
# %%
