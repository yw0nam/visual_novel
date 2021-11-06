# %%
import re
import pandas as pd
import numpy as np
from glob import glob

data = pd.read_csv('/mnt/hdd/spow12/visual_novel/data.csv')
# %%
# def conversation2monologue(text, dialog_type):
#     """
#     Change conversation2monologue that mapped wrong
#     Args:
#         text ([str]): [description]
#         dialog_type ([str]): [description]

#     Returns:
#         [type]: [description]
#     """
#     if dialog_type == 'conversation':
#         if '（' == text[0] and '）' ==text[-1]:
#             return 'monologue'
#         else:
#             return 'conversation'
#     else:
#         return dialog_type
    
# data['dialog_type'] = data.apply(lambda x: conversation2monologue(x['text'], x['dialog_type']), axis=1)
# %%
# conv = data.query("dialog_type == 'conversation'")
# conv[conv['text'].map(lambda x: '（' == x[0])]
# %%
a = data[data['text'].map(lambda x: '[' in x)]
# %%

# data[data['text'].map(lambda x: "龍成" in x)]
data['text'].loc[1239] = '名前は、龍成君。年齢は５歳で、髪は結構短め。青いＴシャツとハーフパンツか'
# data['text'].loc[11] = '志那都荘まで行って欲しいんですけど'
# data['text'].loc[73] = "ああ……志那都荘だっけ？"
# data['text'].loc[223] = 'すみません、客じゃないんです。鞍馬玄十郎はいますか？'
data['text'].loc[2613] = "あなたの担任になる[ちゅう]中[じょう]条比奈実です"
a = data[data['text'].map(lambda x: '[' in x)]
# %%
# Mapping dictionary


name_mapping = {"龍成":  "りゅうせい", 
 "玄十郎": "げんじゅうろう",
 "鞍馬": "くらま",
 "小春": "",
 "比奈実": "ひなみ"}


{ "志那都":"しなつ"}
# %%

a['text'].iloc[0]
# %%
comp = re.compile("[[][ぁ-ゔァ-ヴ\s]*[]].")
for i in range(20, 30):
    print(a['text'].iloc[i])
    temp = comp.findall(a['text'].iloc[i])
    print(temp, '\n')
# %%
data['split_text'] =  data['text'].map(lambda x: comp.findall(x))
# %%
data['len_split'] = data['split_text'].map(lambda x: len(x))

# %%
data['len_split'].value_counts()
# %%
print(data.query('len_split == 1' )['text'].iloc[0])
# %%

# %%
