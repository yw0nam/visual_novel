# %%
import re
import pandas as pd
import unicodedata
# %%
data = pd.read_csv('./../../data/data_predict_nonuse.csv')
data['normalized_text'] = data['normalized_text'].map(lambda x: x.replace('_', '……'))# # %%
# %%
data = data[data['normalized_text'].map(lambda x: '\\n' not in x)]
# %%
data['normalized_text'] = data['normalized_text'].map(lambda x: re.sub("\d*;「", "", x))
data['normalized_text'] = data['normalized_text'].map(lambda x: re.sub("\d*;「", "", x))
data['normalized_text'] = data['normalized_text'].map(lambda x: re.sub("%\d*;", "", x))
data['normalized_text'] = data['normalized_text'].map(lambda x: re.sub("\d*;", "", x))
data['normalized_text'] = data['normalized_text'].map(lambda x: re.sub("%;", "", x))
# %%
data['label'] = data['label'].replace({'LABEL_0': 0, 'LABEL_1':1})
data = data[data['normalized_text'].map(lambda x: "●" not in x)]
data = data.query("label == 1")
# %%
data['normalized_text'] = data['normalized_text'].map(lambda x: unicodedata.normalize('NFKC', x))
data.to_csv('./../../data/data_remove_wrongstrings_final.csv',index=False, encoding='utf-8')
# %%
