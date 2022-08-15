# %%
import re
from tqdm import tqdm
import pandas as pd 
# %%
data = pd.read_csv('./../../data/data_predict_nonuse.csv')
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
# %%
data.to_csv('./../../data/data_remove_wrongstrings_final.csv',index=False)
# %%
