# %%
import re
import pandas as pd
import numpy as np
from glob import glob

data = pd.read_csv('/mnt/hdd/spow12/visual_novel/data.csv')
# %%
comp = re.compile("[[][ぁ-ゔァ-ヴ\s]*[]].")
data['text_remove_yomigana'] = data['text'].map(lambda x: re.sub(comp, '', x))
# %%
a = data[data['text'].map(lambda x: '[' in x)]
# %%
