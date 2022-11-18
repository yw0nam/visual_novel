# %%
import re
import pandas as pd
from sudachipy import tokenizer, dictionary
from glob import glob
import unicodedata
#%%
mode_a = tokenizer.Tokenizer.SplitMode.A
tokenizer_obj = dictionary.Dictionary().create()

# %%
lab_pathes = glob('./../../../Adaspeech/raw_data/visual_novel.en_trim_dur/*/*.lab')

# %%
for path in lab_pathes:
    with open(path, "r") as f:
        raw_text = f.readline().strip("\n")
    with open(path, "w") as f1:
        text = " ".join([m.surface() for m in tokenizer_obj.tokenize(raw_text, mode_a)])
        f1.write(unicodedata.normalize("NFKC", text))
        # f1.write(unicodedata.normalize(
        #     "NFKC", raw_text.replace(b'\xe2\x80\x95'.decode(), b'\xe3\x83\xbc'.decode())))
# %%
# %%
" ".join([m.surface() for m in tokenizer_obj.tokenize(raw_text, mode_a)])
# %%
with open(lab_pathes[0], "r") as f:
    raw_text = f.readline().strip("\n")
# %%

with open(path, "r") as f:
    raw_text = f.readline().strip("\n")

# %%
