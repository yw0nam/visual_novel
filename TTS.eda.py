# %%
import re
import pandas as pd
import numpy as np
from glob import glob
import os, subprocess
from tqdm import tqdm
data = pd.read_csv('/mnt/hdd/spow12/visual_novel/data.csv')

# %%
data_voice = data.dropna(subset=['voice'])
# %%
riddle_joker_root = '/mnt/hdd/spow12/visual_novel/RiddleJoker/voice/'
riddle_joker_save = '/mnt/hdd/spow12/visual_novel/RiddleJoker/wav/'
riddle_joker = data_voice[data_voice['game_name']=='RiddleJoker']
# %%
riddle_joker_voices = glob(riddle_joker_root+'/*.ogg')
# %%
for path in tqdm(riddle_joker_voices):
    command = ("ffmpeg -i {input_path} -b 768k -ac 1 -ar 48000 {output_path}.wav".format(input_path=path, 
                                                                                     output_path=os.path.join(riddle_joker_save, os.path.basename(path).split('.')[0])))
    output = subprocess.call(command, shell=True, stdout=None)

# %%
riddle_joker_voices = glob(riddle_joker_save+'/*')
# %%
len(riddle_joker_voices)
# %%
