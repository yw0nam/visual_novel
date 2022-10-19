import os
import numpy as np
import pandas as pd
import yaml
from utils import extract_pitch_energy
from tqdm import tqdm

csv = pd.read_csv('./../../../mfa/data/metadata/visual_novel/data.remove_wrongstrings.replace_noun.filter_only_dict.csv', index_col=0)
config = yaml.load(
    open("./config/preprocess.yaml", "r"), Loader=yaml.FullLoader
)

tqdm.pandas()

out_path = '../../data/TTS/preprocessed/'
os.makedirs(os.path.join(out_path,'pitch'), exist_ok=True)

csv.progress_apply(lambda x: extract_pitch_energy(
    out_path, config, x['name'], x['voice'], x['path']
    )
    ,axis=1
)