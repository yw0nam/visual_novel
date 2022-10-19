# %%
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# %%
# check jp ponts
import matplotlib
import matplotlib.font_manager as fm
print(matplotlib.matplotlib_fname())
#fm._rebuild()
[f for f in fm.fontManager.ttflist if 'Noto' in f.name]
# %%
matplotlib.rcParams['font.family'] = ['Noto Sans CJK JP']
# %%
csv = pd.read_csv(
    './../../../mfa/data/metadata/visual_novel/data.remove_wrongstrings.replace_noun.filter_only_dict.csv', index_col=0)

# %%
def cal_mean(root_path, char, path):
    
    basename = os.path.basename(path)[:-4]
    pitch = np.load(os.path.join(root_path, '%s-pitch-%s.npy'%(char, basename)))
    
    nonzero_ids = np.where(pitch != 0)[0]
    interp_fn = interp1d(
        nonzero_ids,
        pitch[nonzero_ids],
        fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
        bounds_error=False,
    )
    pitch = interp_fn(np.arange(0, len(pitch)))
    return (pitch.mean(), pitch.std())

# %%
root_path = './../../data/TTS/preprocessed/pitch/'
means = []
stds = []

for i in tqdm(range(len(csv))):
    try:
        mean, std = cal_mean(root_path, csv['name'].iloc[i], csv['path'].iloc[i])
        means.append(mean)
        stds.append(std)
    except:
        means.append(np.nan)
        stds.append(np.nan)
# %%
csv['pitch_mean'] = means
csv['pitch_std'] = stds
# %%
csv = csv[~csv['pitch_mean'].isna()]
# %%
scaler = StandardScaler()
scaler.fit(csv['pitch_mean'].to_numpy().reshape(-1, 1))
# %%
csv['norm_pitch_mean'] = scaler.transform(csv['pitch_mean'].to_numpy().reshape(-1, 1))
# %%
sns.displot(data=csv.query("game_name=='SenrenBanka'"),
            x='norm_pitch_mean', kind='kde', hue='name')
# %%
sns.displot(data=csv.query("game_name=='CafeStella'"),
            x='norm_pitch_mean', kind='kde', hue='name')
# %%
sns.displot(data=csv.query("game_name=='RiddleJoker'"),
            x='norm_pitch_mean', kind='kde', hue='name')
# %%
csv.query("norm_pitch_mean <= 1.6").to_csv(
    './../../data/data.remove_wrongstrings.replace_noun.filter_only_dict.remove_high_pitch.csv', index=False)

# %%
