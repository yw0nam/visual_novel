import pandas as pd
import pyworld as pw
import librosa
import numpy as np
from scipy.interpolate import interp1d
import os

def make_chara_mapping(df):

    chara_dicts = dict()
    df_chara = df.reset_index(drop=True)
    for i in range(len(df_chara)):
        chara_dicts[df_chara['First_name'].iloc[i]
                    ] = df_chara['First_name_pronun'].iloc[i]
        if df_chara['Last_name_present'].iloc[i] == 'Y':
            chara_dicts[df_chara['Last_name'].iloc[i]
                        ] = df_chara['Last_name_pronun'].iloc[i]
    return chara_dicts

def replace_noun(x, lexicons, col_gamename, gamename):
    if col_gamename == gamename:
        for key in lexicons.keys():
            x = x.replace(key, lexicons[key])
    return x

def extract_pitch_energy(out_dir, config, speaker, basename, wav_path, STFT=None):
    
    wav, _ = librosa.load(wav_path)
    wav = wav.astype(np.float32)

    # Compute fundamental frequency
    pitch, t = pw.dio(
        wav.astype(np.float64),
        config["preprocessing"]["audio"]["sampling_rate"],
        frame_period=config["preprocessing"]["stft"]["hop_length"] / config["preprocessing"]["audio"]["sampling_rate"] * 1000,
    )
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, config["preprocessing"]["audio"]["sampling_rate"])
    
    nonzero_ids = np.where(pitch != 0)[0]
    interp_fn = interp1d(
        nonzero_ids,
        pitch[nonzero_ids],
        fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
        bounds_error=False,
    )
    pitch = interp_fn(np.arange(0, len(pitch)))
    
    pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
    np.save(os.path.join(out_dir, "pitch", pitch_filename), pitch)

    # energy_filename = "{}-energy-{}.npy".format(speaker, basename)
    # np.save(os.path.join(out_dir, "energy", energy_filename), energy)