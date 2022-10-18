import pandas as pd

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

