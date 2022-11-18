# %%
import re
import pandas as pd
from sudachipy import tokenizer, dictionary
from utils import make_chara_mapping, replace_noun
# %%
mode_a = tokenizer.Tokenizer.SplitMode.A
tokenizer_obj = dictionary.Dictionary().create()
data = pd.read_csv('../../data/data_remove_wrongstrings_final.csv')
# %%
df = pd.read_excel('../../data/temp_2.xlsx')
df_non_chara = df.query("need_map == 'Y' and etc != 'Chara_name'")

df_non_chara = df_non_chara.drop_duplicates(subset='words')
df_non_chara = df_non_chara.drop(91)
df_chara_name = pd.read_excel('../../data/chara_names.xlsx')

game_ls = ['SenrenBanka', 'RiddleJoker', 'CafeStella']
# %%
for i in range(3):
    lexicon = {}
    map_keys = []
    map_values = []
    
    df_game_word = df_non_chara.query("game == @game_ls[%d]" % (i))
    df_game_chara = df_chara_name.query("game == @game_ls[%d]" % (i))
    mapping_table_word = dict(zip(df_game_word.words, df_game_word.pronun))
    mapping_table_chara = make_chara_mapping(df_game_chara)

    map_keys = map_keys + list(mapping_table_word.keys()) + \
        list(mapping_table_chara.keys())
    map_values = map_values + list(mapping_table_word.values()) + \
        list(mapping_table_chara.values())
    
    for j in range(len(map_values)):
        lexicon[map_keys[j]] = "".join(
            [m.reading_form() for m in tokenizer_obj.tokenize(map_values[j], mode_a)])
        
    data['normalized_text'] = data.apply(lambda x: replace_noun(x['normalized_text'], lexicon, x['game_name'], game_ls[i]), axis=1)
# %%
data.to_csv('./../../data/data.remove_wrongstrings.replace_noun.csv',index=False)
# %%
