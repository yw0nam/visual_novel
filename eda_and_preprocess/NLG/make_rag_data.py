# %%
from tqdm import tqdm
import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, root_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from sentence_transformers import SentenceTransformer
from utils import jload
import re
import logging
import pandas as pd
import chromadb
# %%
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
datas = jload('/data/research_data/dataset/visual_novel/llm/ver_1.2/train.json')
# %%
dicts = []
embeddings = []
for data in tqdm(datas):
    target_chara, target_sentence = data['chat_template'][2]['content'].split(':')
    context = data['chat_template'][1]['content'][63:].split('\n')
    for i, sentence in enumerate(reversed(context)):
        if '「' in sentence:
            if sentence.split(':')[0] != target_chara:
                encoding = model.encode(sentence.split(':')[1])
                break
    embeddings.append(encoding)
    dicts.append({
        "target_chara": target_chara,
        "target_sentence": target_sentence,
        'context': "\n".join(context),
        'query': sentence,
    })
# %%
logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO,
    datefmt='%m/%d/%Y %I:%M:%S %p',
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('/data/research_data/db/visual_novel/log/rag_log.log')
logger.addHandler(file_handler)

client = chromadb.PersistentClient(path="/data/research_data/db/visual_novel/data/")

# %%
try:
    collection = client.create_collection(
        name="chat_RAG",
        metadata={"hnsw:space": "cosine"}
    )
except:
    collection = client.get_collection('chat_RAG')
# %%
ids = list(map(lambda x: 'chat_RAG_' + str(x), list(range(len(dicts)))))
embeddings_ls = list(map(lambda x: x.tolist(), embeddings))
# %%
collection.upsert(
        embeddings=embeddings_ls,
        metadatas=dicts,
        ids=ids
    )
logger.info("upsert done")
logger.info(f"total collection count: {collection.count()}")
# %%
collection.query(model.encode('いいから。ほら早くしないと、バレるぞ？').tolist())
