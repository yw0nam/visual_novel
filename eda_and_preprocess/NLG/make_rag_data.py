# %%
from tqdm import tqdm
import os, sys
os.environ['HF_HOME']="/data/research_data/cache"
os.environ['HF_DATASETS_CACHE']="/data/research_data/cache/datasets"
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
from transformers import BertJapaneseTokenizer, BertModel
import torch
class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                        truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)
# %%
# model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
model = SentenceBertJapanese('sonoisa/sentence-bert-base-ja-mean-tokens-v2')
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
                encoding = model.encode([sentence.split(':')[1]]).squeeze()
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
        name="chat_RAG_jpbert",
        metadata={"hnsw:space": "cosine"}
    )
except:
    collection = client.get_collection('chat_RAG_jpbert')
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
collection.query(model.encode(['いいから。ほら早くしないと、バレるぞ？']).tolist(), n_results=5)

# %%
