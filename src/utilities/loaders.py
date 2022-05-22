import pandas as pd
import datasets
from datasets import load_dataset
import gensim
import requests

def load_w2vec_model(url, LOCAL_MODEL):
    """Apapted from:
    https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
    """

    local_filename = url.split('/')[-1]
    filepath = f"data/langmodels/{local_filename}"

    if not LOCAL_MODEL:
        print('> Downloading word2vec models.\n')
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
    
    print('> Loading word2vec from gensim\n')
    model = gensim.models.KeyedVectors.load_word2vec_format(filepath, limit=10000)

    return model

def load_reviews(LOCAL_DATA):

    if LOCAL_DATA:
        print('> Loading dataset from the local file.\n')
        data = pd.read_csv('data/raw/raw.csv')
    else:
        print('> Downloading remote dataset (10 min)\n')
        data = pd.DataFrame(datasets.load_dataset('amazon_reviews_multi')["train"])
        data.to_csv('data/raw/raw.csv', sep=',', index=False)

    return data
