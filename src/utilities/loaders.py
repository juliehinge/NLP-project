import pandas as pd
from datasets import load_dataset

def load_w2vec_model(url):

    """
    Apapted from:
    https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
    """

    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(f"data/langmodels/{local_filename}", 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)

def load_reviews(LOCAL_DATA):

    if LOCAL_DATA:
        print('> Loading dataset from the local file.\n')
        data = pd.read_csv('data/raw/raw.csv')
    else:
        print('> Downloading remote dataset (10 min)\n')
        data = pd.DataFrame(datasets.load_dataset('amazon_reviews_multi')["train"])
        data.to_csv('data/raw/raw.csv', sep=',', index=False)

    return data
