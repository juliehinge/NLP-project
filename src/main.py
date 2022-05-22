
print('> Loading modules\n')

from utilities.translation_pipeline import Pipe
from utilities.loaders import load_reviews, load_w2vec_model
from utilities.lstm import LSTM, loss_calc

import warnings
#from datasets import load_dataset

from datasets import list_datasets, load_dataset, list_metrics, load_metric
import datasets

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

warnings.filterwarnings('ignore')

# Flags
LOCAL_DATA = True # Reviews and models are downloaded if True
LOCAL_MODEL = True

def main():

    # --------- Training phase
    data = load_reviews(LOCAL_DATA, 'train')
    de_model = load_w2vec_model(
        "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.de.align.vec",
        LOCAL_DATA
    )


    #en_model = load_w2vec_model(
     #   "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec",
      #  LOCAL_MODEL
    #)


    print('> Map stars into corresponding sentiment\n')
    data['sentiment'] = data.stars.replace({4:1, 5:1, 1:0, 2:0})

    print('> Prepare german dataset for tokenization\n')
    german_data = data.loc[(data['language']=='de') & (data['stars']!= 3),['review_body', 'language','sentiment', 'product_category']]
    ge_df = german_data.loc[(german_data['product_category']=='home'),['review_body', 'language','sentiment', 'product_category']]

    print('> Transforming the data into tensors')
    pipe = Pipe(ge_df[:10],'german','de',de_model,128)
    torch_data = pipe.emb()
    print(f"Torch data shape: {torch_data.shape}\n")
    
    print('> Putting data into DataLoader\n')
    target = ge_df['sentiment'].to_numpy()[:10]
    target = torch.from_numpy(target).float().reshape(-1,1)
    td = TensorDataset(torch_data, target)
    training_loader = DataLoader(td, batch_size=5, shuffle=True)

    lstm = LSTM()
    lstm.train(
        epochs=2,
        trainloader=training_loader
    )

    print("> Training performance")
    for i, traindata in enumerate(training_loader):
        x, y_true = traindata
        probs = torch.flatten(lstm.model(x))
        yhat = probs.detach().apply_(lambda prob: 1 if prob > .5 else 0)
        correct = (yhat==y_true.flatten()).sum().item()
        print(f'>> Batch {i} | Accuracy: {correct/y_true.size(0)}')
    print()

    # --------- Validation phase
    # TODO: change training loader to validation_loader



if __name__ == '__main__':
    main()
