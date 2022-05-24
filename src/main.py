
print('> Loading modules\n')
from utilities.translation_pipeline import EmbeddingsPipeline, prepare_data
from utilities.loaders import load_reviews, load_w2vec_model
from utilities.lstm import LSTM, loss_calc

from datasets import list_datasets, load_dataset, list_metrics, load_metric
import datasets

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import F1Score, Accuracy

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Flags
LOCAL_DATA = True # Reviews and models are downloaded if True
LOCAL_MODEL = True
TORCH_DATA_TRAIN_PATH = None # Path to the saved torch data
TORCH_DATA_VAL_PATH = None # Path to the saved torch data
FRAC_TRAINING_SAMPLES = .05 # Fraction of training samples to be used

def main():

    # --------- Training phase of BiLSTM on English dataset
    # Load training data
    training_data_en = load_reviews(LOCAL_DATA, 'train')
    reviews_train, target_train = prepare_data(
        df=training_data_en,
        language='en',
        frac_samples=FRAC_TRAINING_SAMPLES)
    
    # Load pretrained eng w2vec
    en_model = load_w2vec_model(
        "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec",
        LOCAL_MODEL
    )

    print('> Transforming the training data into tensors\n')
    if TORCH_DATA_TRAIN_PATH is not None:
        torch_data_train = torch.load(TORCH_DATA_TRAIN_PATH)
    else:
        embeddings_pipeline = EmbeddingsPipeline(
            w2vec_model=en_model,
            translate=False
        )
        torch_data_train = embeddings_pipeline.transform_reviews_to_emb(
            reviews=reviews_train,
            save_embeddings_path=f"data/embeddings/reviewtraincount_{len(reviews_train)}_en.pt"
        )

    print('> Getting the training loader object\n')
    td = TensorDataset(torch_data_train, target_train)
    training_loader = DataLoader(td, batch_size=5, shuffle=True)

    print('> Started Training')
    lstm = LSTM(batches_print=1)
    lstm.train(
       epochs=2,
       trainloader=training_loader
    )
    print('> Finished Training')

    # --------- Validation phase of BiLSTM on English dataset
    # -- load validation data
    validation_data_en = load_reviews(LOCAL_DATA, 'validation')
    reviews_val, target_val = prepare_data(
        df=validation_data_en,
        language='en',
        frac_samples=1)
    
    print('> Transforming the validation data into tensors\n')
    if TORCH_DATA_VAL_PATH is not None:
        torch_data_train = torch.load(TORCH_DATA_VAL_PATH)
    else:
        embeddings_pipeline = EmbeddingsPipeline(
            w2vec_model=en_model,
            translate=False
        )
        torch_data_val = embeddings_pipeline.transform_reviews_to_emb(
            reviews=reviews_val,
            save_embeddings_path=f"data/embeddings/reviewsvalcount_{len(reviews_val)}_en.pt"
        )

    print('> Getting the validationn loader object\n')
    td = TensorDataset(torch_data_val, target_val)
    val_loader = DataLoader(td, batch_size=len(torch_data_val), shuffle=True)
    
    print("> Validation performance")
    for valdata in val_loader:

        # Prepate input data
        x, y_true = valdata
        ytrue = y_true.flatten().to(torch.int8)

        # Compute yhat
        probs = torch.flatten(lstm.model(x))
        yhat = probs.detach().apply_(lambda prob: 1 if prob > .5 else 0).to(torch.int8)

        # Compute corresponding metrics
        f1 = F1Score(num_classes=2)
        print(f'>> F1: {f1(yhat, ytrue)}')

        accuracy = Accuracy(num_classes=2)
        print(f'>> Accuracy: {accuracy(yhat, ytrue)}')

    print()

if __name__ == '__main__':
    main()
