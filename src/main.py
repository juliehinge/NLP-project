
from utilities.translation_pipeline import EmbeddingsPipeline, prepare_data
from utilities.loaders import load_reviews, load_w2vec_model
from utilities.lstm import LSTM, LstmModel

from datasets import list_datasets, load_dataset, list_metrics, load_metric
import datasets

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

import numpy as np
import warnings
warnings.filterwarnings('ignore')


# Flags related to the whole experiment
LOCAL_DATA = False # Reviews and models are downloaded if True
LOCAL_MODEL = False

# Flags related to training of BiLSTM
TORCH_DATA_TRAIN_PATH = None # Path to the saved torch data
TORCH_DATA_VAL_PATH = None # Path to the saved torch data
FRAC_TRAINING_SAMPLES = 1 # Fraction of training samples to be used
BATCH_SIZE = 25

# Flags related to test phase
RUN_FINAL_TEST = True # should be ran only once
FINAL_MODEL_PATH = None # Example: 'data/trainedmodels/May-24-2022-12.pt'



def main():

    # ----- PART 1: Train BiLSTM on english reviews
    # Training BiLSTM
    if FINAL_MODEL_PATH is not None:
        bilstm_model = LstmModel()
        bilstm_model.load_state_dict(torch.load(FINAL_MODEL_PATH))
        bilstm_model.eval()
    else:
        bilstm_model = train_bilstm()

    # ----- PART 2: Testing methods
    if RUN_FINAL_TEST:

        # Load german testing data
        test_data_de = load_reviews(LOCAL_DATA, 'test')
        reviews_test_de, target_test_de = prepare_data(
            df=test_data_de,
            language='de',
            frac_samples=1
        )
        print(f"> Size of test data: {len(reviews_test_de)}\n")

    
        method1(reviews_test_de, target_test_de, bilstm_model)
        method2(reviews_test_de, target_test_de, bilstm_model)

def method1(reviews_test_de, target_test_de, model):

    """
    Translate german reviews to english and then
    lookup corresponding vectors in aligned english w2vec
    """

    en_model = load_w2vec_model(
        "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec",
        LOCAL_MODEL
    )

    print('> Transforming the test data into tensors\n')
    embeddings_pipeline = EmbeddingsPipeline(
        w2vec_model=en_model,
        translate=True,
        from_language='de'
    )
    torch_data_test = embeddings_pipeline.transform_reviews_to_emb(reviews=reviews_test_de)

    print('> Getting the validationn loader object\n')
    td = TensorDataset(torch_data_test, target_test_de)
    test_loader = DataLoader(td, batch_size=len(reviews_test_de))

    print("> Test performance - method 1: translate reviews and use english w2vec")
    for testdata in test_loader:

        # Prepate input data
        x, y_true = testdata
        ytrue = y_true.flatten().to(torch.int8)

        # Compute yhat
        probs = torch.flatten(model(x))
        yhat = probs.detach().apply_(lambda prob: 1 if prob > .5 else 0).to(torch.int8)

        # Compute corresponding metrics
        yhat, ytrue = yhat.numpy(), ytrue.numpy()
        print(f'>> F1: {f1_score(ytrue, yhat)}')
        print(f'>> Accuracy: {accuracy_score(ytrue, yhat)}')

    print()

def method2(reviews_test_de, target_test_de, model):
    """
    Load german reviews and lookup their
    corresponding embeddings in aligned german w2vec
    """

    de_model = load_w2vec_model(
        "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.de.align.vec",
        LOCAL_MODEL
    )

    print('> Transforming the test data into tensors\n')
    embeddings_pipeline = EmbeddingsPipeline(
        w2vec_model=de_model,
        translate=False
    )
    torch_data_test = embeddings_pipeline.transform_reviews_to_emb(reviews=reviews_test_de)

    print('> Getting the validationn loader object\n')
    td = TensorDataset(torch_data_test, target_test_de)
    test_loader = DataLoader(td, batch_size=len(reviews_test_de))

    print("> Test performance - method 2: do NOT translate reviews and use german w2vec")
    for testdata in test_loader:

        # Prepate input data
        x, y_true = testdata
        ytrue = y_true.flatten().to(torch.int8)

        # Compute yhat
        probs = torch.flatten(model(x))
        yhat = probs.detach().apply_(lambda prob: 1 if prob > .5 else 0).to(torch.int8)

        # Compute corresponding metrics
        yhat, ytrue = yhat.numpy(), ytrue.numpy()
        print(f'>> F1: {f1_score(ytrue, yhat)}')
        print(f'>> Accuracy: {accuracy_score(ytrue, yhat)}')

    print()


def train_bilstm():

    # Load pretrained eng w2vec
    en_model = load_w2vec_model(
        "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec",
        LOCAL_MODEL
    )
    
    # Load training data
    training_data_en = load_reviews(LOCAL_DATA, 'train')
    reviews_train, target_train = prepare_data(
        df=training_data_en,
        language='en',
        frac_samples=FRAC_TRAINING_SAMPLES
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
    training_loader = DataLoader(td, batch_size=BATCH_SIZE, shuffle=True)

    # Load validation data
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

    print('> Getting the validation loader object\n')
    td = TensorDataset(torch_data_val, target_val)
    val_loader = DataLoader(td, batch_size=len(torch_data_val))

    # --------- Training phase of BiLSTM on English dataset
    print('> Started Training')
    lstm = LSTM(batches_print=5)
    lstm.train(
       epochs=10,
       trainloader=training_loader,
       valoader=val_loader
    )
    print('> Finished Training')


    # --------- Validation phase of BiLSTM on English dataset
    print("> Validation performance")
    x, y_true = next(iter(val_loader))
    ytrue = y_true.flatten().to(torch.int8)

    # Compute yhat
    probs = torch.flatten(lstm.model(x))
    yhat = probs.detach().apply_(lambda prob: 1 if prob > .5 else 0).to(torch.int8)

    # Compute corresponding metrics
    yhat, ytrue = yhat.numpy(), ytrue.numpy()
    print(f'>> F1: {f1_score(ytrue, yhat)}')
    print(f'>> Accuracy: {accuracy_score(ytrue, yhat)}')

    print()

    return lstm.model

if __name__ == '__main__':
    main()
