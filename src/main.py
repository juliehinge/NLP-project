print('> Loading modules\n')
from utilities.translation_pipeline import Pipe
from utilities.loaders import load_reviews

import gensim
from gensim.models import Word2Vec

import warnings
warnings.filterwarnings('ignore')

# Flags
LOCAL_DATA = True

def main():

    data = load_reviews(LOCAL_DATA)

    # TODO: add to readme that these need to be downloaded and put to the langmodel folder
    # or try to find a function that does this automatically
    print('> Loading germans word2vec from gensim\n')
    de_model = gensim.models.KeyedVectors.load_word2vec_format("data/langmodels/wiki.de.align.vec", limit=10000)

    print('> Map stars into corresponding sentiment\n')
    data['sentiment'] = data.stars.replace({4:1, 5:1, 1:0, 2:0})

    print('> Prepare german dataset for tokenization\n')
    german_data = data.loc[(data['language']=='de') & (data['stars']!= 3),['review_body', 'language','sentiment', 'product_category']]
    ge_df = german_data.loc[(german_data['product_category']=='home'),['review_body', 'language','sentiment', 'product_category']]

    print('> Transforming the data into tensors')
    pipe = Pipe(ge_df,'german','de',de_model,128)
    torch_data = pipe.emb()
    print(f"Torch data shape: {torch_data.shape}")

if __name__ == '__main__':
    main()
