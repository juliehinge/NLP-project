print('> Loading modules\n')
from utilities.translation_pipeline import Pipe
from utilities.loaders import load_reviews, load_w2vec_model
import warnings
warnings.filterwarnings('ignore')

# Flags
LOCAL_DATA = True # Reviews and models are downloaded if True

def main():

    data = load_reviews(LOCAL_DATA)
    de_model = load_w2vec_model(
        "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.de.align.vec",
        LOCAL_DATA
    )

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
