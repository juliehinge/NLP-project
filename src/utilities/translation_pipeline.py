import numpy as np
import torch

# Tokenization
import regex as re

# NLTK
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Translation
import translators as ts


class EmbeddingsPipeline:

    """
    Takes in reviews and returns these in w2vec form.
    """

    def __init__(self, w2vec_model, translate=False, from_language='de'):
        
        # Which w2vec model should be used
        self.w2vec_model = w2vec_model
        
        # Should the text be translated before tokenization
        self.translate = translate
        self.from_language = from_language # and from which language if translate=True

    def translate_reviews(self, reviews):
        """
        Returns list of translated reviews.
        """

        reviews_translated = []
        for review in reviews:

            # translate
            review_translated = ts.google(
                review,
                from_language=self.from_language,
                to_language='en'
            )

            # add to the final result list
            reviews_translated.append(review_translated)
        
        return reviews_translated
    
    def tokenize_reviews(self, reviews):
        """
        Returns "a list of lists (nested list)"
        where each token is a string in the inner list.
        """

        if self.translate:
            reviews = self.translate_reviews(reviews)
        
        # Final nested list to be returned
        token_list = []

        # Regular Expression to seperate tokens
        # Our tokens:
        # words and hashtags |   a symbol suffixing a word | emojis | @user
        #r_html = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)" # broken :(
        r_words_and_hashtags = r"[#a-zA-Z0-9_\-\—\'\;\:\&\\]+"
        r_emojies = r"[\U00010000-\U0010ffff]"
        r_user = r"\@user"
        r_punctuation_types = r"[\!\#\$\%\(\)\*\+\,\-\.\/\<\=\>\?\@\[\]\^\_\`\{\|\}\~]{1}"

        # Final regex
        regex = r_words_and_hashtags + "|" + r_emojies  + "|" + r_user + "|" + r_punctuation_types 
        
        # Dictionary with charecters and their better replacements
        replace_dict = {
            "`" : "'",
            "’" : "'",
            "´" : "'",
            r"\n" : " "
        }

        for i, review in enumerate(reviews):
        
            # Unescape special HTML charecters
            # review = unescape(review)

            # Loop through and replace problematic charecters with better solutions
            for char_to_replc in replace_dict.keys():
                review = review.replace(char_to_replc, replace_dict[char_to_replc])    

            # Lowercase to reduce the number of unique tokens
            review = review.lower()
            tokens = re.findall(regex, review)

            # Stopword removal
            stopwords_set = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stopwords_set]

            # Save the tokens
            token_list.append(tokens)
        
        return token_list

    def transform_reviews_to_emb(self, reviews, save_embeddings_path=None):

        # Returns tokenized reviews  
        reviews = self.tokenize_reviews(reviews)      
        
        # The final result will be 3d array:
        # First dimension = # of reviews
        # Second dimension = # of tokens in each review - 128 max, if not 128 then end is padded
        # Third dimension = size of embeddings - for all 300 
        result = np.empty((len(reviews), 128, 300))
        
        for i, review in enumerate(reviews):
            
            review_embeddings = np.zeros((128, 300))
            for j, token in enumerate(review[:128]): # Making sure that maximum # of tokens is 128
                
                if token in self.w2vec_model:
                    review_embeddings[j] = self.w2vec_model[token]
                else:
                    review_embeddings[j] = np.zeros(300)
            
            result[i] = review_embeddings
        
        # Arrays to tensors
        result = torch.Tensor(result)

        # Save embeddings for later use
        if save_embeddings_path is not None:
            torch.save(result, save_embeddings_path)

        return result


def prepare_data(df, language, frac_samples):
    """
    Returns relevant reviews with corresponding sentiment.
    """

    # Preprocessing of data
    df['sentiment'] = df.stars.replace({4: 1, 5: 1, 1: 0, 2: 0})
    la_data = df.loc[
        (df['language'] == language) & (df['stars'] != 3), ['review_body', 'language', 'sentiment', 'product_category']]
    la_df = la_data.loc[
        (la_data['product_category'] == 'home'), ['review_body', 'language', 'sentiment', 'product_category']]

    # Shuffles data and selects given proportion of data
    la_df = la_df.sample(frac = frac_samples, random_state=42)

    # Get reviews as a 1d list
    reviews = la_df['review_body'].tolist()

    # Get target as a 1D tensor
    target = la_df['sentiment'].to_numpy()
    target = torch.from_numpy(target).float().reshape(-1, 1)

    return reviews, target
