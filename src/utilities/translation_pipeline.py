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

class Pipe:
    def __init__(self,foreign_lan,stop_lang,lang, lan_model, seq_len):
        self.foreign_lan = foreign_lan
        self.stop_lang = stop_lang
        self.lang = lang
        self.lan_model = lan_model
        self.seq_len = seq_len

    def stop_words(self):
        stop = set(stopwords.words(self.stop_lang))
        lang_stop = self.foreign_lan['review_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        return lang_stop

    def google_trans(self):
        lang_stop = self.stop_words()
        
        language = []
        # TODO: undo 10 reviews only
        for sentence in lang_stop[:100]:
            new_word = ts.google(sentence,from_language=self.lang,to_language='en')
            language.append(new_word)
        return language

    def to_array(self):
        language = self.google_trans()
        array = np.array(language)
        return array 

    def tokens(self):
        array = self.to_array()
        '''Input: A 1D numpy array of strings. One element is one "tweet"
        Output: Splitted words, one output line per input line, with spaces between tokens.
        Returns "a list of lists" where each token is a string in the list'''
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

        for i, review in enumerate(array):
        
            # Unescape special HTML charecters
            # review = unescape(review)

            # Loop through and replace problematic charecters with better solutions
            for char_to_replc in replace_dict.keys():
                review = review.replace(char_to_replc, replace_dict[char_to_replc])    

            # Lowercase to reduce the number of unique tokens
            review = review.lower()
            tokens = re.findall(regex, review)

            # Store string in np array
            token_list.append(tokens)
        
        return token_list

    def emb(self):
        
        reviews = self.tokens() # returns tokenized reviews        
        
        # The final result will be 3d array:
        # First dimension = # of reviews
        # Second dimension = # of tokens in each review - 128 max, if not 128 then start is padded
        # Third dimension = size of embeddings - for all 300 
        result = np.empty((len(reviews), 128, 300))
        
        for i, review in enumerate(reviews):
            
            emb_en = np.empty((len(review), 300))
            for j, token in enumerate(review[:128]): # Making sure that maximum # of tokens is 128
                
                # TODO: Eng model
                if token in self.lan_model:
                    emb_en[j] = self.lan_model[token]
                else:
                    emb_en[j] = np.zeros(300)
            
            # Padding if neccessary
            padding_size = max(128 - len(review), 0)
            
            # Add it to the final result
            if padding_size > 0:
                padding = np.zeros((padding_size, 300))
                stacked = np.vstack((padding, emb_en))
                result[i] = stacked
            else:
                result[i] = emb_en

        return torch.Tensor(result)