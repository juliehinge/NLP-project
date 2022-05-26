
import pandas as pd
import numpy as np

import random
random.seed(42)
import re
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('data/raw/raw_train.csv')

df['sentiment'] = df.stars.replace({4: 1, 5: 1, 1: 0, 2: 0})
la_data = df.loc[
    (df['language'] == 'en') & (df['stars'] != 3), ['review_body', 'language', 'sentiment', 'product_category']]
df = la_data.loc[
    (la_data['product_category'] == 'home'), ['review_body', 'language', 'sentiment', 'product_category']]

df = df.sample(frac = 0.1)

def tokenize(text):
    return np.array(re.split('\s', str(text)), dtype='object')



def getFeatures(data, stopwordSplit, freqSplit):
    features = {}
    for instance in df.review_body:
        for word in tokenize(instance):
            if word not in features:
                features[word] = 0
            features[word] += 1
    return sorted(sorted(features, key=features.get, reverse=True)[stopwordSplit:freqSplit])

features = np.array(getFeatures(df.review_body, 100, 800))

def first(X):
    return np.array([np.isin(features,tokenize(x)).astype(int) for x in X])


X_train = first(df.review_body)


le = preprocessing.LabelEncoder() 
y_train = le.fit_transform(df.sentiment)


le = preprocessing.LabelEncoder() 
y_train = le.fit_transform(df.sentiment)

print(y_train)



clf = LogisticRegression(random_state=0).fit(X_train, y_train)
print(f"train score {clf.score(X_train, y_train)}\n")




df = pd.read_csv('data/raw/raw_test.csv')

df['sentiment'] = df.stars.replace({4: 1, 5: 1, 1: 0, 2: 0})
la_data = df.loc[
    (df['language'] == 'de') & (df['stars'] != 3), ['review_body', 'language', 'sentiment', 'product_category']]
df = la_data.loc[
    (la_data['product_category'] == 'home'), ['review_body', 'language', 'sentiment', 'product_category']]

df_test = df.sample(frac = 0.05)




X_test = first(df_test.review_body)
y_test = le.transform(df_test.sentiment)

print(f"Test score {clf.score(X_test, y_test)}\n")


