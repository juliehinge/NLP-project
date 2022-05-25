# NLP-project, group 1

This experiment is using a bidirectional long short-term memory network (biLSTM), to predict the sentiment of Amazon product reviews. Our approach tackles the problem by adapting the resources available in one language to other languages and can be seen as a solution to problems that require the transfer of sentiment classification knowledge between two or more languages. The transfer learning approach is applied by creating embeddings using aligned vectors from FastText and corpora in the chosen languages, English and German. These languages will then be passed  to our BiLSTM network that will predict if the review has a positive or negative sentiment. By using this approach we will examine the novel part of the research, which is cross-lingual sentiment classification with English as the source language and German as the target language. The main objectives are presented, including current resources in sentiment classification for the target language and exploration of the chosen methods, consisting of an alignment method and a method using Google Translate. 

## Clone the repository
You should clone our repository in order to reproduce our result.

## Virtual environment 

Creating virtual environment using conda
```
conda create --name group1project python=3.8
```

Activating the environment
```
conda activate group1project
```

Installing the requirements
```
pip install -r requirements.txt
```

## Run the code
The following commands will run all our code, but first it downloads all the necessary data, which takes a while. 
```
cd src
python main.py
```


