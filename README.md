# NLP-project, group 1
This research paper examines the problem of labelling reviews with either a positive or negative sentiment, in a cross-lingual setting using a BiLSTM model. Two methods were implemented to compare different approaches. A method using Google Translate to translate a target language to a source language, compared to using aligned word embeddings using FastText. As expected, our project shows that the method using Google Translate achieves higher evaluation scores in sentiment classification tasks than the alignment method. Our results can be reproduced by cloning our repository: https://github.itu.dk/juhi/nlp-project.  

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


