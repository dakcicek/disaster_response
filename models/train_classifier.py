#!/usr/bin/env python
# # ML Pipeline Preparation

import nltk
nltk.download('punkt')
nltk.download('wordnet')
import re
import sys
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize, sent_tokenize
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import learning_curve,GridSearchCV,RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import make_scorer
from time import time
from nltk.stem import SnowballStemmer
from string import punctuation
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    This function loads data from given database path 
    and returns a dataframe
    Input:
        database_filepath: database file path
    Output:
        X: traing message list
        Y: training target
        category names  
    """
    # load data from database
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('DisasterResponse',engine)
    
    # define features and target
    X = df.message
    y = df.iloc[:,4:]
    category_names = list(df.columns[4:])
    
    return X, y

def tokenize(text):
    """
    Tokenization function to process the text data to normalize, lemmatize, and tokenize text. 
    Input: Text data
    Output: List of clean tokens 
    """
     # remove punctations
    text =  ''.join([c for c in text if c not in punctuation])
    
    #tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_pipeline():
    print('Building pipeline..')

    improved_pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, stop_words='english')),
        ('tfidf', TfidfTransformer(norm='l2')),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42, n_estimators=60,max_depth=4,n_jobs=4,verbose=3 )))
    ])
    return improved_pipeline

        
def evaluate_model(model, X_test, Y_test):
    """
    Prints the classification report for the given model and test data
    Input:
        model: trained model
        X_test: test data for the predication 
        Y_test: true test labels for the X_test data
    Output:
        None 
    """
    #Accuracy over all

    y_pred = model.predict(X_test)
    print('Overall Accuracy: {}'.format( accuracy_score(Y_test.iloc[:,1:].values, np.array([x[1:] for x in y_pred]))))

    # Calculate the accuracy for each of them.
    for i in range(len(Y_test.columns)):
        print("Category Name:", Y_test.columns[i],"\n", classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(Y_test.columns[i], accuracy_score(Y_test.iloc[:, i].values, y_pred[:,i])))



def save_model(model, model_file_name):
    pickle.dump(model, open(model_file_name, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print('Building model...')
        model = build_pipeline()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
