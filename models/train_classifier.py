import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn.decomposition import TruncatedSVD

import pickle


def load_data(database_filepath):
    """Loads from database and return  X and Y """
         
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('Disaster_Messages', engine)
    
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns.tolist()
    
    return X, Y ,category_names
    


def tokenize(text):
    
    """ Tokenizer that normalize case , remove punctuation and stop words then using stemming and lemmatization for  word normalization """
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    
    tokens = word_tokenize(text)

  
    tokens = [w for w in tokens if w not in stopwords.words("english")]

    stemmed = [PorterStemmer().stem(w) for w in tokens]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in stemmed]

    return tokens



def build_model():
    
    """ Build a machine learning pipeline that returns the GridSearchCV model  """
    
    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('other',TruncatedSVD()),
                    ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
                    ])

   
   
    parameters = { 
        
               'clf__estimator__min_samples_split': [2,3,5]
        
                 }
    cv = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1)

    return cv
    
   

def evaluate_model(model, X_test, Y_test, category_names):
    
    """Print the testing  results (accuracy, precision and recall) of the model """
    
    Y_pred = model.predict(X_test)
    
    accuracy = (Y_pred == Y_test).mean()
    print( 'Accuracy :',accuracy)
    for n, col in enumerate(category_names):
        print('Category: {}\n'.format(col))
        print(classification_report(Y_test[col], Y_pred[:, n]))
    


def save_model(model, model_filepath):
    """save the model to disk as a pickle file """
    
    filename =  model_filepath
    pickle_out =open(filename, 'wb')
    pickle.dump(model, pickle_out)
    pickle_out.close()

    
    
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()