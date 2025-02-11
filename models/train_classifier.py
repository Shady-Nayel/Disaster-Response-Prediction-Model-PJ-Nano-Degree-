import sys
import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download(['stopwords','punkt', 'wordnet', 'averaged_perceptron_tagger'])
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report , accuracy_score
from sklearn.model_selection import train_test_split
import joblib
from sqlalchemy import create_engine
import os
import warnings
# Suppress warning messages
warnings.filterwarnings('ignore')

model_filepath = os.getcwd()

database_filepath=os.path.join(os.getcwd(), "..", "data", "DisasterResponse.db")

def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('df', engine)
    df["message"]=df["message"].astype(str)
    #contains all zeroes
    df=df.drop("child_alone",axis=1)
    #
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    df = df.drop(columns=['id', 'original', 'genre'], axis=1)
    return df

def tokenize(sent):
    lemmatizer=WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))| {"us","also","im","http","dont","un"}
    if isinstance(sent, float) and np.isnan(sent):  # Check for NaN values
        sent = ""
    sent=re.sub(r'[^A-Za-z\s+]', '',sent).strip()
    sent=sent.lower()
    sent=re.sub(r'aa|(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '',sent)
    sent=sent.split()
    sent=[lemmatizer.lemmatize(str(i),"v") for i in sent if i not in stop_words]
    sent=" ".join(sent)
    return sent

def overall_accuracy(y_test,y_pred):
    """
    Custom accuracy function for multi-output classification.
    It checks if all predicted labels in a row match the true labels.
    """
    row_accuracy = (y_test == y_pred).all(axis=1)
    overall_accuracy = row_accuracy.mean()

    return overall_accuracy

def build_model():
    '''
    Build a ML pipeline using ifidf, Logistic Regression, and gridsearch
    Input: None
    Output:
        Results of GridSearchCV
    '''
    pipeline = Pipeline([
                        ('tfidf', TfidfVectorizer(max_features=100,ngram_range=(1,3))),
                        ('clf', MultiOutputClassifier(LogisticRegression()))
                        ])

    param_grid = {
        'clf__estimator__C': [0.01, 0.1, 1, 10],  # Regularization strength
        'clf__estimator__penalty': ['l1', 'l2'],  # Regularization type
        'clf__estimator__solver': ['liblinear', 'saga'],  # Solvers that support l1 & l2
    }
        
    cv = GridSearchCV(pipeline, param_grid=param_grid,cv=4,scoring="accuracy")
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model performance using test data
    Input: 
        model: Model to be evaluated
        X_test: Test data (features)
        Y_test: True lables for Test data
        category_names: Labels for 36 categories
    Output:
        Print accuracy and classfication report for each category
    '''
    Y_pred = model.predict(X_test)
    
    # Calculate the accuracy for each of them.
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))
    return Y_pred


def save_model(model, model_filepath):
    '''
    Save model as a pickle file 
    Input: 
        model: Model to be saved
        model_filepath: path of the output pick file
    Output:
        A pickle file of saved model
    '''
    joblib.dump(model, "model.pkl")

def main():
    try:
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        df= load_data(database_filepath)
        df["message"]=df["message"].apply(lambda x: tokenize(x))
        df=df.drop(df[(df["message"]=="") | (df["message"]=="nan")].index).reset_index(drop=True)
        X=df["message"]
        Y=df.iloc[:,1:]
        category_names=df.iloc[:,1:].columns
        
        print('Building model...')
        model = build_model()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        Y_pred=evaluate_model(model, X_test, Y_test, category_names)

        print("Overall Accuarcy",overall_accuracy(Y_test,Y_pred))

        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    except:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()