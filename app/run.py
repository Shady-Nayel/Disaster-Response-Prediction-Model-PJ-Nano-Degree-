import json
import plotly
import pandas as pd
from flask import Flask,render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
import sys
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download(['stopwords','punkt', 'wordnet', 'averaged_perceptron_tagger'])
import os
import warnings
# Suppress warning messages
warnings.filterwarnings('ignore')


app = Flask(__name__)

model_filepath = os.path.join(os.getcwd(), "..", "models", "model.pkl")
database_filepath=os.path.join(os.getcwd(), "..", "data", "DisasterResponse.db")

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

# load data
engine = create_engine(f'sqlite:///{database_filepath}')
df = pd.read_sql_table('df', engine)

df=df.drop("child_alone",axis=1)
df['related']=df['related'].map(lambda x: 1 if x == 2 else x)

# load model
model = joblib.load(model_filepath)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([tokenize(query)])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()