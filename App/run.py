import json
import plotly
import pandas as pd
import pickle

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import plotly.graph_objs as go
# from sklearn.externals import joblib
from sqlalchemy import create_engine
import sqlite3

app = Flask(__name__)

db_filepath = '/Users/benfarrell/Documents/Udacity/DataScience/6_Disaster_Response/Data/DisasterResponse.db'
table = db_filepath.split('/')[-1].split('.db')[0]
conn = sqlite3.connect(db_filepath)
df = pd.read_sql(f'SELECT * FROM {table}',conn)

print(df.head())

# engine = create_engine('sqlite:///DisasterResponse.db')
# df = pd.read_sql_table('DisasterResponse', engine)

# load model
with open("../6_Disaster_Response/AdaBoost.pkl",'rb') as file:
    model = pickle.load(file)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    results = df.iloc[:,4:].sum().sort_values(ascending=True)
    results_names = [item.replace('_',' ').title() for item in list(results.index)]
    results_values = list(results.values)

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
        },

        {
            'data': [
                Bar(
                    x=results_values,
                    y=results_names,
                    orientation = 'h',
                )
            ],

            'layout': {
                'title': 'Frequency of Disaster in Dataset',
                'yaxis': {
                    'title': ""
                },
                'xaxis': {
                    'title': "Count"
                },
                'height': 1000,
                'margin' : dict(
                    l=150,
                    pad=4
                )
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    #return render_template('master.thlm', data_set = genre_names)
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
