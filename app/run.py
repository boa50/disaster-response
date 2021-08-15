import json
import re
import plotly
import pandas as pd
import pickle

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = pickle.load(open('./models/best_model.pkl', 'rb'))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals

    # Comparison between requests and offers
    reqests_offers_sums = df[['request', 'offer']].sum()

    reqests_offers_labels = reqests_offers_sums.keys()
    reqests_offers_values = reqests_offers_sums.values

    reqests_offers_chart = {
        'data': [
            Pie(
                labels=reqests_offers_labels,
                values=reqests_offers_values
            )
        ],
        'layout': {
            'title': 'Quantity of Requests x Offers'
        }
    }

    # Most requested things
    df_requests = df[df['request'] == 1]

    dropped_columns = ['id', 'message', 'original', 'genre', 'related', 'request', 'offer', 'direct_report']
    for column in df.columns:
        if re.search('related', column) is not None:
            dropped_columns.append(column)

    bars_qtd = 7
    idx = bars_qtd - 1

    requests_sums = df_requests.drop(columns=dropped_columns).sum()
    requests_sums_sorted = requests_sums.sort_values(ascending=False)

    requests_labels = requests_sums_sorted[:idx].keys().to_list()
    requests_values = list(requests_sums_sorted[:idx].values)

    requests_labels.append('ohters')
    requests_values.append(requests_sums_sorted[idx:].sum())

    requests_chart = {
        'data': [
            Bar(
                x=requests_labels,
                y=requests_values
            )
        ],
        'layout': {
            'title': 'Distribution of Request Messages',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Type of Request"
            }
        }
    }

    # Most offered things
    df_offers = df[df['offer'] == 1]

    bars_qtd = 7
    idx = bars_qtd - 1

    offers_sums = df_offers.drop(columns=dropped_columns).sum()
    offers_sums_sorted = offers_sums.sort_values(ascending=False)

    offers_labels = offers_sums_sorted[:idx].keys().to_list()
    offers_values = list(offers_sums_sorted[:idx].values)

    offers_labels.append('ohters')
    offers_values.append(offers_sums_sorted[idx:].sum())

    offers_chart = {
        'data': [
            Bar(
                x=offers_labels,
                y=offers_values
            )
        ],
        'layout': {
            'title': 'Distribution of Offer Messages',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Type of Offer"
            }
        }
    }
    
    # create visuals
    graphs = [reqests_offers_chart,requests_chart,offers_chart]
    
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
