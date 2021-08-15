import re
import pandas as pd

from plotly.graph_objs import Bar, Pie

def request_offers_comparison(df):
    """Create a pie chart comparing the number of requests and the number
    of offers.
    
    Arguments:
    - df (Dataframe): The dataframe to extract data from

    Returns:
    - dict: A dictionary with the data for the pie chart and the layout text
    """

    reqests_offers_sums = df[['request', 'offer']].sum()

    reqests_offers_labels = reqests_offers_sums.keys()
    reqests_offers_values = reqests_offers_sums.values

    return {
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


def get_dropped_columns(df):
    """Get the columns to be dropped and used by the bar charts
    
    Arguments:
    - df (Dataframe): The dataframe to extract data from

    Returns:
    - dropped_columns: An array containing the columns to be dropped
    """

    dropped_columns = ['id', 'message', 'original', 'genre', 'related', 'request', 'offer', 'direct_report']
    for column in df.columns:
        if re.search('related', column) is not None:
            dropped_columns.append(column)

    return dropped_columns


def bar_chart(df, title='', yaxis='', xaxis=''):
    """Create a simple bar chart based on the arguments.
    
    Arguments:
    - df (Dataframe): The dataframe to extract data from

    Returns:
    - dict: A dictionary with the data for the bar chart and the layout text
    """

    dropped_columns = get_dropped_columns(df)
    bars_qtd = 7
    idx = bars_qtd - 1

    sums = df.drop(columns=dropped_columns).sum()
    sums_sorted = sums.sort_values(ascending=False)

    labels = sums_sorted[:idx].keys().to_list()
    values = list(sums_sorted[:idx].values)

    labels.append('ohters')
    values.append(sums_sorted[idx:].sum())

    return {
        'data': [
            Bar(
                x=labels,
                y=values
            )
        ],
        'layout': {
            'title': title,
            'yaxis': {
                'title': yaxis
            },
            'xaxis': {
                'title': xaxis
            }
        }
    }


def requested_things(df):
    """Create a bar chart showing the most requested things.
    
    Arguments:
    - df (Dataframe): The dataframe to extract data from

    Returns:
    - dict: A dictionary with the data for the bar chart and the layout text
    """

    df_requests = df[df['request'] == 1]

    return bar_chart(df_requests, 'Distribution of Request Messages',
                    'Count', 'Type of Request')


def offered_things(df):
    """Create a bar chart showing the most offered things.
    
    Arguments:
    - df (Dataframe): The dataframe to extract data from

    Returns:
    - dict: A dictionary with the data for the bar chart and the layout text
    """

    df_offers = df[df['offer'] == 1]

    return bar_chart(df_offers, 'Distribution of Offer Messages',
                    'Count', 'Type of Offer')