import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """Load the data from a database file and fill the X and y values for the model
    
    Arguments:
    - database_filepath (str): The path to the database to be read

    Returns:
    - X: The x values (inputs) for use in the model
    - y: The y values (labels) for use in the model
    - category_names: The names of the categories from the labels
    """

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_response', engine)

    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    """Tokenize the text data for the machine learning model
    
    Arguments:
    - text (str): The text to be tokenized

    Returns:
    - tokens: The tokens generated based on the text
    """

    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')

    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if not
              (token in stop_words or token in string.punctuation)]

    return tokens


def build_model():
    """Builds the model to be used to make predictions from the text

    Returns:
    - model: The model to be used for the machine learning
    """
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize, 
                                max_features=10000, 
                                min_df=1, 
                                ngram_range=(1, 1))),
        ('clf', MultiOutputClassifier(ComplementNB(alpha=0.5, norm=True), n_jobs=-1))
    ])
    
    return model


def column_metrics(y_true, y_pred, column):
    """Return metrics for one column
    
    Arguments:
    - y_true (Dataframe): A Dataframe containing the true values of the labels
    - y_pred (Dataframe): A Dataframe containing the predicted values of the labels
    - column (str): The name of the colum

    Returns:
    - report: The metrics at a format to be printed
    - report_dict: The metrics at a format to used to calculate the model metrics
    """

    report = classification_report(y_true[column], y_pred[column], zero_division=0)
    report_dict = classification_report(y_true[column], y_pred[column], zero_division=0, output_dict=True)
    
    return report, report_dict


def model_metrics(column_metrics):
    """Return metrics for the Machine Learning model
    
    Arguments:
    - column_metrics (arr): An array containg metrics from each output column

    Returns:
    - accuracy: The accuracy metric from the model
    - precision: The precision metric from the model
    - recall: The recall metric from the model
    - f1_score: The f1_score metric from the model
    """

    accuracy = 0
    precision = 0
    recall = 0
    f1_score = 0
    for metrics in column_metrics:
        accuracy += metrics['accuracy']
        precision += metrics['macro avg']['precision']
        recall += metrics['macro avg']['recall']
        f1_score += metrics['macro avg']['f1-score']

    metrics_size = len(column_metrics)

    accuracy /= metrics_size
    precision /= metrics_size
    recall /= metrics_size
    f1_score /= metrics_size
    
    return accuracy, precision, recall, f1_score


def evaluate_model(model, X_test, Y_test, category_names):
    """Print metrics from each category and the entire model
    
    Arguments:
    - model (ML model): The model to be evaluated
    - X_test (Dataframe): The input values to be used to make predictions
    - Y_test (Dataframe): The true values of the labels to be compared with the predictions
    - category_names (arr): The array containing the categories to be analyzed
    """

    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=Y_test.columns)

    metrics = []
    print('    The column metrics are:')
    for column in category_names:
        report, report_dict = column_metrics(Y_test, y_pred_df, column)
        metrics.append(report_dict)
        
        print("    " + column)
        print("    " + report)
    
    print()

    accuracy, precision, recall, f1_score = model_metrics(metrics)
    print('    The model metrics are:')
    print('    Accuracy: {:.2f}%'.format(accuracy))
    print('    Precision: {:.2f}%'.format(precision))
    print('    Recall: {:.2f}%'.format(recall))
    print('    F1 score: {:.2f}%'.format(f1_score))
    print()


def save_model(model, model_filepath):
    """Save the machine learning model in a pickle file
    
    Arguments:
    - model (ML model): The model to be saved
    - model_filepath (str): The path to save the model into
    """

    pickle.dump(model, open(model_filepath, 'wb'))


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
