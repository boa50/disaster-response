import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load the messages and categories data inside a Dataframe
    
    Arguments:
    - messages_filepath (str): The filepath that contains the messages data
    - categories_filepath (str): The filepath that contains the categories data

    Returns:
    - Dataframe: A dataframe with the messages and categories data
    """

    # Load the data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Join the data based on the id
    df = messages.merge(categories, on='id')

    return df


def has_only_zero_and_one(df, column):
    """Check if the column in the df has only 0 and 1 values
    
    Arguments:
    - df (Dataframe): The dataframe to be analyzed
    - column (str): The column name to be analyzed
    """
    
    other_values = df[column].unique()
    other_values = np.delete(other_values, np.where(other_values == 0))
    other_values = np.delete(other_values, np.where(other_values == 1))
    
    if other_values.size == 0:
        return True
    
    return False


def clean_data(df):
    """Preprocess and cleans the data form messages and categories Dataframe
    
    Arguments:
    - df (Dataframe): The dataframe containing all the raw data from messages 
    and categories

    Returns:
    - Dataframe: The processed and cleaned dataframe
    """

    # Split the categories into separate columns
    categories = df['categories'].str.split(';', expand=True)

    # Replace the categories column names
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x : x[:-2])
    categories.columns = category_colnames

    # Convert the categories values to numbers
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
        # categories[column] = categories[column].astype(int)

    # Replace the old categories column with the new ones
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # Remove duplicate rows
    df.drop_duplicates(ignore_index=True, inplace=True)

    # Make sure that values are 0 or 1
    for column in df.drop(['id', 'message', 'original', 'genre'], axis=1).columns:
        if not has_only_zero_and_one(df, column):
            print('    Column {} has different values than 0 and 1'.format(column))
            df[column] = df[column].apply(lambda x : 1 if x >= 1 else 0)
            
            if has_only_zero_and_one(df, column):
                print('    Column {} fixed'.format(column))
            else:
                print('    Column {} not fixed'.format(column))

    return df


def save_data(df, database_filename):
    """Save the dataframe as a table named disaster_response in a database
    
    Arguments:
    - df (Dataframe): The dataframe to be saved in the database
    - database_filename (str): The name of the database file
    """

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_response', engine, if_exists='replace', index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()