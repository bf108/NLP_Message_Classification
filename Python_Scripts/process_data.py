import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages and categories dataset
    messages = pd.read_csv('messages_filepath')
    categories = pd.read_csv('categories_filepath')

    # merge datasets
    return messages.merge(categories,on='id')

def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';',expand=True)

    # Make related column binary
    categories[0] = categories[0].str.replace('2','1')

    #Rename columns corectly
    categories.columns = categories.iloc[0].apply(lambda x: x.split('-')[0]).values

    # Convert column contents to binary 1/0 integer
    for col in categories:
    # set each value to be the last character of the string
        categories[col] = categories[col].apply(lambda x: x[-1]).astype(int)

    # Drop child alone because it is all zeros which provides no guidance fo
    categories.drop(columns='child_alone',inplace=True)

    # drop the original categories column from `df`
    df.drop(columns=['categories'],inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(database_filename.split('.db')[0],
              con = engine,
              index= False,
              if_exists='replace')

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
