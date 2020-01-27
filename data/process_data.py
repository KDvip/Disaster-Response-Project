import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    """ load massages and categories data from filepaths.
    Args:
        messages_filepath : Path to messages csv file
        categories_filepath : Path to categories csv file
    Return:
        datafram (df) : merge the messages and categories datasets
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return  df 
    


def clean_data(df):
    
    """ Clean data included in the DataFrame and transform categories part:
       - Split categories into separate category columns.
       - Convert category values to just numbers 0 or 1.
       - Replace categories column in df with new category columns
       - Remove duplicates .
    Args:
         df :DataFrame
    Return:
         df :cleaned DataFrame
    """
    # Create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x:x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Iterate through the category columns in df to keep only the last character of each string (the 1 or 0)
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]  
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1,inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(subset='message', inplace=True)

    return df
    
def save_data(df, database_filename):
    
    """ Save df to Sql path.
    Args:
         df : cleaned Datafram .
         database_filename : file path to save df as  Sql data base
    Retun:
        None           
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('Disaster_Messages', engine, index=False ,if_exists='replace')
   

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