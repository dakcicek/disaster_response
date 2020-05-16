# import necessary packages
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function loads two data from given csv files and returns a merged Pandas Dataframe
    Input:
        messages_filepath: messages csv file
        categories_filepath: categories csv file
    Output:
        result_df: merged Pandas Dataframe
    """
    #load first messages csv to Pandas Dataframe
    message_df = pd.read_csv(messages_filepath)
    message_df.head()
    
    #load first messages csv to dataframe
    categories_df = pd.read_csv(categories_filepath)
    categories_df.head()
    #merge two dataframe on id column, by inner join
    result_df = message_df.merge(categories_df, how='inner',on='id')
    print("Result dataset row count: {} ".format(result_df.shape[0]))
    
    return result_df
    

def clean_data(df):
    """
    This function cleans, transform and concat the Pandas Dataframes
    Input:
        df: Pandas Dataframe that will be cleaned and transformed
    Output:
        df: cleaned Pandas Dataframe
    """
    #split the categories column to get all pieces
    only_categories = df.categories.str.split(';', expand=True)
    #get the first row
    first_row = only_categories.loc[0].values
    #split and get values until last two char
    first_row = [x[:-2] for x in first_row]
    #set these values as columns
    only_categories.columns = first_row
    #get last numeric chars from each cell
    only_categories = only_categories.applymap(lambda x : int(x[-1]))
    #drop old column
    df.drop('categories', axis=1, inplace=True)
    #concat result df and only numeric categories
    df = pd.concat([df, only_categories], axis=1)
    #check duplicates
    duplicated_count = df.duplicated().sum()
    print('There are {} duplicated message'.format(duplicated_count))
    #drop duplicates
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    """
    This function creates database and save given Pandas Dataframe to database
    Input:
        df: Pandas Dataframe that will be saved to database
        database_filename: database file name that will be created
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    engine.execute('drop table if exists DisasterResponse')
    df.to_sql('DisasterResponse', engine,index=False)


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