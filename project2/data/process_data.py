# import libraries 
import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine

def load_data(credits_filepath, movies_filepath):
    """
    Loads and merges datasets from 2 filepaths.
    
    Parameters:
    Movies_filepath: Movies csv file
    credits_filepath: credits csv file
    
    Returns:
    df: dataframe containing Movies_filepath and credits_filepath merged
    
    """
    # load datasets
    credits = pd.read_csv(credits_filepath)
    movies = pd.read_csv(movies_filepath)
    # merge datasets on common id and assign to df
    df = credits.merge(movies, how ='outer', on =['title'])
    df.dropna(inplace=True)
    movies_plotly = df[['movie_id','budget','popularity','runtime','vote_average','vote_count','revenue']]
    return movies_plotly

def clean_data(df):
    """
    Cleans the dataframe.
    
    Parameters:
    df: DataFrame
    
    Returns:
    df: Cleaned DataFrame
    
    """
    df.dropna(inplace=True)
    movies_plotly = df[['movie_id','budget','popularity','runtime','vote_average','vote_count','revenue']]
    movies_plotly=movies_plotly.astype(float)

    movies_plotly.isnull().sum()
    movies_plotly.dropna(inplace=True)
  


    return df
    

   
    
def save_data(df, database_filepath):
    """Stores df in a SQLite database."""
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('moviesplotly', engine, index=False, if_exists='replace')  



def main():
    """Loads data, cleans data, saves data to database"""
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    Movies: {}\n    Credits: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
        print(df.isnull().sum())

    
    else:
        print('Please provide the filepaths of the movies and credits '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'tmdb_credits.csv tmdb_movies.csv '\
              'InsertDatabaseNamemovies_plotly.db')


if __name__ == '__main__':
    main()
