import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import warnings
# Suppress warning messages
warnings.filterwarnings('ignore')

messages_filepath="disaster_messages.csv"
categories_filepath="disaster_categories.csv"
database_name="DisasterResponse.db"

def load_clean_data(messages_filepath, categories_filepath):
    message_df=pd.read_csv(messages_filepath)
    categories_df=pd.read_csv(categories_filepath,delimiter=",")
    categories_df=pd.concat([categories_df["id"],categories_df["categories"].str.split(";",expand=True)],axis=1)
    columns=["id"]
    for i in range(len(categories_df.columns[1:])):
        columns.append(categories_df[i].str.split("-")[0][0])
    categories_df.columns=columns

    for i in categories_df.columns:
        if i=="id":
            pass
        else:
            categories_df[i]=categories_df[i].str.split("-",expand=True)[1]

    df=pd.merge(message_df,categories_df,on="id",how="inner")
    df=df.drop_duplicates()
    
    for i in df.columns[4:]:
        df[i]=df[i].astype(int)

    return df

def save_data(dataframe, database_filepath):
    """Saves the given dataframe in a given database"""
    
    engine=create_engine('sqlite:///'+database_filepath)
    dataframe.to_sql('df', engine, index=False,if_exists="replace")  
    return engine

def main():
        try:
             print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
             df = load_clean_data(messages_filepath, categories_filepath)
             print('Data Has Been Loaded and Cleaned...')
             print('Saving data...\n    DATABASE: {}'.format(database_name))
             save_data(df, database_name)
             print('Cleaned data saved to database!')
        except:
            print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'YourDataBase.db')

    

if __name__ == '__main__':
    main()