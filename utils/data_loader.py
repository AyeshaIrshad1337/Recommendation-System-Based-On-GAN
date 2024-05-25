import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    rating = pd.read_csv(path)
    return rating

def split_data(data, ratio=0.2):
    data_train, data_test = train_test_split(data, test_size=ratio, random_state=42)
    return data_train, data_test

def interaction_matrix(df):
    interaction_matrix = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    return interaction_matrix
