import pandas as pd
from bpr import bpr_update

movie = pd.read_csv('D:\Recommender System Based On GANs\data\movies.csv')
user = pd.read_csv('D:\Recommender System Based On GANs\data\\ratings.csv')
bpr_update(movie, user)
