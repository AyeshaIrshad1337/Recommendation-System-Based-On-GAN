from classes import user , movie
from numpy import random
from util import random_vector, min_rating num_users
from random import seed 
import pandas as pd
import numpy as np

def read_ratings(filename):
    seed(42)
    np.random.seed(42)
    r_cols=['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(filename,  sep=',', names=r_cols, encoding='latin-1')

    ratings["user_id"]=ratings["user_id"].astype('int')
    ratings["movie_id"]=ratings["movie_id"].astype('int')
    ratings["rating"]=ratings["rating"].astype('int')

    numusers=num_users()
    msks = ratings['user_id'] < numusers
    ratings = ratings[msks]
    users=dict()
    testcount = 0
    traincount = 0
    trainuserdict = dict()

    for index, row in ratings.iterrows():
        userid =int(row['user_id'])
        movieid = int(row['movie_id'])
        rating = int(row['rating'])
        minmovierating = min_rating()
        if rating >= minmovierating:
            if random.random() < 0.7:
                traincount += 1
                if userid not in users:
