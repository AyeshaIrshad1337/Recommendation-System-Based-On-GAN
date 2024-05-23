import pandas as pd
from lightfm import Dataset

def load_dataset(movie_path, rating_path):
    """Load dataset from csv files and return a pandas dataframe.
        input: movie_path, rating_path
        return: movie, rating"""
    movie = pd.read_csv(movie_path)
    rating = pd.read_csv(rating_path)
    return movie, rating
def interaction_matrix(ratings):
    """Create interaction matrix from pandas dataframe.
        input: ratings
        return: interaction matrix"""
    dataset= Dataset()
    dataset.fit(ratings['userId'], ratings['movieId'])
    interaction, weights = dataset.build_interactions([(x[0], x[1]) for x in ratings[['userId', 'movieId']].values])
    return dataset, interaction , weights 
if __name__ == '__main__':
    movie_path = 'data/movies.csv'
    rating_path = 'data/ratings.csv'
    movie, rating = load_dataset(movie_path, rating_path)
    dataset, interaction, weights = interaction_matrix(rating)
    print(dataset)
    print(interaction)
    print(weights)