import pandas as pd
from sklearn.model_selection import train_test_split
def load_data(path):
    rating = pd.read_csv(path)
    return rating
def split_data(data, ratio=0.2):
    data_train, data_test = train_test_split(data, test_size=ratio, random_state=42)
    return data_train, data_test
def interaction_matrix(df):
    interaction_matrix=df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    return interaction_matrix
if __name__=="__main__":
   
    data = load_data(path="./data/ratings.csv")
    train , test = split_data(data)
    interaction_matrix = interaction_matrix(train)
    print(interaction_matrix)