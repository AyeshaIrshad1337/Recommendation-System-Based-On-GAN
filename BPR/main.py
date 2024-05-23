from utils.data_loader import load_data, split_data, interaction_matrix
from model.bpr import BPR
from utils.evaluation import evaluation

def main():
    df = load_data('./data/ratings.csv')
    train, test = split_data(df)
    interactions_matrix = interaction_matrix(train)

    n_users, n_items = interactions_matrix.shape
    model = BPR(n_users, n_items, n_factors=10, n_epochs=10)
    model.fit(interactions_matrix.values)
    
    test_interactions = interaction_matrix(test)
    print(f'Precision@10: {evaluation(model, test_interactions.values, 10)}')

if __name__ == "__main__":
    main()
