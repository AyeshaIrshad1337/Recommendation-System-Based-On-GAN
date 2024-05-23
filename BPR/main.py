from utils.data_loader import load_data, split_data, interaction_matrix
from model.bpr import BPR
from utils.evaluation import precision_at_k, recall_at_k, hit_ratio_at_k, ndcg_at_k, map_at_k, f1_score_at_k


def main():
    df = load_data('./data/ratings.csv')
    train, test = split_data(df)
    interactions_matrix = interaction_matrix(train)

    n_users, n_items = interactions_matrix.shape
    model = BPR(n_users, n_items, n_factors=10, n_epochs=50)
    model.fit(interactions_matrix.values)
    
    test_interactions = interaction_matrix(test)
    print(f'Precision@10: {precision_at_k(model, test_interactions.values, 10)}')
    print(f'Recall@10: {recall_at_k(model, test_interactions.values, 10)}')
    print(f'Hit Ratio@10: {hit_ratio_at_k(model, test_interactions.values, 10)}')
    print(f'NDCG@10: {ndcg_at_k(model, test_interactions.values, 10)}')
    print(f'MAP@10: {map_at_k(model, test_interactions.values, 10)}')
    print(f'F1 Score@10: {f1_score_at_k(model, test_interactions.values, 10)}')
if __name__ == "__main__":
    main()
