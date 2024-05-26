from utils.data_loader import load_data, split_data, interaction_matrix
from models.bpr import BPR
from models.amr import BPRModel, train_amr_step
from models.collagan import CollaGAN
from models.acae import ACAE
from models.apr import APRModel  # Import the APR model
import numpy as np
import tensorflow as tf
import implicit
from scipy.sparse import csr_matrix
from utils.evaluation import precision_at_k, recall_at_k, hit_ratio_at_k, ndcg_at_k, map_at_k, f1_score_at_k

def convert_to_implicit_format(interactions):
    return csr_matrix(interactions)

def dcg(relevance_scores):
    return np.sum([(2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores)])

def map(model, test_interaction, k=10):
    aps = []
    for user in range(test_interaction.shape[0]):
        user_interaction = test_interaction[user].nonzero()[1]
        if len(user_interaction) == 0:
            continue
        scores = model.recommend(user, test_interaction, N=k, filter_already_liked_items=False)
        top_k_items = [x[0] for x in scores]
        hits = 0
        precisions = []
        for i, item in enumerate(top_k_items):
            if item in user_interaction:
                hits += 1
                precisions.append(hits / (i + 1))
        if precisions:
            aps.append(np.mean(precisions))
        else:
            aps.append(0)
    return np.mean(aps)

def eval_at_k(model, test_interactions, k=10):
    recalls = []
    precisions = []
    ndcgs = []
    for user in range(test_interactions.shape[0]):
        user_interactions = test_interactions[user].nonzero()[1]
        if len(user_interactions) == 0:
            continue
        scores = model.recommend(user, test_interactions, N=k, filter_already_liked_items=False)
        top_k_items = [x[0] for x in scores]
        
        precisions.append(len(set(top_k_items) & set(user_interactions)) / k)
        recalls.append(len(set(top_k_items) & set(user_interactions)) / len(user_interactions))
        relevance_scores = [1 if item in user_interactions else 0 for item in top_k_items]
        ideal_relevance_scores = sorted(relevance_scores, reverse=True)
        if dcg(ideal_relevance_scores) == 0:
            ndcgs.append(0)
        else:
            ndcgs.append(dcg(relevance_scores) / dcg(ideal_relevance_scores))
    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)

def evaluate_implicit_model(model, train_interactions, test_interactions, k=10):
    precisions, recalls, ndcgs = eval_at_k(model, test_interactions, k)
    maps = map(model, test_interactions, k)
    
    print(f'Precision@{k}: {precisions}')
    print(f'Recall@{k}: {recalls}')
    print(f'NDCG@{k}: {ndcgs}')
    print(f'MAP@{k}: {maps}')

def train_amr(model, optimizer, interactions, epochs, batch_size):
    num_training_samples = len(interactions.nonzero()[0])
    user_inputs = np.random.randint(model.user_embedding.input_dim, size=num_training_samples)
    pos_item_inputs = np.random.randint(model.item_embedding.input_dim, size=num_training_samples)
    neg_item_inputs = np.random.randint(model.item_embedding.input_dim, size=num_training_samples)

    for epoch in range(epochs):
        for start in range(0, num_training_samples, batch_size):
            end = start + batch_size
            user_batch = user_inputs[start:end]
            pos_item_batch = pos_item_inputs[start:end]
            neg_item_batch = np.random.randint(model.item_embedding.input_dim, size=len(user_batch))
            
            loss = train_amr_step(model, optimizer, user_batch, pos_item_batch, neg_item_batch)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.numpy()}")
def train_apr(model, optimizer, interactions, epochs, batch_size):
    num_training_samples = len(interactions.nonzero()[0])
    user_inputs = np.random.randint(model.num_users, size=num_training_samples)
    pos_item_inputs = np.random.randint(model.num_items, size=num_training_samples)
    neg_item_inputs = np.random.randint(model.num_items, size=num_training_samples)

    for epoch in range(epochs):
        for start in range(0, num_training_samples, batch_size):
            end = start + batch_size
            user_batch = user_inputs[start:end]
            pos_item_batch = pos_item_inputs[start:end]
            neg_item_batch = np.random.randint(model.num_items, size=len(user_batch))
            
            loss, adv_loss = model.train_step(user_batch, pos_item_batch, neg_item_batch, optimizer)
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy()}, Adv Loss: {adv_loss.numpy()}")

def main():
    df = load_data('./data/ratings.csv')
    train, test = split_data(df)
    interactions_matrix = interaction_matrix(train)

    n_users, n_items = interactions_matrix.shape

    # Convert interaction matrix to implicit format
    train_interactions = convert_to_implicit_format(interactions_matrix.values)
    test_interactions = convert_to_implicit_format(interaction_matrix(test).values)

    # Example: Using BPR model from implicit library
    model = implicit.bpr.BayesianPersonalizedRanking(factors=10, iterations=50)
    model.fit(train_interactions)

    # Evaluate BPR model
    evaluate_implicit_model(model, train_interactions, test_interactions)

    # Example: Using AMR model
    bpr_model = BPRModel(n_users, n_items, embedding_dim=10)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Train AMR model
    train_amr(bpr_model, optimizer, interactions_matrix.values, epochs=10, batch_size=256)
    
    # Evaluate AMR model
    evaluate_implicit_model(bpr_model, train_interactions, test_interactions)
    
    # Example: Using CollaGAN model
    collagan = CollaGAN(n_users, n_items, embedding_dim=10)
    collagan.train(interactions_matrix.values, epochs=50, batch_size=128)
    
    # Evaluate CollaGAN model
    evaluate_implicit_model(collagan, train_interactions, test_interactions)
    
    # Example: Using ACAE model
    acae = ACAE(n_users, n_items, embedding_dim=10)
    acae.train(interactions_matrix.values, epochs=50, batch_size=128)
    
    # Evaluate ACAE model
    evaluate_implicit_model(acae, train_interactions, test_interactions)
    
    # Example: Using APR model
    apr_model = APRModel(n_users, n_items, embedding_dim=10)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Train APR model
    train_apr(apr_model, optimizer, interactions_matrix.values, epochs=10, batch_size=256)
    
    # Evaluate APR model
    evaluate_implicit_model(apr_model, train_interactions, test_interactions)


if __name__ == "__main__":
    main()
