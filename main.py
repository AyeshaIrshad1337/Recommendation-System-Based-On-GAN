from utils.data_loader import load_data, split_data, interaction_matrix
from models.bpr import BPR
from models.apr import APR
from models.amr import BPRModel, train_amr_step
from models.collagan import CollaGAN
from models.acae import ACAE
from utils.evaluation import precision_at_k, recall_at_k, hit_ratio_at_k, ndcg_at_k, map_at_k, f1_score_at_k
import numpy as np
import tensorflow as tf

def main():
    df = load_data('./data/ratings.csv')
    train, test = split_data(df)
    interactions_matrix = interaction_matrix(train)

    n_users, n_items = interactions_matrix.shape
    
    # Example: Using APR model
    # model = APR(n_users, n_items, n_factors=10, n_epochs=50)
    # model.fit(interactions_matrix.values)
    
    # # Evaluate APR model
    # evaluate_model(model, test, "APR")

    # Example: Using AMR model
    bpr_model = BPRModel(n_users, n_items, embedding_dim=10)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Train AMR model
    train_amr(bpr_model, optimizer, interactions_matrix.values, epochs=10, batch_size=256)
    
    # Evaluate AMR model
    evaluate_model(bpr_model, test, "AMR")
    
    # Example: Using CollaGAN model
    collagan = CollaGAN(n_users, n_items, embedding_dim=10)
    collagan.train(interactions_matrix.values, epochs=50, batch_size=128)
    
    # Evaluate CollaGAN model
    evaluate_model(collagan, test, "CollaGAN")
    
    # Example: Using ACAE model
    acae = ACAE(n_users, n_items, embedding_dim=10)
    acae.train(interactions_matrix.values, epochs=50, batch_size=128)
    
    # Evaluate ACAE model
    evaluate_model(acae, test, "ACAE")

def evaluate_model(model, test, model_name):
    test_interactions = interaction_matrix(test)
    print(f'[{model_name}] Precision@10: {precision_at_k(model, test_interactions.values, 10)}')
    print(f'[{model_name}] Recall@10: {recall_at_k(model, test_interactions.values, 10)}')
    print(f'[{model_name}] Hit Ratio@10: {hit_ratio_at_k(model, test_interactions.values, 10)}')
    print(f'[{model_name}] NDCG@10: {ndcg_at_k(model, test_interactions.values, 10)}')
    print(f'[{model_name}] MAP@10: {map_at_k(model, test_interactions.values, 10)}')
    print(f'[{model_name}] F1 Score@10: {f1_score_at_k(model, test_interactions.values, 10)}')

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
    
    model.save_weights("amr_model.h5")

if __name__ == "__main__":
    main()
