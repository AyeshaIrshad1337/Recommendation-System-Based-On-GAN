import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from utils.evaluation import eval_at_k, map, f1_score_at_k
class BPRModel(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim):
        super(BPRModel, self).__init__()
        self.user_embedding = layers.Embedding(num_users, embedding_dim)
        self.item_embedding = layers.Embedding(num_items, embedding_dim)
    
    def call(self, inputs):
        user_inputs, pos_item_inputs, neg_item_inputs = inputs
        user_embedding = self.user_embedding(user_inputs)
        pos_item_embedding = self.item_embedding(pos_item_inputs)
        neg_item_embedding = self.item_embedding(neg_item_inputs)
        
        pos_score = tf.reduce_sum(user_embedding * pos_item_embedding, axis=1)
        neg_score = tf.reduce_sum(user_embedding * neg_item_embedding, axis=1)
        
        return pos_score, neg_score

    def predict(self, user, items):
        user_embedding = self.user_embedding(tf.convert_to_tensor([user], dtype=tf.int32))
        item_embeddings = self.item_embedding(tf.convert_to_tensor(items, dtype=tf.int32))
        scores = tf.reduce_sum(user_embedding * item_embeddings, axis=1)
        return scores.numpy()
    
    def recommend(self, user, user_items, N=10, filter_already_liked_items=True):
        user_embedding = self.user_embedding(tf.convert_to_tensor([user], dtype=tf.int32))
        all_item_embeddings = self.item_embedding.embeddings
        
        scores = tf.reduce_sum(user_embedding * all_item_embeddings, axis=1).numpy().flatten()
        
        if filter_already_liked_items:
            liked_items = user_items[user].nonzero()[1]
            scores[liked_items] = -np.inf
        
        top_items = np.argsort(-scores)[:N]
        return [(item, scores[item]) for item in top_items]
def train_amr(model, optimizer, interactions, epochs, batch_size):
    num_training_samples = len(interactions.nonzero()[0])
    user_inputs = np.random.randint(model.user_embedding.input_dim, size=num_training_samples)
    pos_item_inputs = np.random.randint(model.item_embedding.input_dim, size=num_training_samples)
    neg_item_inputs = np.random.randint(model.item_embedding.input_dim, size=num_training_samples)
    precisions, recalls, ndcgs, hits, f1_scores, map_scores = [], [], [], [], [], [] 
    for epoch in range(epochs):
        for start in range(0, num_training_samples, batch_size):
            end = start + batch_size
            user_batch = user_inputs[start:end]
            pos_item_batch = pos_item_inputs[start:end]
            neg_item_batch = np.random.randint(model.item_embedding.input_dim, size=len(user_batch))
                
            loss = train_amr_step(model, optimizer, user_batch, pos_item_batch, neg_item_batch)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.numpy()}")
        precision, recall, ndcg, hit = eval_at_k(model, interactions, k=10)
        map_scores.append(map(model, interactions, k=10))
        f1_scores.append(f1_score_at_k(model, interactions, k=10))
        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
        hits.append(hit)
    return precisions, recalls, ndcgs, hits, f1_scores, map_scores
def train_amr_step(model, optimizer, user_inputs, pos_item_inputs, neg_item_inputs):
    with tf.GradientTape() as tape:
        pos_score, neg_score = model((user_inputs, pos_item_inputs, neg_item_inputs))
        loss = -tf.reduce_mean(tf.math.log(tf.nn.sigmoid(pos_score - neg_score)))
    gradients = tape.gradient(loss, model.trainable_variables)
    
    for var in model.trainable_variables:
        if "item_embedding" in var.name:
            adv_noise = 0.1 * tf.sign(gradients[model.trainable_variables.index(var)])
            var.assign_add(adv_noise)
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
