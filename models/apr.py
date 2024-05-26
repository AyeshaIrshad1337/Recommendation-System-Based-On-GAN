import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class APRModel(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim, epsilon=0.1):
        super(APRModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
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
    
    def compute_loss(self, pos_score, neg_score):
        loss = -tf.reduce_mean(tf.math.log(tf.nn.sigmoid(pos_score - neg_score)))
        return loss

    def compute_adv_loss(self, user_inputs, pos_item_inputs, neg_item_inputs):
        with tf.GradientTape() as tape:
            pos_score, neg_score = self.call((user_inputs, pos_item_inputs, neg_item_inputs))
            loss = self.compute_loss(pos_score, neg_score)
        gradients = tape.gradient(loss, self.trainable_variables)
        
        adv_noise = [self.epsilon * tf.sign(grad) for grad in gradients]
        adv_loss = -tf.reduce_mean(tf.math.log(tf.nn.sigmoid(pos_score - neg_score)))
        
        return adv_loss, adv_noise
    
    def train_step(self, user_inputs, pos_item_inputs, neg_item_inputs, optimizer):
        with tf.GradientTape() as tape:
            pos_score, neg_score = self.call((user_inputs, pos_item_inputs, neg_item_inputs))
            loss = self.compute_loss(pos_score, neg_score)
        gradients = tape.gradient(loss, self.trainable_variables)
        
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        adv_loss, adv_noise = self.compute_adv_loss(user_inputs, pos_item_inputs, neg_item_inputs)
        for i in range(len(self.trainable_variables)):
            self.trainable_variables[i].assign_add(adv_noise[i])
        
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss, adv_loss

    def recommend(self, user, user_items, N=10, filter_already_liked_items=True):
        user_vector = np.eye(self.num_users)[user:user+1]
        user_embedding = self.user_embedding(tf.convert_to_tensor([user], dtype=tf.int32))
        all_item_embeddings = self.item_embedding.embeddings
        
        scores = tf.reduce_sum(user_embedding * all_item_embeddings, axis=1).numpy().flatten()
        
        if filter_already_liked_items:
            liked_items = user_items[user].nonzero()[1]
            scores[liked_items] = -np.inf
        
        top_items = np.argsort(-scores)[:N]
        return [(item, scores[item]) for item in top_items]
