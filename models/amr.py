import tensorflow as tf
from tensorflow.keras import layers

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
        user_embedding = self.user_embedding(user)
        item_embeddings = self.item_embedding(items)
        scores = tf.reduce_sum(user_embedding * item_embeddings, axis=1)
        return scores

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
