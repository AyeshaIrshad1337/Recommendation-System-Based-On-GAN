import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from utils.evaluation import eval_at_k, map, f1_score_at_k
class ACAE:
    def __init__(self, num_users, num_items, embedding_dim):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.autoencoder, self.encoder = self.build_autoencoder()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_autoencoder(self):
        input_items = layers.Input(shape=(self.num_items,))
        x = layers.Dense(128, activation='relu')(input_items)
        encoded = layers.Dense(self.embedding_dim, activation='relu')(x)

        x = layers.Dense(128, activation='relu')(encoded)
        decoded = layers.Dense(self.num_items, activation='sigmoid')(x)

        autoencoder = Model(input_items, decoded)
        encoder = Model(input_items, encoded)

        return autoencoder, encoder

    def build_discriminator(self):
        encoded_input = layers.Input(shape=(self.embedding_dim,))
        x = layers.Dense(64, activation='relu')(encoded_input)
        x = layers.Dense(32, activation='relu')(x)
        validity = layers.Dense(1, activation='sigmoid')(x)

        return Model(encoded_input, validity)

    def build_gan(self):
        self.discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
        self.discriminator.trainable = False

        input_items = layers.Input(shape=(self.num_items,))
        encoded_repr = self.encoder(input_items)
        validity = self.discriminator(encoded_repr)

        gan = Model(input_items, validity)
        gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

        return gan

    def train(self, interaction_matrix, epochs, batch_size=128):
        real = tf.ones((batch_size, 1))
        fake = tf.zeros((batch_size, 1))
        precisions, recalls, ndcgs, hits, f1_scores, map_scores = [], [], [], [], [], []
        for epoch in range(epochs):
            idx = np.random.randint(0, interaction_matrix.shape[0], batch_size)
            item_batch = interaction_matrix[idx]

            # Generate encoded items
            encoded_items = self.encoder.predict(item_batch)

            # Generate fake encoded items
            fake_encoded_items = np.random.normal(size=(batch_size, self.embedding_dim))

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(encoded_items, real)
            d_loss_fake = self.discriminator.train_on_batch(fake_encoded_items, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator (autoencoder)
            g_loss = self.gan.train_on_batch(item_batch, real)

            print(f"{epoch + 1}/{epochs} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%] [G loss: {g_loss}]")
            precision, recall, ndcg, hit = eval_at_k(self, interaction_matrix, k=10)
            map_scores.append(map(self, interaction_matrix, k=10))
            f1_scores.append(f1_score_at_k(self, interaction_matrix, k=10))
            precisions.append(precision)
            recalls.append(recall)
            ndcgs.append(ndcg)
            hits.append(hit)
        return precisions, recalls, ndcgs, hits, f1_scores, map_scores
    def predict(self, vector):
        # Input should be an interaction vector of items
        return self.autoencoder.predict(vector)
    def recommend(self, user, user_items, N=10, filter_already_liked_items=True):
        user_vector = np.zeros((1, self.num_items))
        user_vector[0, user_items[user].nonzero()[1]] = 1
        scores = self.predict(user_vector).flatten()
        
        if filter_already_liked_items:
            liked_items = user_items[user].nonzero()[1]
            scores[liked_items] = -np.inf
        
        top_items = np.argsort(-scores)[:N]
        return [(item, scores[item]) for item in top_items]