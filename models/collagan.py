import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

class CollaGAN:
    def __init__(self, num_users, num_items, embedding_dim):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_generator(self):
        user_input = layers.Input(shape=(self.num_users,))
        x = layers.Dense(128, activation='relu')(user_input)
        x = layers.Dense(self.embedding_dim, activation='relu')(x)
        item_output = layers.Dense(self.num_items, activation='sigmoid')(x)

        return Model(user_input, item_output)

    def build_discriminator(self):
        item_input = layers.Input(shape=(self.num_items,))
        x = layers.Dense(128, activation='relu')(item_input)
        x = layers.Dense(64, activation='relu')(x)
        validity = layers.Dense(1, activation='sigmoid')(x)

        return Model(item_input, validity)

    def build_gan(self):
        self.discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
        self.discriminator.trainable = False

        user_input = layers.Input(shape=(self.num_users,))
        generated_items = self.generator(user_input)
        validity = self.discriminator(generated_items)

        gan = Model(user_input, validity)
        gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

        return gan

    def train(self, interaction_matrix, epochs, batch_size=128):
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, interaction_matrix.shape[0], batch_size)
            user_batch = np.eye(self.num_users)[idx]  # Create one-hot encoding of users
            real_items = interaction_matrix[idx]

            # Generate fake items
            fake_items = self.generator.predict(user_batch)
 
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_items, real)
            d_loss_fake = self.discriminator.train_on_batch(fake_items, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator
            g_loss = self.gan.train_on_batch(user_batch, real)

            print(f"{epoch + 1}/{epochs} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%] [G loss: {g_loss}]")

    def predict(self, user_vector):
        return self.generator.predict(user_vector)
