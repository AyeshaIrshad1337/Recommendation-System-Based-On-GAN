import numpy as np
import tensorflow as tf

class APR:
    def __init__(self, n_users, n_items, n_factors, lr=0.01, reg=0.01, n_epochs=50, epsilon=0.5):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.epsilon = epsilon
        self.user_factors = np.random.normal(scale=1./self.n_factors, size=(n_users, n_factors))
        self.item_factors = np.random.normal(scale=1./self.n_factors, size=(n_items, n_factors))
    
    def fit(self, interactions):
        for epoch in range(self.n_epochs):
            for user, pos_item, neg_item in self._sample(interactions):
                self._update_factors(user, pos_item, neg_item)
                self._adversarial_training(user, pos_item, neg_item)
            print(f'Epoch {epoch + 1}/{self.n_epochs}')
    
    def _sample(self, interactions):
        user_item_pairs = np.argwhere(interactions > 0)
        for user, pos_item in user_item_pairs:
            neg_item = np.random.choice(np.where(interactions[user] == 0)[0])
            yield user, pos_item, neg_item
    
    def _update_factors(self, user, pos_item, neg_item):
        user_factor = self.user_factors[user]
        pos_item_factor = self.item_factors[pos_item]
        neg_item_factor = self.item_factors[neg_item]
        
        positive_prediction = np.dot(user_factor, pos_item_factor)
        negative_prediction = np.dot(user_factor, neg_item_factor)
        
        loss = 1 / (1 + np.exp(positive_prediction - negative_prediction))
        
        self.user_factors[user] += self.lr * (loss * (pos_item_factor - neg_item_factor) - self.reg * user_factor)
        self.item_factors[pos_item] += self.lr * (loss * user_factor - self.reg * pos_item_factor)
        self.item_factors[neg_item] -= self.lr * (loss * user_factor - self.reg * neg_item_factor)

    def _adversarial_training(self, user, pos_item, neg_item):
        user_factor = self.user_factors[user]
        pos_item_factor = self.item_factors[pos_item]
        neg_item_factor = self.item_factors[neg_item]
        
        delta_u = self.epsilon * tf.sign(tf.random.normal(user_factor.shape))
        delta_i = self.epsilon * tf.sign(tf.random.normal(pos_item_factor.shape))
        
        adv_user_factor = user_factor + delta_u
        adv_pos_item_factor = pos_item_factor + delta_i
        
        positive_prediction = np.dot(adv_user_factor, adv_pos_item_factor)
        negative_prediction = np.dot(adv_user_factor, neg_item_factor)
        
        loss = 1 / (1 + np.exp(positive_prediction - negative_prediction))
        
        self.user_factors[user] += self.lr * (loss * (adv_pos_item_factor - neg_item_factor) - self.reg * adv_user_factor)
        self.item_factors[pos_item] += self.lr * (loss * adv_user_factor - self.reg * adv_pos_item_factor)
        self.item_factors[neg_item] -= self.lr * (loss * adv_user_factor - self.reg * neg_item_factor)

    def predict(self, user, item):
        return np.dot(self.user_factors[user], self.item_factors[item])
