import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split


class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_items, embedding_size=16, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.user_embedding_layer = layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            name='user_embedding_layer',
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(
            input_dim=num_users,
            output_dim=1,
            name="user_bias")
        self.item_embedding_layer = layers.Embedding(
            input_dim=num_items,
            output_dim=embedding_size,
            name='item_embedding_layer',
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.item_bias = layers.Embedding(
            input_dim=num_items,
            output_dim=1,
            name="item_bias")

    def call(self, inputs):
        user_vector = self.user_embedding_layer(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        item_vector = self.item_embedding_layer(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])
        dot_user_item = tf.tensordot(user_vector, item_vector, 2)
        x = dot_user_item + user_bias + item_bias
        return tf.nn.relu(x)


class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance


class NNRecommender(Singleton):
    def __init__(self):
        if not hasattr(self, "nn_model"):
            self.nn_model = None
            self.num_users = None
            self.num_items = None
            self.idx_enc_ratings_df = None
            self.user_idx2id = None
            self.user_id2idx = None
            self.course_idx2id = None
            self.course_id2idx = None

    def process_ratings(self, ratings_df):
        idx_enc_ratings_df = ratings_df.copy()
        user_list = idx_enc_ratings_df["user"].unique().tolist()
        user_idx2id = {i: x for i, x in enumerate(user_list)}
        user_id2idx = {x: i for i, x in enumerate(user_list)}
        course_list = idx_enc_ratings_df["item"].unique().tolist()
        course_idx2id = {i: x for i, x in enumerate(course_list)}
        course_id2idx = {x: i for i, x in enumerate(course_list)}
        idx_enc_ratings_df["user"] = idx_enc_ratings_df["user"].map(user_id2idx)
        idx_enc_ratings_df["item"] = idx_enc_ratings_df["item"].map(course_id2idx)
        idx_enc_ratings_df["rating"] = idx_enc_ratings_df["rating"].values.astype("int")
        return idx_enc_ratings_df, user_idx2id, user_id2idx, course_idx2id, course_id2idx

    def generate_train_test_datasets(self, idx_enc_ratings_df, scale=True):
        min_rating = min(idx_enc_ratings_df["rating"])
        max_rating = max(idx_enc_ratings_df["rating"])
        idx_enc_ratings_df = idx_enc_ratings_df.sample(frac=1, random_state=123)
        x = idx_enc_ratings_df[["user", "item"]].values
        if scale:
            y = idx_enc_ratings_df["rating"].apply(
                lambda rating: (rating - min_rating) / (max_rating - min_rating)).values
        else:
            y = idx_enc_ratings_df["rating"].values
        x_train, x_remaining, y_train, y_remaining = train_test_split(x, y, test_size=0.2, random_state=123)
        x_val, x_test, y_val, y_test = train_test_split(x_remaining, y_remaining, test_size=0.5, random_state=123)
        return x_train, x_val, x_test, y_train, y_val, y_test

    def train_nn(self, ratings_df, embedding_size=16, lr=0.001, batch_size=64, epochs=10):
        num_users = len(ratings_df['user'].unique())
        num_items = len(ratings_df['item'].unique())
        model = RecommenderNet(num_users, num_items, embedding_size)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        self.idx_enc_ratings_df, self.user_idx2id, self.user_id2idx, self.course_idx2id, self.course_id2idx = self.process_ratings(
            ratings_df)
        x_train, x_val, x_test, y_train, y_val, y_test = self.generate_train_test_datasets(self.idx_enc_ratings_df,
                                                                                           scale=True)
        model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            verbose=1
        )

        self.num_users = num_users
        self.num_items = num_items
        self.nn_model = model

    def predict_nn(self, user_id, top_courses=10, similarity_threshold=0.0):
        unique_course_ids = self.idx_enc_ratings_df['item'].unique()
        user_enrolled_courses = self.idx_enc_ratings_df[self.idx_enc_ratings_df['user'] == self.user_id2idx[user_id]][
            'item'].unique()
        candidate_course_ids = np.setdiff1d(unique_course_ids, user_enrolled_courses)
        user_idx = self.user_id2idx[user_id]
        user_courses = np.array([[user_idx, course_id] for course_id in candidate_course_ids])
        predicted_ratings = self.nn_model.predict(user_courses).flatten()
        above_threshold_indices = np.where(predicted_ratings >= similarity_threshold)[0]
        candidate_course_ids = candidate_course_ids[above_threshold_indices]
        predicted_ratings = predicted_ratings[above_threshold_indices]
        top_course_indices = np.argsort(predicted_ratings)[-top_courses:][::-1]
        top_courses = [candidate_course_ids[idx] for idx in top_course_indices]
        top_scores = [predicted_ratings[idx] for idx in top_course_indices]
        top_course_ids = [self.course_idx2id[idx] for idx in top_courses]
        recommended_courses = {top_course_ids[i]: top_scores[i] for i in range(len(top_course_ids))}
        return recommended_courses

    def extract_embeddings(self, user_emb_filepath='../data/user_embeddings.csv', item_emb_filepath='../data/item_embeddings.csv'):
        # Extract user embeddings
        user_embeddings = self.nn_model.user_embedding_layer.get_weights()[0]
        user_embeddings_df = pd.DataFrame(user_embeddings)
        user_embeddings_df.columns = [f'UFeature{i}' for i in range(user_embeddings_df.shape[1])]
        user_embeddings_df.insert(0, 'user', pd.Series([self.user_idx2id[i] for i in range(len(user_embeddings_df))]))
        user_embeddings_df.to_csv(user_emb_filepath, index=False)

        # Extract item embeddings
        item_embeddings = self.nn_model.item_embedding_layer.get_weights()[0]
        item_embeddings_df = pd.DataFrame(item_embeddings)
        item_embeddings_df.columns = [f'CFeature{i}' for i in range(item_embeddings_df.shape[1])]
        item_embeddings_df.insert(0, 'item', pd.Series([self.course_idx2id[i] for i in range(len(item_embeddings_df))]))
        item_embeddings_df.to_csv(item_emb_filepath, index=False)

        return user_embeddings_df, item_embeddings_df


# # Testing code
# recommender = NNRecommender()
# ratings_filepath = pd.read_csv('../data/ratings.csv')
# recommender.train_nn(ratings_filepath)
# recommended_courses = recommender.predict_nn(user_id=300.0)
# print(recommended_courses)
# recommender.extract_embeddings()
