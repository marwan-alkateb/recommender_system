import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Ignore warnings
warnings.filterwarnings('ignore')

# Constants
RANDOM_STATE = 123
N_CLUSTERS_RANGE = range(2, 10)
MODEL_NAME = 'clustering.pkl'


# Function to train clustering models
def train(user_profile_df, algorithm_choice='kmeans', num_clusters=N_CLUSTERS_RANGE):
    # Step 1: Normalize the user profile matrix before clustering or PCA
    user_profile_df.drop(columns=['user'], errors='ignore', inplace=True)
    feature_names = list(user_profile_df.columns)
    user_profile_df[feature_names] = StandardScaler().fit_transform(user_profile_df[feature_names])
    std_user_profile = user_profile_df[feature_names].values

    best_sil_score, best_k, best_model = -np.inf, None, None
    sil_scores = []

    # Step 2: Select clustering model and tune hyperparameters
    if algorithm_choice == 'kmeans':
        # Ensure num_clusters is an iterable
        if not isinstance(num_clusters, (list, range)):
            num_clusters = range(2, num_clusters + 1)

        for n_clusters in tqdm(num_clusters, position=0, leave=True):
            model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE).fit(std_user_profile)
            sil_score = silhouette_score(std_user_profile, model.labels_)
            sil_scores.append(sil_score)
            if sil_score > best_sil_score:
                best_sil_score = sil_score
                best_k = n_clusters
                best_model = model

    elif algorithm_choice == 'dbscan':
        # Hyperparameter tuning for DBSCAN
        eps_values = np.linspace(0.1, 2.0, num=20)  # Adjust the range as needed
        for eps in eps_values:
            for min_samples in range(2, 10):  # You might need to tune this range
                model = DBSCAN(eps=eps, min_samples=min_samples).fit(std_user_profile)
                if len(set(model.labels_)) > 1:  # DBSCAN can sometimes assign all points to noise
                    sil_score = silhouette_score(std_user_profile, model.labels_)
                    sil_scores.append(sil_score)
                    if sil_score > best_sil_score:
                        best_sil_score = sil_score
                        best_k = None  # DBSCAN doesn't have a fixed number of clusters
                        best_model = model

    else:
        raise ValueError("Invalid algorithm choice. Use 'kmeans' or 'dbscan'.")

    # Save the best model
    joblib.dump(best_model, MODEL_NAME)

    return best_k, best_sil_score, sil_scores


# Function to plot silhouette scores
def plot_scores(scores):
    plt.figure(figsize=(10, 6))
    plt.plot(N_CLUSTERS_RANGE, scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.grid(True)
    plt.show()


# Function to predict clustering and recommend courses
def predict_clustering(user_profile_df, ratings_df, user_id, top_courses, similarity_threshold):
    column_to_drop = 'user'

    if column_to_drop not in user_profile_df.columns:
        raise ValueError(f"Column {column_to_drop} does not exist in the DataFrame")

    user_ids = user_profile_df[column_to_drop].values  # Extract user_ids before dropping the column

    if user_id not in user_ids:
        raise ValueError(f"USER {user_id} does not exist in the user_profile_df")

    user_profile_df.drop(columns=[column_to_drop], errors='ignore', inplace=True)
    feature_names = list(user_profile_df.columns)

    model = joblib.load(MODEL_NAME)

    if isinstance(model, KMeans):
        clusters = model.predict(user_profile_df[feature_names])
    elif isinstance(model, DBSCAN):
        clusters = model.fit_predict(user_profile_df[feature_names])
    else:
        raise ValueError("Loaded model is neither KMeans nor DBSCAN.")

    user_cluster_df = pd.DataFrame({'user': user_ids, 'cluster': clusters})

    # Get the cluster number for the given user
    user_cluster_num = user_cluster_df[user_cluster_df['user'] == user_id]['cluster'].values[0]

    # Get the user IDs in the same cluster
    users_in_cluster = user_cluster_df[user_cluster_df['cluster'] == user_cluster_num]['user'].values

    # Filter ratings to only those from users in the same cluster
    cluster_ratings = ratings_df[ratings_df['user'].isin(users_in_cluster)]

    # Calculate most-visited courses
    most_visited_courses = cluster_ratings['item'].value_counts().head(top_courses)

    # Normalize the scores to be between 0 and 1
    max_count = most_visited_courses.max()
    most_visited_courses_normalized = most_visited_courses / max_count

    # Filter courses based on similarity threshold
    filtered_courses = most_visited_courses_normalized[most_visited_courses_normalized >= similarity_threshold]

    # Convert the Series to a dictionary
    most_visited_courses_dict = filtered_courses.to_dict()

    return most_visited_courses_dict


# TODO: ___________________________________________________ TESTING ___________________________________________________

# Load Data
# genres_df = pd.read_csv('../data/genres.csv')
# ratings_filepath = pd.read_csv('../data/ratings.csv')
# user_profile_df = pd.read_csv('../data/generated_user_profile_vectors.csv')

'''
ratings_df has the following shape:
user,item,rating
1889878,CC0101EN,3.0
# One user can rate multiple items. 
'''
''' 
description_df has following shape:
,COURSE_ID,TITLE,DESCRIPTION,TEXT
0,ML0201EN,robots are coming  build iot apps with watson swift and node red, have fun with ...
'''

# generate_user_profiles(genres_df, ratings_filepath, user_profile_filepath='../data/old_generated_user_profile_vectors.csv')

# best_k, best_score, scores = train(user_profile_df)
# plot_scores(scores)
#
# Get users and their clusters
# recommendations = predict_clustering(user_profile_df, ratings_filepath, user_id=2, top_courses=5, similarity_threshold=0.5)
# print(recommendations)
