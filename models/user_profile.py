import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, pairwise_distances
from sklearn.preprocessing import StandardScaler


def clean_csv_file(filepath):
    """
    Cleans the CSV file by removing rows with NaN values.

    Args:
        filepath (str): The path to the CSV file to be cleaned.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df = pd.read_csv(filepath)
    initial_shape = df.shape
    df = df.dropna()
    df.to_csv(filepath, index=False)
    print(f"Cleaned {filepath}. Removed {initial_shape[0] - df.shape[0]} rows containing NaN values.")
    return df


def generate_user_profiles(genres_filepath='../data/genres.csv', ratings_filepath='../data/ratings.csv',
                           user_profile_filepath='../data/generated_user_profile_vectors.csv'):
    """
    Generates user profiles by multiplying ratings with course profiles and summing them up.

    Args:
        genres_filepath (str): Path to the genres CSV file.
        ratings_filepath (str): Path to the ratings CSV file.
        user_profile_filepath (str): Path to save the generated user profile vectors CSV file.

    Returns:
        pd.DataFrame: User profiles with the desired format.
    """
    genres_df = pd.read_csv(genres_filepath)
    ratings_df = pd.read_csv(ratings_filepath)

    # Check for NaN values in genres_df
    if genres_df.isnull().values.any():
        warnings.warn("genres_df contains NaN values.")
        clean_csv_file(genres_filepath)
    if 'COURSE_ID' not in genres_df.columns or 'TITLE' not in genres_df.columns:
        raise ValueError("genres_df must contain 'COURSE_ID' and 'TITLE' columns.")

    # Check for NaN values in ratings_df
    if ratings_df.isnull().values.any():
        warnings.warn("ratings_df contains NaN values after cleaning.")
        clean_csv_file(ratings_filepath)
    if 'user' not in ratings_df.columns or 'item' not in ratings_df.columns or 'rating' not in ratings_df.columns:
        raise ValueError("ratings_df must contain 'user', 'item', and 'rating' columns.")

    course_ids = genres_df['COURSE_ID'].values
    course_profiles = genres_df.drop(columns=['COURSE_ID', 'TITLE']).values
    user_ids = np.sort(ratings_df['user'].unique())

    user_id2idx_dict = {user_id: idx for idx, user_id in enumerate(user_ids)}
    course_id2idx_dict = {course_id: idx for idx, course_id in enumerate(course_ids)}

    # Initialize the sparse matrix
    user_ids_replaced_with_idx = ratings_df['user'].map(user_id2idx_dict)  # rows
    course_ids_replaced_with_idx = ratings_df['item'].map(course_id2idx_dict)  # columns
    ratings_values = ratings_df['rating'].values  # data

    if user_ids_replaced_with_idx.isnull().values.any():
        raise ValueError("user_ids_replaced_with_idx contains NaN values.")
    if course_ids_replaced_with_idx.isnull().values.any():
        raise ValueError("course_ids_replaced_with_idx contains NaN values.")

    ratings_matrix = csr_matrix((ratings_values, (user_ids_replaced_with_idx, course_ids_replaced_with_idx)),
                                shape=(len(user_ids), len(course_ids)))

    # Perform matrix multiplication
    user_profiles_matrix = ratings_matrix.dot(course_profiles)

    # Convert the resulting matrix to a DataFrame
    user_profile_df = pd.DataFrame(user_profiles_matrix, columns=genres_df.columns[2:])
    user_profile_df.insert(0, 'user', user_ids)

    # Check for NaN values in the final user_profile_df
    if user_profile_df.isnull().values.any():
        raise ValueError("user_profile_df contains NaN values.")

    user_profile_df.to_csv(user_profile_filepath, index=False)

    return user_profile_df


def apply_pca(user_profile_df, variance_threshold=0.9, pca_filepath='../data/generated_pca_user_profile_vectors.csv'):
    """
    Applies PCA to reduce the dimensionality of user profiles.

    Args:
        user_profile_df (pd.DataFrame): DataFrame containing user profiles.
        variance_threshold (float): The amount of variance that needs to be explained by the selected components.
        pca_filepath (str): Path to save the PCA-transformed user profile vectors CSV file.

    Returns:
        pd.DataFrame: The PCA-transformed user profiles.
    """
    # Step 1: Standardize the data to prevent issues when performing clustering or PCA
    features = user_profile_df.drop(columns='user')

    standardized_features = StandardScaler().fit_transform(features)

    # Step 2: Find the optimized n_components for PCA
    pca = PCA()
    pca.fit(standardized_features)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    # Step 3: Apply PCA on the user profile feature vectors with the optimized n_components value
    transformed_features = PCA(n_components=n_components).fit_transform(standardized_features)
    user_ids = user_profile_df['user'].reset_index(drop=True)
    transformed_features_df = pd.DataFrame(transformed_features,
                                           columns=[f'PC{i}' for i in range(transformed_features.shape[1])])
    pca_transformed_user_profile_df = pd.concat([user_ids, transformed_features_df], axis=1)

    pca_transformed_user_profile_df.to_csv(pca_filepath, index=False)

    return pca_transformed_user_profile_df


def recommend_courses(user_id, user_profile_filepath='../data/generated_user_profile_vectors.csv',
                      ratings_filepath='../data/ratings.csv', distance_metric='cosine', top_courses=10,
                      similarity_threshold=0.3):
    """
    Recommends courses based on user profiles.

    Args:
        user_id (int): The ID of the user to whom we want to recommend courses.
        user_profile_filepath (str): Path to the user profile vectors CSV file.
        ratings_filepath (str): Path to the ratings CSV file.
        distance_metric (str): The distance metric to use ('cosine', 'euclidean', 'jaccard', or 'manhattan').
        top_courses (int): The number of courses to recommend.
        similarity_threshold (float): The minimum similarity score for a user to be considered.

    Returns:
        dict: A dictionary where the keys are the course IDs and the values are the similarity scores.
    """
    ratings_df = pd.read_csv(ratings_filepath)
    user_profile_df = pd.read_csv(user_profile_filepath)

    if user_id not in user_profile_df['user'].values:
        raise ValueError(f'{user_id} does not exist in the given user_profiles!')

    if user_id not in ratings_df['user'].values:
        raise ValueError(f'{user_id} does not exist in the given ratings_filepath!')

    if not 0 <= similarity_threshold <= 1:
        raise ValueError(f'Threshold Value: {similarity_threshold} is not in range of (0,1)!')

    user_profile = user_profile_df[user_profile_df['user'] == user_id].drop(columns='user').values.reshape(1, -1)
    other_profiles = user_profile_df[user_profile_df['user'] != user_id].drop(columns='user').values

    if distance_metric == 'cosine':
        distances = cosine_similarity(user_profile, other_profiles)
        similarity_scores = distances[0]
    elif distance_metric == 'euclidean':
        distances = euclidean_distances(user_profile, other_profiles)
        max_distance = np.max(distances)
        similarity_scores = 1 - (distances[0] / max_distance)
    elif distance_metric == 'jaccard':
        user_profile_binary = (user_profile > 0).astype(int)
        other_profiles_binary = (other_profiles > 0).astype(int)
        jaccard_distances = pdist(np.vstack([user_profile_binary, other_profiles_binary]), metric='jaccard')
        similarity_scores = 1 - squareform(jaccard_distances)[0, 1:]
    elif distance_metric == 'manhattan':
        distances = pairwise_distances(user_profile, other_profiles, metric='manhattan')
        max_distance = np.max(distances)
        similarity_scores = 1 - (distances[0] / max_distance)
    else:
        raise ValueError("Invalid distance metric. Choose either 'cosine', 'euclidean', 'jaccard', or 'manhattan'.")

    # Get the indices of the users with the nearest profiles
    nearest_users = user_profile_df[user_profile_df['user'] != user_id]['user'][
        similarity_scores >= similarity_threshold]

    # Get the courses that these users rated the highest
    top_courses_series = ratings_df[ratings_df['user'].isin(nearest_users)]['item'].value_counts()

    # Create a dictionary where the keys are the course IDs and the values are the similarity scores
    top_courses_dict = {course_id: round(similarity_scores[i], 2) for i, course_id in
                        enumerate(top_courses_series.index[:top_courses]) if
                        similarity_scores[i] >= similarity_threshold}

    # Sort the dictionary by values in descending order
    top_courses_dict = dict(sorted(top_courses_dict.items(), key=lambda item: item[1], reverse=True))

    return top_courses_dict

# TODO: ___________________________________________________ TESTING ___________________________________________________

# user_profile_df = generate_user_profiles()
#
# user_id = 2103428
# if user_id != user_profile_df['user'].iloc[-1]:
#     print("User ID does not exist in the DataFrame.")
# top_courses = recommend_courses(user_id=user_id,  distance_metric='cosine', top_courses=10,
#                                 similarity_threshold=0.2)
# print(f"Top courses are: {top_courses}")
#
# apply_pca(user_profile_df)
