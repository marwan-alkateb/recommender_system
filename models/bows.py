# Bag of Words (BoW) Feature Extraction for Course Recommender System
#
# This script extracts Bag of Words (BoW) features from course titles and descriptions.
# BoW features represent the frequency of words in a text, which are essential for
# transforming textual data into numerical vectors. These vectors enable machine learning
# algorithms to process and understand the content of courses, forming the foundation
# for a content-based recommender system. By filtering out stopwords and focusing on
# meaningful words (optionally, only nouns), this script ensures that the resulting
# features are relevant and useful for identifying user interests in online courses.

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer, CountVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform


# Function to train and save the Bag of Words similarity matrix
def train_bows(desc_df, sim_filepath, stop_words_filepath='../data/stop_words.csv', distance_metric='cosine'):
    # Load additional stopwords
    additional_stop_words = pd.read_csv(stop_words_filepath)['stopword'].tolist()
    stop_words = list(ENGLISH_STOP_WORDS.union(additional_stop_words))

    # Concatenate title and description with title given more weight
    desc_df['TEXT'] = desc_df['TITLE'] + ' ' + desc_df['TITLE'] + ' ' + desc_df['DESCRIPTION']

    # Vectorize text based on the selected distance metric
    if distance_metric == 'jaccard':
        vectorizer = CountVectorizer(stop_words=stop_words, binary=True)
        description_matrix = vectorizer.fit_transform(desc_df['TEXT'])
        description_matrix = description_matrix.toarray()  # Ensure the matrix is dense for Jaccard
        jaccard_distances = pdist(description_matrix, metric='jaccard')
        sim_matrix = 1 - squareform(jaccard_distances)

    elif distance_metric == 'manhattan':
        vectorizer = CountVectorizer(stop_words=stop_words, max_features=5000, ngram_range=(1, 2))
        description_matrix = vectorizer.fit_transform(desc_df['TEXT'])
        description_matrix = normalize(description_matrix, norm='l1', axis=1)
        # Apply Truncated SVD for dimensionality reduction
        svd = TruncatedSVD(n_components=100)
        description_matrix_reduced = svd.fit_transform(description_matrix)
        sim_matrix = 1 - pairwise_distances(description_matrix_reduced, metric=distance_metric)

    elif distance_metric == 'euclidean':
        vectorizer = CountVectorizer(stop_words=stop_words)
        description_matrix = vectorizer.fit_transform(desc_df['TEXT'])
        description_matrix = normalize(description_matrix.toarray())  # Normalize and convert to dense
        # Apply Truncated SVD for dimensionality reduction
        svd = TruncatedSVD(n_components=100)
        description_matrix_reduced = svd.fit_transform(description_matrix)
        sim_matrix = 1 - pairwise_distances(description_matrix_reduced, metric=distance_metric)

    else:  # cosine similarity
        vectorizer = TfidfVectorizer(stop_words=stop_words)
        description_matrix = vectorizer.fit_transform(desc_df['TEXT'])
        sim_matrix = 1 - pairwise_distances(description_matrix, metric=distance_metric)

    # Save the similarity matrix to a CSV file
    sim_df = pd.DataFrame(sim_matrix)
    sim_df.to_csv(sim_filepath, index=False, float_format='%.2f')


# Example usage
# Assuming desc_df is your dataframe with 'TITLE' and 'DESCRIPTION' columns
# train_bows(desc_df, 'similarity_matrix.csv', distance_metric='manhattan')

# Function to recommend courses based on similarity scores
def recommend_courses(idx_id_dict, id_idx_dict, ratings_df, sim_df, user_id, top_courses=10, similarity_threshold=0.5):
    # Get the list of courses the user has enrolled in
    enrolled_courses = ratings_df[ratings_df['user'] == user_id]['item'].to_list()
    sim_matrix = np.array(sim_df)
    unselected_course_ids = set(idx_id_dict.values()).difference(enrolled_courses)

    id_sim_dict = {}

    # Calculate similarity scores for unselected courses
    for enrolled_course in enrolled_courses:
        for unselected_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselected_course in id_idx_dict:
                idx1, idx2 = id_idx_dict[enrolled_course], id_idx_dict[unselected_course]
                sim_score = sim_matrix[idx1][idx2]
                if sim_score >= similarity_threshold:
                    id_sim_dict[unselected_course] = max(id_sim_dict.get(unselected_course, 0), sim_score)

    # Sort and return the top courses based on similarity scores
    id_sim_dict = {k: v for k, v in sorted(id_sim_dict.items(), key=lambda item: item[1], reverse=True)}
    return {k: id_sim_dict[k] for k in list(id_sim_dict.keys())[:top_courses]}
