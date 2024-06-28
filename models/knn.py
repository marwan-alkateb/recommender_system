import pandas as pd
from surprise import KNNBasic
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle

# The dense format of the ratings is more preferred as it saves memory space
# ratings_filepath = pd.read_csv('../data/ratings.csv')

# However the sparse matrix could be beneficial to apply computations such as cosine similarity
# sparse_ratings_df = ratings_filepath.pivot(index='user', columns='item', values='rating').fillna(0).reset_index().rename_axis(
#     index=None, columns=None)


import numpy as np


def train(ratings_df_url, url='knn.pkl', distance_metric='cosine', num_neighbors=10):
    # Read the course rating dataset with columns user item rating
    reader = Reader(line_format='user item rating', sep=',', skip_lines=1, rating_scale=(2, 5))

    # Load the dataset from the CSV file
    course_dataset = Dataset.load_from_file(ratings_df_url, reader=reader)
    trainset, testset = train_test_split(course_dataset, test_size=.3)

    # Define a KNNBasic() model
    sim_options = {
        'name': distance_metric,
        'user_based': False,  # compute similarities between items
    }
    model = KNNBasic(sim_options=sim_options, k=num_neighbors)

    # Train the KNNBasic model on the trainset
    model.fit(trainset)

    # Predict ratings for the testset
    predictions = model.test(testset)

    # Compute RMSE
    rmse = accuracy.rmse(predictions)

    # Save the model to a file
    with open(url, 'wb') as f:
        pickle.dump(model, f)

    return rmse


def recommend_courses(user_id, model_file, top_courses, similarity_threshold):
    if not 0 <= similarity_threshold <= 1:
        raise ValueError('The similarity threshold must be a value between 0 and 1.')

    # Load the model from the file
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    # Get the similarity matrix from the model
    similarity_matrix = model.sim

    # Get a list of all course ids in the training set
    trainset = model.trainset
    all_courses = trainset.all_items()
    course_ids = [trainset.to_raw_iid(course) for course in all_courses]

    # Predict the rating for each course for the given user
    predicted_ratings = {}
    for course_id in course_ids:
        predicted_rating = model.predict(user_id, course_id)
        predicted_ratings[course_id] = predicted_rating.est

    print(f"Predicted ratings: {predicted_ratings}")

    # Normalize the predicted ratings to the range [0, 1]
    min_rating = min(predicted_ratings.values())
    max_rating = max(predicted_ratings.values())

    if min_rating == max_rating:
        # Assign a constant score if all ratings are the same
        normalized_ratings = {course_id: 0.5 for course_id in predicted_ratings.keys()}
    else:
        normalized_ratings = {course_id: (rating - min_rating) / (max_rating - min_rating)
                              for course_id, rating in predicted_ratings.items()}

    print(f"Normalized ratings: {normalized_ratings}")

    # Filter the courses by normalized predicted rating
    filtered_ratings = {course_id: rating for course_id, rating in normalized_ratings.items() if
                        rating >= similarity_threshold}

    print(f"Filtered ratings: {filtered_ratings}")

    # Sort the courses by normalized predicted rating
    sorted_ratings = sorted(filtered_ratings.items(), key=lambda x: x[1], reverse=True)

    # Get the top top_courses course ids and their scores
    recommended_courses = dict(sorted_ratings[:top_courses])

    print(f"Recommended courses: {recommended_courses}")

    # Calculate similarity scores for the top recommended courses
    similarities = {}
    for course_id in recommended_courses.keys():
        inner_id = trainset.to_inner_iid(course_id)
        similarities[course_id] = {}
        for other_course_id in recommended_courses.keys():
            if course_id != other_course_id:
                other_inner_id = trainset.to_inner_iid(other_course_id)
                similarity_score = similarity_matrix[inner_id, other_inner_id]
                similarities[course_id][other_course_id] = similarity_score

    print(f"Similarities between recommended courses: {similarities}")

    return recommended_courses

# Testing:
# rmse = train("../data/ratings.csv")
# print(f"Model trained and saved with RMSE: {rmse}")
# # Recommend 5 courses for user 2
# recommended_courses = recommend_courses(300.0, 'knn.pkl', 10, 0.5)
# print(f"Recommended courses for user 2: {recommended_courses}")