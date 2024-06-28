import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import math
from sklearn.metrics.pairwise import cosine_similarity

def train_regression(ratings_df, user_emb_df, item_emb_df, test_size=0.3, random_state=123, filepath='regression.pkl'):
    # Merge user embedding features
    user_emb_merged = pd.merge(ratings_df, user_emb_df, how='left', left_on='user', right_on='user').fillna(0)
    # Merge course embedding features
    merged_df = pd.merge(user_emb_merged, item_emb_df, how='left', left_on='item', right_on='item').fillna(0)

    # Extract embedding features
    user_embeddings = merged_df[[f"UFeature{i}" for i in range(16)]]
    course_embeddings = merged_df[[f"CFeature{i}" for i in range(16)]]

    # Combine user and course features by concatenation
    regression_dataset = pd.concat([user_embeddings, course_embeddings], axis=1)

    # Rename the columns of the resulting DataFrame
    regression_dataset.columns = [f"UFeature{i}" for i in range(user_embeddings.shape[1])] + [f"CFeature{i}" for i in range(course_embeddings.shape[1])]

    # Add the 'rating' column from the original DataFrame to the regression dataset
    regression_dataset['rating'] = merged_df['rating']

    X = regression_dataset.iloc[:, :-1]  # features except rating
    y = regression_dataset.iloc[:, -1]  # rating

    # Split the dataset into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Define the models
    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet()
    }

    # Define the hyperparameters to tune
    params = {
        'Ridge': {'alpha': [1e-3, 1e-2, 1e-1, 1]},
        'Lasso': {'alpha': [1e-3, 1e-2, 1e-1, 1]},
        'ElasticNet': {'alpha': [1e-3, 1e-2, 1e-1, 1], 'l1_ratio': [0.2, 0.5, 0.8]}
    }

    best_model = None
    best_rmse = float('inf')

    for name in tqdm(models.keys()):
        model = models[name]
        grid = GridSearchCV(model, params[name], cv=5)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"{name} RMSE: {rmse}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = grid

    # Save the best model to a file
    joblib.dump(best_model, filepath)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def recommend_courses(user_id, top_courses, similarity_threshold, user_emb, item_emb, filepath='regression.pkl'):
    # Load the trained model
    model = joblib.load(filepath)

    # Get the user's profile
    user_profile = user_emb[user_emb['user'] == user_id].drop(columns=['user'])

    if user_profile.empty:
        raise ValueError(f"User ID {user_id} not found in user embeddings.")

    # Ensure user_profile is a single row
    user_profile = user_profile.iloc[0].values.reshape(1, -1)

    # Initialize an empty dictionary to store the predicted ratings
    predicted_ratings = {}

    # Initialize an empty dictionary to store similarity scores
    similarity_scores = {}

    # Iterate over all courses
    for _, course in item_emb.iterrows():
        # Prepare the input data by concatenating the user's profile and the course's profile
        course_features = course.drop('item').values.reshape(1, -1)

        # Compute cosine similarity between user profile and course features
        similarity = cosine_similarity(user_profile, course_features)[0][0]

        # Ensure similarity is within the range [0, 1]
        similarity = (similarity + 1) / 2

        similarity_scores[course['item']] = similarity

        # Combine user and course features by concatenation
        input_data = np.hstack((user_profile, course_features))

        # Create a DataFrame with appropriate column names
        feature_names = [f"UFeature{i}" for i in range(user_profile.shape[1])] + [f"CFeature{i}" for i in range(course_features.shape[1])]
        input_data_df = pd.DataFrame(input_data, columns=feature_names)

        # Predict the rating using the best model
        predicted_rating = model.predict(input_data_df)[0]

        # Apply sigmoid function to ensure the rating is between 0 and 1
        predicted_rating = sigmoid(predicted_rating)

        # Add the course ID and the predicted rating to the dictionary
        predicted_ratings[course['item']] = predicted_rating

    # Sort the dictionary by the predicted ratings in descending order
    predicted_ratings = {k: v for k, v in sorted(predicted_ratings.items(), key=lambda item: item[1], reverse=True)}

    # Filter the dictionary to only include courses with a predicted rating above the threshold
    filtered_ratings = {k: v for k, v in predicted_ratings.items() if v > similarity_threshold}

    # Sort the filtered ratings by similarity scores in descending order
    sorted_filtered_ratings = {k: v for k, v in sorted(filtered_ratings.items(), key=lambda item: similarity_scores[item[0]], reverse=True)}

    # Return the top top_courses courses by similarity scores
    return {k: similarity_scores[k] for k in list(sorted_filtered_ratings.keys())[:top_courses]}

# Usage example:
# ratings_filepath = pd.read_csv('../data/ratings.csv')
# user_emb_df = pd.read_csv('../data/user_embeddings.csv')
# item_emb_df = pd.read_csv('../data/item_embeddings.csv')
# train_regression(ratings_filepath, user_emb_df, item_emb_df)
# top_courses_similarity = recommend_courses(300.0, 5, 0.8, user_emb_df, item_emb_df)
# print(top_courses_similarity)
