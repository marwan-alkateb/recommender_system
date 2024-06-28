import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import math


# Function to train classification models
def train_classification(ratings_df, user_emb_df, item_emb_df, test_size=0.3, random_state=123,
                         filepath='classification.pkl'):
    # Merge user embedding features
    user_emb_merged = pd.merge(ratings_df, user_emb_df, how='left', left_on='user', right_on='user').fillna(0)
    # Merge course embedding features
    merged_df = pd.merge(user_emb_merged, item_emb_df, how='left', left_on='item', right_on='item').fillna(0)

    # Extract embedding features
    user_embeddings = merged_df[[f"UFeature{i}" for i in range(16)]]
    course_embeddings = merged_df[[f"CFeature{i}" for i in range(16)]]

    # Aggregate the two feature columns using element-wise addition
    classification_dataset = user_embeddings + course_embeddings.values

    # Rename the columns of the resulting DataFrame
    classification_dataset.columns = [f"Feature{i}" for i in range(16)]

    # Add the 'rating' column from the original DataFrame to the classification dataset
    classification_dataset['rating'] = merged_df['rating']

    # Map the ratings to classes 3, 4, and 5
    classification_dataset['rating'] = classification_dataset['rating'].astype(int)

    X = classification_dataset.iloc[:, :-1]  # Features except rating
    y = classification_dataset.iloc[:, -1]  # Rating

    # Split the dataset into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Define the models
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier()
    }

    # Define the hyperparameters for grid search
    params = {
        'LogisticRegression': {'C': [0.1, 1]},
        'DecisionTree': {'max_depth': [5, 10]},
        'RandomForest': {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    }

    best_model = None
    best_f1 = 0

    # Use tqdm progress bar for model training
    for name in tqdm(models.keys(), desc='Training models'):
        model = models[name]
        grid = GridSearchCV(model, params[name], cv=5, verbose=2, n_jobs=-1)  # Enable verbose output and parallel jobs
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"{name} F1 Score: {f1}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = grid

    # Save the best model to a file
    joblib.dump(best_model, filepath)


# Sigmoid function for transforming predicted ratings to similarity scores
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Function to recommend courses based on trained classification model
def recommend_courses(user_id, top_courses, user_emb, item_emb, filepath='classification.pkl'):
    # Get the user's profile
    user_profile = user_emb[user_emb['user'] == user_id].drop(columns=['user'])

    if user_profile.empty:
        raise ValueError(f"User ID {user_id} not found in user embeddings.")

    model = joblib.load(filepath)

    # Initialize an empty dictionary to store the predicted ratings
    predicted_scores = {}

    # Use tqdm progress bar for course recommendations
    for _, course in tqdm(item_emb.iterrows(), total=item_emb.shape[0], desc='Predicting ratings'):
        # Prepare the input data by merging the user's profile and the course's profile
        course_features = course.drop('item').values.reshape(1, -1)

        if course_features.shape[1] != user_profile.shape[1]:
            raise Exception(f"Course {course['item']} features do not match user profile features.")

        # Combine user and course features
        input_data = user_profile.values + course_features

        # Create a DataFrame with appropriate column names
        input_data_df = pd.DataFrame(input_data, columns=[f"Feature{i}" for i in range(16)])

        # Predict the rating using the best model
        predicted_rating_class = model.predict(input_data_df)[0]

        # Transform the predicted rating into a similarity score between 0 and 1
        predicted_score = sigmoid(predicted_rating_class)

        # Add the course ID and the predicted score to the dictionary
        predicted_scores[course['item']] = predicted_score

    # Sort the dictionary by the predicted scores in descending order
    predicted_scores = {k: v for k, v in sorted(predicted_scores.items(), key=lambda item: item[1], reverse=True)}

    # Return the top courses
    return dict(list(predicted_scores.items())[:top_courses])

# # TODO: Testing
# ratings_filepath = pd.read_csv('../data/ratings.csv')
# user_emb_df = pd.read_csv('../data/user_embeddings.csv')
# item_emb_df = pd.read_csv('../data/item_embeddings.csv')
# train_classification(ratings_filepath, user_emb_df, item_emb_df)
# print(recommend_courses(2, 5, user_emb_df, item_emb_df))
