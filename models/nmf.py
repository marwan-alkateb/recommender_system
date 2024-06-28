from surprise import NMF, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
import pickle


def train_nmf(ratings_df_url, model_file='nmf.pkl'):
    # Read the course rating dataset with columns user item rating
    reader = Reader(line_format='user item rating', sep=',', skip_lines=1, rating_scale=(2, 5))

    # Load the dataset from the CSV file
    course_dataset = Dataset.load_from_file(ratings_df_url, reader=reader)
    trainset, testset = train_test_split(course_dataset, test_size=.3)

    # Define a NMF model
    model = NMF(n_factors=32, init_low=0.5, init_high=5.0, verbose=True, random_state=123)

    # Train the NMF model on the trainset
    model.fit(trainset)

    # Predict ratings for the testset
    predictions = model.test(testset)

    # Compute RMSE
    rmse = accuracy.rmse(predictions)

    # Save the model to a file
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    return rmse


def predict_nmf(user_id, top_courses=10, similarity_threshold=0.5, model_file='nmf.pkl'):
    if not 0 <= similarity_threshold <= 1:
        raise ValueError('The similarity threshold must be a value between 0 and 1.')

    # Load the model from the file
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    # Get a list of all course ids in the training set
    trainset = model.trainset
    all_courses = trainset.all_items()
    course_ids = [trainset.to_raw_iid(course) for course in all_courses]

    # Predict the rating for each course for the given user
    predicted_ratings = {}
    for course_id in course_ids:
        predicted_rating = model.predict(user_id, course_id)
        predicted_ratings[course_id] = predicted_rating.est

    # Normalize the predicted ratings to be between 0 and 1
    min_rating = min(predicted_ratings.values())
    max_rating = max(predicted_ratings.values())

    if min_rating == max_rating:
        normalized_ratings = {course_id: 0.5 for course_id in predicted_ratings.keys()}
    else:
        normalized_ratings = {course_id: (rating - min_rating) / (max_rating - min_rating)
                              for course_id, rating in predicted_ratings.items()}

    # Filter the courses by normalized rating
    filtered_ratings = {course_id: rating for course_id, rating in normalized_ratings.items() if
                        rating >= similarity_threshold}

    # Sort the courses by normalized rating
    sorted_ratings = sorted(filtered_ratings.items(), key=lambda x: x[1], reverse=True)

    # Get the top top_courses course ids and their scores
    recommended_courses = dict(sorted_ratings[:top_courses])

    return recommended_courses

# # Train the model and print the RMSE
# rmse = train_nmf("../data/ratings.csv", "nmf.pkl")
# print(f"Model trained and saved with RMSE: {rmse}")
#
# # Recommend 5 courses for user 2 with a similarity threshold of 0.5
# recommended_courses = predict_nmf(2, 5, 0.5, 'nmf.pkl')
# print(f"Recommended courses for user 2: {recommended_courses}")
