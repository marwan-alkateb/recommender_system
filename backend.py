import pandas as pd
import models.bows as bows
import models.clustering as clustering
import models.user_profile as user_profile
import models.knn as knn
import models.nmf as nmf
from single_recommender import get_nn
import models.regression as regression
import models.classification as classification

# List of available models
MODELS = (
    "Similarity Matrix",
    "User Profile",
    "Clustering",
    "Clustering with PCA",
    "KNN",
    "NMF",
    "Neural Network",
    "Regression with Embedding Features",
    "Classification with Embedding Features"
)

# URLs for data files
URLS = {
    "sims": "data/sim.csv",
    "bows": "data/bows.csv",
    "genres": 'data/genres.csv',
    "ratings": "data/ratings.csv",
    "descriptions": "data/description.csv",
    "user_profiles": "data/generated_user_profile_vectors.csv",
    "pca_user_profiles": "data/generated_pca_user_profile_vectors.csv",
    "user_embeddings": "data/user_embeddings.csv",
    "item_embeddings": "data/item_embeddings.csv",
    "stop_words": "data/stop_words.csv"
}


# Function to load data from CSV files
def load_data(data_type):
    data = pd.read_csv(URLS[data_type])
    return data


# Function to update user ratings and generate user profiles if necessary
def update_ratings(user_id, model_selection, selected_courses, rating_value=4.0):
    res_dict = {}
    ratings_df = load_data('ratings')

    # Remove existing ratings for the user
    ratings_df = ratings_df[ratings_df['user'] != user_id]

    if len(selected_courses) > 0:
        res_dict['user'] = [user_id] * len(selected_courses)
        res_dict['item'] = selected_courses
        res_dict['rating'] = [rating_value] * len(selected_courses)
        new_df = pd.DataFrame(res_dict)
        updated_ratings_df = pd.concat([ratings_df, new_df])
        updated_ratings_df.to_csv(URLS['ratings'], index=False)

        # Generate user profiles for certain models
        if model_selection in [MODELS[0], MODELS[1], MODELS[2], MODELS[3]]:
            user_profile.generate_user_profiles(
                genres_filepath=URLS['genres'],
                ratings_filepath=URLS['ratings'],
                user_profile_filepath=URLS['user_profiles']
            )
            user_profile.apply_pca(load_data('user_profiles'), pca_filepath=URLS['pca_user_profiles'])
        return user_id


# Function to get course dictionaries
def get_courses_dicts():
    bows_df = load_data('bows').groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    courses_idx_id_dict = bows_df['doc_id'].to_dict()
    courses_id_idx_dict = {v: k for k, v in courses_idx_id_dict.items()}
    return courses_idx_id_dict, courses_id_idx_dict


# Function to train the selected model with provided parameters
def train(model_name, params):
    if model_name == MODELS[0]:  # Similarity Matrix
        distance_metric = params.get('distance_metric', 'cosine')
        bows.train_bows(
            desc_df=load_data('descriptions'),
            sim_filepath=URLS['sims'],
            stop_words_filepath=URLS['stop_words'],
            distance_metric=distance_metric
        )
    elif model_name == MODELS[1]:  # User Profile
        pass
    elif model_name in [MODELS[2], MODELS[3]]:  # Clustering
        profile_type = 'user_profiles' if model_name == MODELS[2] else 'pca_user_profiles'
        num_clusters = params.get('num_clusters', 10)
        algorithm_choice = params.get('algorithm_choice', 'kmeans')
        clustering.train(
            load_data(profile_type),
            algorithm_choice=algorithm_choice,
            num_clusters=num_clusters
        )
    elif model_name == MODELS[4]:  # KNN
        distance_metric = params.get('distance_metric', 'cosine')
        num_neighbors = params.get('num_neighbors',10)
        knn.train(URLS['ratings'], distance_metric=distance_metric, num_neighbors=num_neighbors)
    elif model_name == MODELS[5]:  # NMF
        nmf.train_nmf('data/ratings.csv', 'nmf.pkl')
    elif model_name == MODELS[6]:  # Neural Network
        get_nn().train_nn(ratings_df=load_data('ratings'))
    elif model_name == MODELS[7]:  # Regression with Embedding Features
        get_nn().train_nn(ratings_df=load_data('ratings'))
        get_nn().extract_embeddings(
            user_emb_filepath=URLS['user_embeddings'],
            item_emb_filepath=URLS['item_embeddings']
        )
        regression.train_regression(
            load_data('ratings'),
            load_data('user_embeddings'),
            load_data('item_embeddings')
        )
    elif model_name == MODELS[8]:  # Classification with Embedding Features
        get_nn().train_nn(ratings_df=load_data('ratings'))
        get_nn().extract_embeddings(
            user_emb_filepath=URLS['user_embeddings'],
            item_emb_filepath=URLS['item_embeddings']
        )
        classification.train_classification(
            ratings_df=load_data('ratings'),
            user_emb_df=load_data('user_embeddings'),
            item_emb_df=load_data('item_embeddings')
        )
    else:
        raise ValueError('Model is not implemented yet!')


# Function to predict course recommendations
def predict(model_name, user_ids, params):
    top_courses = params.get('top_courses', 10)
    similarity_threshold = params.get('similarity_threshold', 0.5)
    users, courses, scores = [], [], []
    res = {}

    for user_id in user_ids:
        if model_name == MODELS[0]:  # Similarity Matrix
            idx_id_dict, id_idx_dict = get_courses_dicts()
            res = bows.recommend_courses(
                idx_id_dict, id_idx_dict, load_data('ratings'), load_data('sims'), user_id,
                top_courses=top_courses, similarity_threshold=similarity_threshold
            )
        elif model_name == MODELS[1]:  # User Profile
            distance_metric = params.get('distance_metric', 'cosine')
            res = user_profile.recommend_courses(
                user_id=user_id, user_profile_filepath=URLS['user_profiles'],
                ratings_filepath=URLS['ratings'],
                distance_metric=distance_metric, top_courses=top_courses,
                similarity_threshold=similarity_threshold
            )
        elif model_name in [MODELS[2], MODELS[3]]:  # Clustering
            profile_type = 'user_profiles' if model_name == MODELS[2] else 'pca_user_profiles'
            res = clustering.predict_clustering(
                load_data(profile_type), load_data('ratings'), user_id=user_id,
                top_courses=top_courses, similarity_threshold=similarity_threshold
            )
        elif model_name == MODELS[4]:  # KNN
            res = knn.recommend_courses(
                user_id=user_id, model_file='knn.pkl', top_courses=top_courses,
                similarity_threshold=similarity_threshold
            )
        elif model_name == MODELS[5]:  # NMF
            res = nmf.predict_nmf(
                user_id, top_courses=top_courses, similarity_threshold=similarity_threshold
            )
        elif model_name == MODELS[6]:  # Neural Network
            res = get_nn().predict_nn(
                user_id=user_id, top_courses=top_courses,
                similarity_threshold=similarity_threshold
            )
        elif model_name == MODELS[7]:  # Regression with Embedding Features
            res = regression.recommend_courses(
                user_id=user_id, top_courses=top_courses,
                similarity_threshold=similarity_threshold,
                user_emb=load_data('user_embeddings'),
                item_emb=load_data('item_embeddings')
            )
        elif model_name == MODELS[8]:  # Classification with Embedding Features
            res = classification.recommend_courses(
                user_id=user_id, top_courses=top_courses,
                user_emb=load_data('user_embeddings'),
                item_emb=load_data('item_embeddings')
            )
        else:
            raise ValueError('Model is not implemented yet!')

        for key, score in res.items():
            users.append(user_id)
            courses.append(key)
            scores.append(score)

    res_df = pd.DataFrame({
        'USER': users,
        'COURSE_ID': courses,
        'SCORE': scores
    })
    return res_df
