import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import pairwise_distances
import numpy as np
import random


def print_bows(desc_df=pd.read_csv('description.csv')):
    additional_stop_words = pd.read_csv('stop_words.csv')['stopword'].tolist()
    stop_words = list(ENGLISH_STOP_WORDS.union(additional_stop_words))
    desc_df['TEXT'] = desc_df['TITLE'] + ' ' + desc_df['TITLE'] + ' ' + desc_df['DESCRIPTION']
    vectorizer = CountVectorizer(stop_words=stop_words)
    description_matrix = vectorizer.fit_transform(desc_df['TEXT'])
    word_freq = description_matrix.sum(axis=0)
    words = vectorizer.get_feature_names_out()
    word_freq_df = pd.DataFrame(word_freq, columns=words).T
    word_freq_df.columns = ['Frequency']
    sorted_word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False)
    sorted_word_freq_df.to_csv('../data/sorted_word_freq.csv')


def train_bows(desc_df, sim_filepath, stop_words_filepath='stop_words.csv'):
    additional_stop_words = pd.read_csv(stop_words_filepath)['stopword'].tolist()
    stop_words = list(ENGLISH_STOP_WORDS.union(additional_stop_words))

    desc_df['TEXT'] = desc_df['TITLE'] + ' ' + desc_df['TITLE'] + ' ' + desc_df['DESCRIPTION']

    vectorizer = TfidfVectorizer(stop_words=stop_words)
    description_matrix = vectorizer.fit_transform(desc_df['TEXT'])
    sim_matrix = 1 - pairwise_distances(description_matrix, metric='cosine')

    sim_df = pd.DataFrame(sim_matrix)
    sim_df.to_csv(sim_filepath, index=False, float_format='%.2f')


def generate_recommendations(idx_id_dict, id_idx_dict, sim_df, enrolled_courses):
    sim_matrix = np.array(sim_df)
    unselected_course_ids = set(idx_id_dict.values()).difference(enrolled_courses)
    id_sim_dict = {}

    for enrolled_course in enrolled_courses:
        for unselected_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselected_course in id_idx_dict:
                idx1, idx2 = id_idx_dict[enrolled_course], id_idx_dict[unselected_course]
                sim_score = sim_matrix[idx1][idx2]
                id_sim_dict[unselected_course] = max(id_sim_dict.get(unselected_course, 0), sim_score)

    return id_sim_dict


def generate_ratings(description_filepath, sim_filepath, stop_words_filepath, output_filepath):
    desc_df = pd.read_csv(description_filepath)

    if desc_df.empty:
        print("Description DataFrame is empty.")
        return

    idx_id_dict = {i: course_id for i, course_id in enumerate(desc_df['COURSE_ID'])}
    id_idx_dict = {course_id: i for i, course_id in enumerate(desc_df['COURSE_ID'])}

    print("Training Bag of Words model...")
    train_bows(desc_df, sim_filepath, stop_words_filepath)
    sim_df = pd.read_csv(sim_filepath)

    if sim_df.empty:
        print("Similarity DataFrame is empty.")
        return

    course_id_to_title = {row['COURSE_ID']: row['TITLE'] for _, row in desc_df.iterrows()}

    ratings_data = []
    user_id = 1.0

    for course_id in desc_df['COURSE_ID']:
        enrolled_courses = [course_id]

        recommendations = generate_recommendations(idx_id_dict, id_idx_dict, sim_df, enrolled_courses)

        # Sort recommendations by similarity score
        sorted_recommendations = sorted(recommendations.items(), key=lambda item: item[1], reverse=True)

        high_score_courses = [k for k, v in sorted_recommendations if v > 0.5]
        mid_score_courses = [k for k, v in sorted_recommendations if 0.1 < v <= 0.5]
        low_score_courses = [k for k, v in sorted_recommendations if 0.0 < v <= 0.1]

        selected_courses = []
        selected_courses.extend(high_score_courses[:3])
        selected_courses.extend(mid_score_courses[:3])
        selected_courses.extend(low_score_courses[:2])

        # Ensure we have enough recommendations by padding with mid and low score courses if necessary
        if len(selected_courses) < 8:
            selected_courses.extend(mid_score_courses[3:])
        if len(selected_courses) < 8:
            selected_courses.extend(low_score_courses[2:])
        if len(selected_courses) > 8:
            selected_courses = selected_courses[:8]

        if len(selected_courses) < 8:
            print(f"Not enough recommendations for course_id {course_id}. Needed 8, got {len(selected_courses)}.")
            continue

        for rec_course in selected_courses:
            if rec_course in high_score_courses[:3]:
                rating = 5.0
            elif rec_course in mid_score_courses[:3]:
                rating = 4.0
            else:
                rating = 3.0

            print(f"Recommendation: {rec_course} with similarity score {recommendations[rec_course]} and rating {rating}")
            ratings_data.append([user_id, rec_course, rating, course_id_to_title[rec_course]])

        user_id += 1.0

    if not ratings_data:
        print("No ratings data generated.")
        return

    ratings_df = pd.DataFrame(ratings_data, columns=['user', 'item', 'rating', 'title'])
    ratings_df.to_csv(output_filepath, index=False, float_format='%.1f')


# Example usage
# generate_ratings('description.csv', 'similarity_matrix.csv', 'stop_words.csv', 'ratings.csv')

# # Delete title column
# ratings_df = pd.read_csv('ratings.csv')
# ratings_df.drop('title',axis=1,inplace=True)
# ratings_df.to_csv('ratings.csv', index=False)
