"""
Exploratory data Analysis on Online Course Enrollment data

This script performs exploratory data analysis (eda) on online course datasets, including course titles, genres, and enrollments.
The main objectives are:
- Identify keywords in course titles using a WordCloud.
- Summarize and visualize the course content dataset.
- Determine popular course genres.
- Analyze and visualize the course enrollment dataset.
- Identify courses with the greatest number of enrolled students.

Importance for Recommender System:
This script is essential for building a personalized course recommender system. By understanding the distribution of course genres,
identifying popular courses, and analyzing user ratings, we can extract meaningful features and insights. These insights will
help in designing algorithms that accurately predict user preferences and recommend the most relevant courses, thereby enhancing
user experience and engagement.
"""

"""
Exploratory data Analysis on Online Course Enrollment data
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

RANDOM_STATE = 123
STOPWORDS_SET = set(STOPWORDS).union(
    ["getting started", "using", "enabling", "template", "university", "end", "introduction", "basic"])
FIGURE_SIZE = (10, 5)


def generate_wordcloud(titles, stopwords):
    wordcloud = WordCloud(stopwords=stopwords, background_color="white", width=800, height=400).generate(titles)
    plt.figure(figsize=FIGURE_SIZE)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def get_genre_counts(df, genres, ascending=False):
    genre_sums = df[genres].sum(axis=0)
    genre_counts_df = pd.DataFrame(genre_sums, columns=['Count'])
    genre_counts_df = genre_counts_df.sort_values(by="Count", ascending=ascending)
    return genre_counts_df


def plot_counts(counts_df, title):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    counts_df.plot(kind='bar', ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Count')
    ax.set_xticklabels(counts_df.index, rotation=90)  # Rotate x-axis labels by 90 degrees
    plt.tight_layout()
    plt.show()


def plot_user_ratings_count(ratings_df):
    user_ratings_count = ratings_df.groupby('user').size().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    plt.hist(user_ratings_count, bins=60, edgecolor='black')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    plt.title('Histogram of User Rating Counts')
    plt.show()


def get_top_courses(ratings_df, course_df, top_n=10):
    enrolls_df = ratings_df.groupby('item').size().sort_values(ascending=False).iloc[:top_n].reset_index()
    enrolls_df.columns = ['COURSE_ID', 'Enrollments']
    merged_df = pd.merge(enrolls_df, course_df[['COURSE_ID', 'TITLE']], on='COURSE_ID', how='left')
    total = ratings_df.shape[0]
    merged_df['Enrollment Percentage'] = (merged_df['Enrollments'] / total) * 100
    return merged_df


def visualize_course_similarity(course_genres_df):
    genre_vectors = course_genres_df.iloc[:, 2:].values
    similarity_matrix = np.zeros((len(genre_vectors), len(genre_vectors)))

    for i in tqdm(range(len(genre_vectors)), desc="Calculating similarities"):
        for j in range(i):
            similarity = cosine_similarity([genre_vectors[i]], [genre_vectors[j]])[0, 0]
            similarity_matrix[i, j] = similarity

    similarity_df = pd.DataFrame(similarity_matrix, index=course_genres_df['COURSE_ID'],
                                 columns=course_genres_df['COURSE_ID'])

    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_df, annot=False, cmap='coolwarm', center=0, linewidths=.5)
    plt.title('Course Similarity Matrix')
    plt.xlabel('Course ID')
    plt.ylabel('Course ID')
    plt.show()


def plot_rating_histogram(course_ratings):
    # Group by rating and count occurrences
    rating_counts = course_ratings.groupby('rating').size().sort_index()
    # Define colors for each rating
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    # Plotting the histogram with different colors for different ratings
    plt.figure(figsize=(8, 6))
    bars = plt.bar(rating_counts.index, rating_counts.values, color=colors)
    # Adding labels and title
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.title('Frequency of Each Rating')
    plt.xticks(rating_counts.index)
    plt.yticks(range(0, max(rating_counts.values) + 1000, 5000))
    # Adding text annotations above each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 100, round(yval), ha='center', va='bottom')
    plt.tight_layout()
    plt.show()


def plot_popularity_trends(course_ratings):
    # Calculate average ratings and number of ratings per course
    course_stats = course_ratings.groupby('item').agg({'rating': ['mean', 'count']})
    course_stats.columns = ['average_rating', 'num_ratings']
    course_stats = course_stats.reset_index()

    # Sort courses by average rating
    course_stats = course_stats.sort_values(by='average_rating', ascending=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(course_stats['average_rating'], course_stats['num_ratings'], alpha=0.5)
    plt.title('Popularity Trends of Courses')
    plt.xlabel('Average Rating')
    plt.ylabel('Number of Ratings')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Set display options
    pd.set_option('display.max_columns', None)

    # Load datasets
    course_ratings = pd.read_csv('../data/ratings.csv')
    course_genres = pd.read_csv('../data/genres.csv')

    plot_popularity_trends(course_ratings)

    print ( course_ratings.groupby('rating').size().sort_values(ascending=False))

    # Group by rating and count occurrences
    rating_counts = course_ratings.groupby('rating').size().sort_index()

    titles = ' '.join(title for title in course_genres['TITLE'])
    generate_wordcloud(titles, STOPWORDS_SET)

    visualize_course_similarity(course_genres)
    #
    # Generate and plot course counts per genre
    genre_counts_df = get_genre_counts(df=course_genres, genres=course_genres.columns[2:])
    plot_counts(genre_counts_df, title='Course Counts per Genre')
    #
    # Plot rating counts for each user
    plot_user_ratings_count(course_ratings)

    # Get and display the top-10 most popular courses
    top_courses_df = get_top_courses(course_ratings, course_genres, top_n=10)
    print(top_courses_df)

"""
SUMMARY
This script performed an exploratory data analysis (eda) on the course metadata and course enrollments datasets. 
It provided preliminary insights into the data, setting the stage for developing a personalized course recommender system. 
Future steps will focus on processing these datasets to extract features suitable for machine learning tasks.
"""
