# Course Recommender System

This project is a Course Recommender System that uses various machine learning algorithms to recommend courses to users
based on their past interactions and course content. The system employs different models including Similarity Matrix,
User Profile, Clustering, KNN, NMF, Neural Networks, Regression with Embedding Features, and Classification with
Embedding Features.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Usage](#usage)
- [Models](#models)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

```
CourseRecommenderSystem/
│
├── data/                           # Contains all the data files required
│   ├── sim.csv
│   ├── bows.csv
│   ├── genres.csv
│   ├── ratings.csv
│   ├── description.csv
│   ├── generated_user_profile_vectors.csv
│   ├── generated_pca_user_profile_vectors.csv
│   ├── user_embeddings.csv
│   ├── item_embeddings.csv
│   ├── stop_words.csv
│
├── models/                         # Contains all the models scripts
│   ├── bows.py
│   ├── clustering.py
│   ├── user_profile.py
│   ├── knn.py
│   ├── nmf.py
│   ├── regression.py
│   ├── classification.py
│
├── venv/                           # Virtual environment folder
│
├── app.py                          # Streamlit app
├── backend.py                      # Backend processing script
├── single_recommender.py           # Neural Network model script
├── README.md                       # Project README file
│
└── requirements.txt                # Required dependencies
```

## Installation

1. **Clone the repository:**
    ```sh
    git https://github.com/marwan-alkateb/recommender_system.git
    cd recommender_system
    ```

2. **Create a virtual environment:**
    ```sh
    python -m venv venv
    ```

3. **Activate the virtual environment:**
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On MacOS/Linux:
        ```sh
        source venv/bin/activate
        ```

4. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Running the Project

To run the Course Recommender System, follow these steps:

1. Open terminal and navigate to the project folder:
    ```sh
    cd recommender_system
    ```

2. Make sure that the virtual environment is created and all requirements are satisfied:
    ```sh
    python -m venv venv
    pip install -r requirements.txt
    ```

3. Activate the virtual environment:
    ```sh
    venv\Scripts\activate
    ```

4. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

## Usage

Once the Streamlit app is running, you can interact with the Course Recommender System via the web interface. The app
will allow you to select courses you have completed, choose a recommendation model, tune hyperparameters, train the
model, and get course recommendations.

## Models

The following machine learning models are implemented in this project:

- **Similarity Matrix**: Uses cosine similarity to recommend courses based on content.
- **User Profile**: Builds user profiles from course genres and recommends similar courses.
- **Clustering**: Clusters users based on their course interactions and recommends courses from the same cluster.
- **KNN (K-Nearest Neighbors)**: Recommends courses based on item similarity.
- **NMF (Non-negative Matrix Factorization)**: Decomposes the user-item matrix for recommendations.
- **Neural Networks**: Uses deep learning for recommendation.
- **Regression with Embedding Features**: Uses regression models on embedding features for recommendation.
- **Classification with Embedding Features**: Uses classification models on embedding features for recommendation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```

