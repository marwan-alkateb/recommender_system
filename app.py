import streamlit as st
import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode
import backend as backend

# Basic webpage setup
st.set_page_config(
    page_title="Course Recommender System",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Function to load data for course descriptions and genres
def load_data():
    data_types = ["descriptions", "genres"]
    return tuple(backend.load_data(data_type) for data_type in data_types)


# Function to load user ratings
def load_ratings():
    ratings = backend.load_data('ratings')
    return ratings


# Initialize the app by first loading datasets
def init_recommender_app():
    # Display a general information note about the prediction process if not already shown
    if 'note_shown' not in st.session_state:
        st.session_state['note_shown'] = False

    info_placeholder = st.empty()

    if not st.session_state['note_shown']:
        with info_placeholder.container():
            st.info(
                'Note: The prediction process for most of the models is based on the available user profiles in our databases and the genres associated with these courses. This ensures personalized and relevant recommendations.'
            )
            if st.button('Dismiss', key='dismiss_button'):
                st.session_state['note_shown'] = True
                info_placeholder.empty()

    with st.spinner('Loading datasets... This might take a few seconds!'):
        descriptions_df, genres_df = load_data()

    st.subheader("Select courses that you have audited or completed:")

    # Build an interactive table for `descriptions_df`
    gb = GridOptionsBuilder.from_dataframe(descriptions_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    # Create a grid response
    response = AgGrid(
        descriptions_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )

    # Capture selected courses
    selected_courses = pd.DataFrame(response["selected_rows"], columns=['COURSE_ID', 'TITLE', 'DESCRIPTION'])
    selected_courses = selected_courses[['COURSE_ID', 'TITLE']]
    st.subheader("Your courses:")
    st.table(selected_courses)

    user_id = load_ratings()['user'].max()

    return user_id, selected_courses, descriptions_df, genres_df


# Train the selected model with user data
def train(params, user_id, model_selection, selected_courses):
    if model_selection in backend.MODELS:
        with st.spinner('Your selected courses are being saved..'):
            backend.update_ratings(user_id, model_selection, selected_courses)

        if len(selected_courses) > 0:
            with st.spinner('We are training your model... Please wait! This might take a bit!'):
                backend.train(model_selection, params)
                st.success('Training Done!')
                st.session_state['trained'] = True
        else:
            st.warning('You need to select courses first!')
    else:
        st.error('Model does not exist in the MODELS list!')


# Predict course recommendations
def predict(params, user_id, model_selection, selected_courses):
    if 'trained' not in st.session_state or not st.session_state['trained']:
        st.error('Model is not trained yet. Please train the model first.')
        return

    user_ids = [backend.update_ratings(user_id, model_selection, selected_courses)]

    with st.spinner('Generating course recommendations...'):
        if model_selection in backend.MODELS:
            res = backend.predict(model_selection, user_ids, params)

    if res.shape[0] > 0:
        st.success(f'{res.shape[0]} Recommendation{"s" if res.shape[0] != 1 else ""} generated 😀')
    else:
        st.warning('Your choice of hyperparameters introduces constraints on the recommendation process. '
                   'To achieve favorable outcomes, it is advisable to explore other hyperparameter configurations.')

    # Adjust score display format
    res['SCORE'] = (res['SCORE'] * 100).astype(int)

    return res


# Check if hyperparameters have changed
def check_hyperparameters_change(new_params, model_selection):
    if 'params' not in st.session_state:
        st.session_state['params'] = new_params
        st.session_state['model_selection'] = model_selection
        st.session_state['trained'] = False
        return True
    if (st.session_state['model_selection'] != model_selection or
            'distance_metric' in new_params and st.session_state['params'].get('distance_metric') != new_params[
                'distance_metric'] or
            'num_clusters' in new_params and st.session_state['params'].get('num_clusters') != new_params[
                'num_clusters'] or
            'algorithm_choice' in new_params and st.session_state['params'].get('algorithm_choice') != new_params[
                'algorithm_choice']):
        st.session_state['params'] = new_params
        st.session_state['model_selection'] = model_selection
        st.session_state['trained'] = False
        return True
    return False


# Initialize the app
user_id, selected_courses_df, descriptions_df, genres_df = init_recommender_app()

# Sidebar | Selecting a recommendation model
st.sidebar.title('Personalized Learning Recommender')
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.subheader('1. Recommendation model')
model_selection = st.sidebar.selectbox("Select model:", backend.MODELS)

# Sidebar | Tuning hyperparameters
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.subheader('2. Tune Hyper-parameters:')
params = {
    'top_courses': st.sidebar.slider('Top Courses', min_value=1, max_value=100, value=20),
    'similarity_threshold': st.sidebar.slider('Similarity Threshold', min_value=0, max_value=100, value=20) / 100
}

# Specific hyperparameters for certain models
if model_selection in [backend.MODELS[0], backend.MODELS[1]]:
    params['distance_metric'] = st.sidebar.selectbox("Select distance metric",
                                                     ('cosine', 'euclidean', 'jaccard', 'manhattan'))

if model_selection in [backend.MODELS[4]]:
    params['num_neighbors'] = st.sidebar.slider('Number of Neighbors', min_value=1, max_value=20, value=10)
    params['distance_metric'] = st.sidebar.selectbox("Select distance metric",
                                                     ('cosine', 'msd', 'pearson', 'pearson_baseline'))

# Clustering model
elif model_selection in [backend.MODELS[2], backend.MODELS[3]]:
    params['num_clusters'] = st.sidebar.slider('Number of Kmeans Clusters', min_value=0, max_value=20, value=5, step=1)
    params['algorithm_choice'] = st.sidebar.selectbox("Select clustering algorithm:", ("kmeans", "dbscan"))

# Check if hyperparameters changed
hyperparameters_changed = check_hyperparameters_change(params, model_selection)

# Sidebar | Training
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.subheader('3. Training:')
if st.sidebar.button("Train Model"):
    train(params, user_id, model_selection, selected_courses=selected_courses_df['COURSE_ID'].values)

# Sidebar | Prediction
st.sidebar.subheader('4. Prediction')
if st.sidebar.button("Recommend New Courses") and selected_courses_df.shape[0] > 0:
    res_df = predict(params, user_id, model_selection, selected_courses_df['COURSE_ID'].values)
    if res_df is not None:
        res_df = res_df[['COURSE_ID', 'SCORE']]
        final_df = pd.merge(res_df, descriptions_df, on=["COURSE_ID"]).drop('COURSE_ID', axis=1)
        final_df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)
        final_df.index = final_df.index + 1
        st.table(final_df)
