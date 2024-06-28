# single_recommender.py
import streamlit as st
import models.nn as nn

st.cache_resource()
def get_nn():
    return nn.NNRecommender()
