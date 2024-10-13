import streamlit as st
import pandas as pd
from main import load_data, create_user_item_matrix, compute_user_similarity, predict_ratings, get_recommendations

# Title of the app
st.title("Movie Recommender System (Collaborative Filtering)")

# Load data
movies, ratings = load_data()

# Display dataset stats
st.sidebar.header("MovieLens Dataset")
st.sidebar.write(f"Number of users: {ratings['user_id'].nunique()}")
st.sidebar.write(f"Number of movies: {movies['movie_id'].nunique()}")
st.sidebar.write(f"Number of ratings: {ratings.shape[0]}")

# Create the user-item matrix
user_item_matrix = create_user_item_matrix(ratings)

# Handle session state for similarity matrix
if "similarity_df" not in st.session_state:
    st.session_state.similarity_df = None

# Compute the user similarity matrix
st.sidebar.header("Model Training")
if st.sidebar.button("Compute Similarity"):
    st.session_state.similarity_df = compute_user_similarity(user_item_matrix)
    st.sidebar.success("User similarity matrix computed!")

# Get user input for recommendations
user_id = st.number_input("Enter User ID", min_value=1, max_value=ratings['user_id'].nunique())
num_recommendations = st.slider("Number of Recommendations", min_value=1, max_value=20, value=10)

# Generate recommendations
if st.button("Get Recommendations"):
    if st.session_state.similarity_df is None:
        st.warning("Please compute the user similarity matrix first!")
    else:
        predicted_ratings_df = predict_ratings(user_item_matrix, st.session_state.similarity_df, user_id)
        recommendations = get_recommendations(predicted_ratings_df, user_item_matrix, user_id, movies, num_recommendations)
        
        if recommendations:
            st.write(f"Top {num_recommendations} recommendations for User {user_id}:")
            for i, movie in enumerate(recommendations, 1):
                st.write(f"{i}. {movie}")
        else:
            st.write(f"No recommendations found for User {user_id}.")
