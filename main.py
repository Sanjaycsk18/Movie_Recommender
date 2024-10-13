 # collaborative_filter.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load MovieLens data
def load_data():
    # Load movies and ratings
    movies = pd.read_csv('data/u.item', sep='|', names=['movie_id', 'title'], usecols=[0, 1], encoding='latin-1')
    ratings = pd.read_csv('data/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    ratings = ratings.drop('timestamp', axis=1)
    
    return movies, ratings

# Create the user-item matrix
def create_user_item_matrix(ratings):
    user_item_matrix = ratings.pivot_table(index='user_id', columns='movie_id', values='rating')
    return user_item_matrix

# Compute cosine similarity between users
def compute_user_similarity(user_item_matrix):
    # Fill NaN with 0 (treat unrated movies as 0 rating)
    user_item_matrix_filled = user_item_matrix.fillna(0)
    similarity_matrix = cosine_similarity(user_item_matrix_filled)
    similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
    
    return similarity_df

# Predict ratings based on user similarity
def predict_ratings(user_item_matrix, similarity_df, user_id, top_n_similar_users=10):
    # Get the similarity scores for the user
    user_similarity_scores = similarity_df[user_id].sort_values(ascending=False)
    
    # Exclude the user themselves from the similarity scores
    similar_users = user_similarity_scores.iloc[1:top_n_similar_users + 1].index
    
    # Get the ratings of the similar users
    similar_users_ratings = user_item_matrix.loc[similar_users]
    
    # Compute the weighted average of ratings from similar users for all movies
    weighted_ratings = np.dot(similar_users_ratings.T, user_similarity_scores[similar_users])
    similarity_sums = user_similarity_scores[similar_users].sum()
    
    # Avoid division by zero
    predicted_ratings = weighted_ratings / (similarity_sums + 1e-10)
    
    # Create a DataFrame of predicted ratings
    predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.columns, columns=['predicted_rating'])
    
    return predicted_ratings_df

# Recommend movies to a user
def get_recommendations(predicted_ratings_df, user_item_matrix, user_id, movies, num_recommendations=10):
    # Get the user's actual ratings
    user_ratings = user_item_matrix.loc[user_id]
    
    # Find movies the user has not rated
    unrated_movies = user_ratings[user_ratings.isna()].index
    
    # Get predicted ratings for the unrated movies
    recommended_movies = predicted_ratings_df.loc[unrated_movies].sort_values(by='predicted_rating', ascending=False)
    
    # Get the top N recommended movie ids
    top_movie_ids = recommended_movies.head(num_recommendations).index
    
    # Return the movie titles
    recommended_movie_titles = movies[movies['movie_id'].isin(top_movie_ids)]['title']
    
    return recommended_movie_titles.tolist()
