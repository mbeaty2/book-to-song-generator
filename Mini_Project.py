# ## Personalization and Machine Learning
# ### Mini-Project - Utilizing a Content Based Recommendation System to Recommend Books Based on User Input Music Playlists
# By Marissa Beaty

#main code
import argparse
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, KeyedVectors
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity as cosine
from sklearn.preprocessing import StandardScaler
from functions import clean_text, calculate_speechiness, calculate_valence, calculate_energy, calculate_instrumentalness, get_playlist_data, authenticate_spotify

# Load the book dataset
book_dataset = pd.read_csv("books_and_genres.csv")

# Define the book title
book_title = "persuasion"

# Get book data
book_data = book_dataset.loc[book_dataset['title'] == book_title]

# Clean the text
book_data["text"] = book_data["text"].str.replace("Produced by Sharon Partridge and Martin Ward.", "")
book_data["text"] = book_data["text"].str.replace("HTML version\nby Al Haines.", "")

# Calculate speechiness
speechiness = calculate_speechiness(book_data)

# Calculate valence
unique_words = clean_text(book_data["text"])
valence = calculate_valence(unique_words)

# Calculate energy
genres = [genre.strip("'") for genre in book_data["genres"].str.split(", ").sum()]
energy = calculate_energy(genres)

# Calculate instrumentalness
total_words = len(book_data["text"].str.split())
instrumentalness = calculate_instrumentalness(total_words)

# Create a dataframe for the features
book_to_features = pd.DataFrame()

# Append the features to the dataframe
book_to_features['speechiness'] = [speechiness]
book_to_features['valence'] = [valence]
book_to_features['energy'] = [energy]
book_to_features['instrumentalness'] = [instrumentalness]

#formating everything correctly to match with the spotify system
labels = [book_title]
book_to_features.index = labels

# Display the final feature dataframe
print("Book Features: ")
print(book_to_features)

def main():
    parser = argparse.ArgumentParser(description="Get data from a Spotify playlist and perform recommendations.")
    parser.add_argument("--playlist_id", required=True, help="ID of the Spotify playlist")
    parser.add_argument("--limit", type=int, default=100, help="Number of songs to retrieve")
    parser.add_argument("--trim", action="store_true", help="Trim the playlist")

    args = parser.parse_args()

    # Authenticate with Spotify
    client_id = 'YOUR_CLIENT_ID'
    client_secret = 'YOUR_CLIENT_SECRET'
    sp = authenticate_spotify(client_id, client_secret)

    # Get playlist features
    playlist_features = get_playlist_data(sp, args.playlist_id, args.limit, args.trim)

    # Add book features to the playlist features
    playlist_features = playlist_features.append(book_to_features)

    # Get subset
    reduced_features = ['speechiness', 'valence', 'energy', 'instrumentalness']
    subset_features = playlist_features[reduced_features]

    # Standardize features
    scaled_features = StandardScaler().fit_transform(subset_features)

    # Calculate cosine similarities
    similarities = cosine(scaled_features)
    similarities = pd.DataFrame(similarities, columns=playlist_features.index, index=playlist_features.index)

    # Select top N similar tracks
    book_track = "persuasion"  # Put in the book title you have selected
    n = 10
    similar_tracks = similarities.sort_values(by=book_track, ascending=False)[book_track].index[1:n + 1]
    print(f"Top {n} similar tracks to {book_track}:")
    print(similar_tracks)

if __name__ == "__main__":
    main()
