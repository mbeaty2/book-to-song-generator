import pandas as pd
import numpy as np
from scipy.spatial import distance
from gensim.models import Word2Vec, KeyedVectors
import re
import nltk
from nltk.corpus import stopwords
import flair

def clean_text(text):
    regexed_text = book_data["text"].str.replace("[^A-Za-z0-9]+", " ")
    
    #making everything in lower case
    for text in regexed_text:
        text= str(text).lower()
        split_text = (text.split(" "))

    return split_text

def calculate_speechiness(split_text):
    regexed_text = book_data["text"].str.replace("[^A-Za-z0-9]+", " ")
    
    for text in regexed_text:
    text= str(text).lower()
    split_text = (text.split(" "))
    
    total_words = len(split_text)
    
    nltk.download('stopwords')
    stopwords = stopwords.words('english')

    #creating my own list of additional stop words
    less_wordiness = ['like', 'always', 'now', 'currently', 'because', 'by',
                      'for', 'can', 'to', 'although', 'though', 'if', 'finally',
                      'like', 'about', 'when', 'until']

    #appending my list to the list of stop words
    for words in less_wordiness:
        if words not in stopwords:
            stopwords.append(words)
            
    #get the ratio of stop words to non-stop words
    stopwords_count = []
    for words in regexed_text:
        for word in words.split():
            if word in stopwords:
                stopwords_count.append(word)
                
    speechiness = len(stopwords_count)/total_words
    
    return speechiness

def calculate_valence(text):
    
    regexed_text = book_data["text"].str.replace("[^A-Za-z0-9]+", " ")
    
    unique_words = []

    for words in regexed_text:
        for word in words.split():
            if word not in unique_words:
                unique_words.append(word)
            
    flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
    s = flair.data.Sentence(unique_words)
    flair_sentiment.predict(s)
    valence = s.labels[0].score
    
    return valence

def calculate_energy(genres):
    book_dataset["genres"] = book_dataset["genres"].str.strip('{}')

    #pulling out all unique genres and adding them to a new list

    book_genres = []
    for genre in book_dataset["genres"]:
        genre = genre.split(", ")
        for g in genre:
            g = g.strip("'")
            if g not in book_genres:
                book_genres.append(g)

    #importing the google news vectors database
    embeddings_file = "GoogleNews-vectors-negative300.bin"

    #getting word vectors for all words in the database and saving it as a variable
    wv = KeyedVectors.load_word2vec_format(embeddings_file, binary=True) #limit=200000) - I removed the limit to ensure I can find all

    #create a dictionary to hold my genres and energy similarities
    book_dict = {key: i for i, key in enumerate(book_genres)}

    #getting word vectors for the genres in our dataset and determining if some are not there
    genres_not_found = {}
    for genre in book_genres:
        if genre in wv.key_to_index:
            book_dict[genre] = wv[genre]
        else:
            book_dict = book_dict
            genres_not_found[genre] = book_dict[genre] #gives us a list of all the book genres not found in the previous dataset.

    #removing the hyphens to see if that finds any more word vectors
    joined_genres_list = []
    for genre in genres_not_found:
        genre = genre.split("-")
        joined_genres = ''.join(genre)
        if joined_genres in wv.key_to_index:
            joined_genres_list.append(joined_genres)
    joined_genres_list #it returns four terms found in the dataset

    #edit my dictionary to fit the names found in the wv directory
    book_dict['nonfiction'] = book_dict.pop('non-fiction')
    book_dict['highschool'] = book_dict.pop('high-school')
    book_dict['chicklit'] = book_dict.pop('chick-lit')
    book_dict['selfhelp'] = book_dict.pop('self-help')
    #these others contain values that are not contained in the directory exactly, but are in a different format, so I will alter them to fit that format
    #we will also have to split them and find their vectors separately
    book_dict['twenty-first-century'] = book_dict.pop('21st-century')
    book_dict['twentieth-century'] = book_dict.pop('20th-century')
    book_dict['coming-age'] = book_dict.pop('coming-of-age')

    #returning to the rest of our hiephenated list
    #finds the four un-hyphenated genres we found above and adds them back to our original list
    #adds what's left to a new list
    hiephenated_genres = []
    for genre in book_dict:
        if genre in wv.key_to_index:
            book_dict[genre] = wv[genre]
        else:
            book_dict = book_dict
            hiephenated_genres.append(genre)

    genres_split = []
    for genre in hiephenated_genres:
        genre = genre.split("-")
        genres_split.append(genre)

    missing_word_vectors = {}
    for genres in genres_split:
        for genre in genres:
            missing_word_vectors[genre] = wv[genre]

    #now I need to add the paired word vectors together to get one singular vector for each genre
    #start by creating a new dictionary to hold it all
    missing_word_vectors_added = {}
    missing_word_vectors_added['short-stories'] = np.add(missing_word_vectors['short'], missing_word_vectors['stories'])/2
    #missing_word_vectors_added['read-for-school'] = np.add(missing_word_vectors['read'], missing_word_vectors['for'], missing_word_vectors['school'])/3
    missing_word_vectors_added['science-fiction'] = np.add(missing_word_vectors['science'], missing_word_vectors['fiction'])/2
    missing_word_vectors_added['speculative-fiction'] = np.add(missing_word_vectors['speculative'], missing_word_vectors['fiction'])/2
    missing_word_vectors_added['literary-fiction'] = np.add(missing_word_vectors['literary'], missing_word_vectors['fiction'])/2
    missing_word_vectors_added['historical-fiction'] = np.add(missing_word_vectors['historical'], missing_word_vectors['fiction'])/2
    missing_word_vectors_added['adult-fiction'] = np.add(missing_word_vectors['adult'], missing_word_vectors['fiction'])/2
    missing_word_vectors_added['mystery-thriller'] = np.add(missing_word_vectors['mystery'], missing_word_vectors['thriller'])/2
    missing_word_vectors_added['graphic-novels'] = np.add(missing_word_vectors['graphic'], missing_word_vectors['novels'])/2
    missing_word_vectors_added['picture-books'] = np.add(missing_word_vectors['picture'], missing_word_vectors['books'])/2
    missing_word_vectors_added['young-adult'] = np.add(missing_word_vectors['young'], missing_word_vectors['adult'])/2
    missing_word_vectors_added['urban-fantasy'] = np.add(missing_word_vectors['urban'], missing_word_vectors['fantasy'])/2
    missing_word_vectors_added['paranormal-romance'] = np.add(missing_word_vectors['paranormal'], missing_word_vectors['romance'])/2
    missing_word_vectors_added['middle-grade'] = np.add(missing_word_vectors['middle'], missing_word_vectors['grade'])/2
    missing_word_vectors_added['new-adult'] = np.add(missing_word_vectors['new'], missing_word_vectors['adult'])/2
    missing_word_vectors_added['realistic-fiction'] = np.add(missing_word_vectors['realistic'], missing_word_vectors['fiction'])/2
    missing_word_vectors_added['historical-romance'] = np.add(missing_word_vectors['historical'], missing_word_vectors['romance'])/2
    missing_word_vectors_added['contemporary-romance'] = np.add(missing_word_vectors['contemporary'], missing_word_vectors['romance'])/2
    missing_word_vectors_added['romantic-suspense'] = np.add(missing_word_vectors['romantic'], missing_word_vectors['suspense'])/2
    #missing_word_vectors_added['twenty-first-century'] = np.add(missing_word_vectors['twenty'], missing_word_vectors['first'], missing_word_vectors['century'])/3
    missing_word_vectors_added['twentieth-century'] =  np.add(missing_word_vectors['twentieth'], missing_word_vectors['century'])/2
    missing_word_vectors_added['coming-age'] = np.add(missing_word_vectors['coming'], missing_word_vectors['age'])/2

    #the two three vectors will have to be added in parts because they cannot be added like the above

    #adding my two three-arrayed genres
    #creating a new dictionary before adding them to the final dictionary
    three_vector_words = {}
    three_vector_words['read-for'] = np.add(missing_word_vectors['read'], missing_word_vectors['for'])
    missing_word_vectors_added['read-for-school'] = np.add(three_vector_words['read-for'], missing_word_vectors['school'])/3
    three_vector_words['twenty-first'] = np.add(missing_word_vectors['twenty'], missing_word_vectors['first'])
    missing_word_vectors_added['twenty-first-century'] = np.add(three_vector_words['twenty-first'], missing_word_vectors['century'])/3

    #updating our original dictionary with the word vectors missing from before
    for genres in book_dict:
        # checking if key present in other dictionary
        if genres in missing_word_vectors_added:
            print(genres)
            book_dict[genres]  = missing_word_vectors_added[genres]
        else:
            book_dict[genres] = book_dict[genres]

    #do the rest
    book_dict['self-help'] = book_dict.pop('selfhelp')
    book_dict['high-school'] = book_dict.pop('highschool')
    book_dict['non-fiction'] = book_dict.pop('nonfiction')
    book_dict['coming-of-age'] = book_dict.pop('coming-age')
    book_dict['21st-century'] = book_dict.pop('twenty-first-century')
    book_dict['20th-century'] = book_dict.pop('twentieth-century')

    #first, we need to get the similarity matrix for each genre
    energy_similarity = {}
    for genre in book_dict:
        energy_similarity[genre] = distance.cosine(energy_wv, book_dict[genre])

    #get minimum and maximums of our dataset so we can normalize the data to fit our range of 0-1
    list_of_values = list(energy_similarity.values())
    max_value = max(list_of_values)
    min_value = min(list_of_values)

    #create an empty dataframe to hold the new values in
    df_energy_similarity = pd.DataFrame()
    for key, value in energy_similarity.items():
        value = (value - min_value) / (max_value - min_value)
        df_energy_similarity = df_energy_similarity.append({'Genres': key, 'Similarity': value}, ignore_index=True)

    df_energy_similarity #taking a look at the dataframe containing the normalized data

    #pull out our preferred book genres and saving them to a list
    book_genres = []
    for genres in book_dataset.loc[book_dataset['title'] == 'persuasion', 'genres']:
        genres = genres.split(", ")
        for g in genres:
            g = g.strip("'")
            if g not in book_genres:
                book_genres.append(g)

    #finding the average energy of the genres listed under persuasion
    book_similarity = df_energy_similarity[df_energy_similarity['Genres'].isin(book_genres)]

    energy = sum(book_similarity['Similarity']) / len(book_similarity['Similarity'])
    
    return energy

def calculate_instrumentalness(split_text):
    total_words = len(split_text)
    
    #creating buckets
    if 0 <= total_words <= 12500:
        instrumentalness = 0
    elif 12500 < total_words <= 25000:
        instrumentalness = (total_words/2500) * 1/11
    elif 25000 < total_words <= 37500:
        instrumentalness = (total_words/37500) * 2/11
    elif 37500 < total_words <= 50000:
        instrumentalness = (total_words/50000) * 3/11
    elif 50000 < total_words <= 62500:
        instrumentalness = (total_words/62500) * 4/11
    elif 62500 < total_words <= 75000:
        instrumentalness = (total_words/7500) * 5/11
    elif 75000 < total_words <= 87500:
        instrumentalness = (total_words/87500) * 6/11
    elif 87500 < total_words <= 100000:
        instrumentalness = (total_words/100000) * 7/11
    elif 100000 < total_words <= 112500:
        instrumentalness = (total_words/112500) * 8/11
    elif 112500 < total_words <= 125000:
        instrumentalness = (total_words/12500) * 9/11
    elif 125000 < total_words <= 137500:
        instrumentalness = (total_words/137500) * 10/11
    else:
        instrumentalness = 1
    
    return instrumentalness

def authenticate_spotify(client_id, client_secret):
    auth_manager = SpotifyClientCredentials(client_id, client_secret)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    return sp

def get_playlist_data(playlist_id, limit=100, trim=True):
    sp = authenticate_spotify(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET)
    playlist = sp.playlist(playlist_id)
    tracks = playlist["tracks"]["items"]
    
    if trim:
        tracks = tracks[-limit:]
    
    ids = [track["track"]["id"] for track in tracks]
    features = pd.DataFrame(sp.audio_features(ids))
    
    labels = [track["track"]["artists"][0]["name"] + " - " + track["track"]["name"] for track in tracks]
    features.index = labels
    
    reduced_features = ['energy', 'valence', 'speechiness', 'instrumentalness']
    features = features[reduced_features]
    
    return features
