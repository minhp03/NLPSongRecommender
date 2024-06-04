Song Recommender System
Project Description

This song recommencer system is designed to analyze metadata from a dataset of digital music in the metal genre. It utilizes natural language processing (NLP) techniques to provide song recommendations based on similarity in song descriptions and branding. Recommendations are output as a CSV file, making it easy to share and analyze the results.
Features

    Data Source: Uses metadata from the meta_Digital_Music.json file focusing on metal music.
    Data Cleaning and Normalization: Processes and cleans metadata for accuracy and better vectorization.
    Text Preprocessing: Implements text normalization, removal of special characters, and elimination of stop words.
    TF-IDF Vectorization: Converts text data into a format suitable for similarity computation using TF-IDF.
    Cosine Similarity Scoring: Computes similarity between songs to identify and recommend similar tracks.
    CSV Output: Outputs the recommendations to a song_recommendations.csv file for each input song.

library Used

    Python 3
    Pandas for robust data manipulation
    NLTK for comprehensive natural language processing
    Scikit-learn for machine learning tools and vectorization

Usage Instructions

To generate song recommendations:

    Ensure the meta_Digital_Music.json is placed in the correct directory as specified in the script.
    Run EX2.py and follow the on-screen prompts to enter song titles for recommendations.
    Recommendations for each song title are saved in song_recommendations.csv.
