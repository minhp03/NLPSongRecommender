import os, re, string, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
data_loc = "D:\Centen- NLP COMP 262\Assignment 3\ex2\meta_Digital_Music.json"
songs_minh = pd.read_json(data_loc, dtype=str, lines=True)
#exploration
print(songs_minh.info())

#check null
for column in songs_minh.columns:
    num_nulls = songs_minh[column].isnull().sum()
    
    #empty
    #empty_check = []
    if songs_minh[column].dtype == 'object':
        #songs_minh[column] = empty_check
        empty_check = (songs_minh[column] == '').sum()

    print(f"Column '{column}' has {empty_check} empty strings.")
    print("'{column}' has {empty_check} null .")

songs_minh['category']

drop_columns = ['category', 'tech1', 'tech2', 
                'fit', 'feature', 'main_cat', 
                'similar_item', 'date', 'imageURL', 'imageURLHighRes']

songs_minh.drop(columns=drop_columns, inplace=True)

#also view
#also buy
#filtering out if also buy and also view are empty
songs_minh = songs_minh[songs_minh['also_buy'].apply(lambda x: len(x) > 0)]
songs_minh = songs_minh[songs_minh['also_view'].apply(lambda x: len(x) > 0)]

songs_minh.reset_index(drop=True, inplace=True)

#3
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def pre_process(text):
    
    
    text = re.sub(r'RT|@[^\s]+|[^\s]*…|\n+|\t+', '', text)
   
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    return text.strip()

#check for non words ( example emoji)
def remove_emojis_and_non_ascii(text):
    ascii_only = re.sub(r'[^\x00-\x7F]', '', text)
    return ascii_only

songs_minh.columns

songs_minh['title'] = songs_minh['title'].apply(pre_process)
songs_minh['brand'] = songs_minh['brand'].apply(pre_process)
songs_minh['description'] = songs_minh['description'].apply(pre_process)

#
songs_minh = songs_minh.drop(songs_minh[songs_minh['description'].isnull()].index).drop_duplicates(subset=['description']).reset_index(drop=True)
songs_minh = songs_minh.drop(songs_minh[songs_minh['brand'].isnull()].index).drop_duplicates(subset=['brand']).reset_index(drop=True)

#combine because title + description could go along with each other
songs_minh['combining'] = songs_minh['brand'] + " " + songs_minh['description']

print(songs_minh['combining'].head(10))
print(songs_minh['combining'].head(10))

#TF IDF
tfidf_minh = TfidfVectorizer(max_features=20000, min_df=2, stop_words='english')
tfidf_matrix = tfidf_minh.fit_transform(songs_minh['combining'])


#Compute Pairwise Cosine Similarity Score
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(cosine_sim)
print(songs_minh['combining'].head(10))

def get_recommendations(song_index, cosine_sim=cosine_sim, num_recommendations=10):
    sim_scores = list(enumerate(cosine_sim[song_index]))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:num_recommendations+1]  
    
    song_indices = [i[0] for i in sim_scores]
    
    return songs_minh.iloc[song_indices]

recommendations = get_recommendations(0)
print(recommendations[['title', 'brand', 'description']])

with open('song_recommendations.csv', 'w') as f:
    for index, row in songs_minh.iterrows():
        recommended_indices = get_recommendations(index)['title'].values
        f.write(f"{row['title']}, {'; '.join(recommended_indices)}\n")



def find_song_index(title, df=songs_minh):
 
    results = df[df['title'].str.contains(title, case=False, na=False)]
    if not results.empty:
        return results.iloc[0].name  
    else:
        return None

def recommender():
    while True:
        user_input = input("Enter a song title to get recommendations (or type 'exit' to quit): ").strip()
        
        if user_input.lower() == 'exit':
            print("Exiting the recommender system. Goodbye!")
            break
        
        song_index = find_song_index(user_input)
        
        if song_index is None:
            print(f"We don't have recommendations for '{user_input}'. Please try a different song title.")
            continue
        else:
            recommendations = get_recommendations(song_index, cosine_sim)
            print("Top recommendations:")
            for index, row in recommendations.iterrows():
                print(f"- {row['title']}")
            print("\n")

recommender()
