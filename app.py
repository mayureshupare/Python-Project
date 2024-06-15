import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from string import punctuation
import nltk

# Download NLTK resources (stopwords) if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    import nltk
    nltk.download('stopwords')

from nltk.corpus import stopwords

# Load data
df = pd.read_csv('prog_book.csv')

# Function to clean text
stop = set(stopwords.words('english'))
def clean_text(text):
    text = ''.join([char for char in text if char not in punctuation])
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop])
    return text

# Preprocess text data
df['clean_Book_title'] = df['Book_title'].apply(clean_text)
df['clean_Description'] = df['Description'].apply(clean_text)

# Vectorize book titles
vectorizer = TfidfVectorizer(analyzer='word', lowercase=False)
title_vectors = vectorizer.fit_transform(df['clean_Book_title']).toarray()

st.title('Programming Book Recommendation System')

# Layout for input
col1, col2 = st.columns(2)

# Taking book name as input
with col1:
    book = st.text_input('Enter book name that you liked : ')

# Taking multiple fields to get similarity
with col2:
    feat = st.selectbox("Select Mode : ", ['Book_title', 'Rating', 'Price'])

# Function to get recommendations
def get_recommendations(book_name, feature, df, title_vectors, mode):
    try:
        book_index = df[df['Book_title'].str.contains(book_name, case=False)].index[0]
    except IndexError:
        return pd.DataFrame({"Error": ["Book not found"]})

    if mode == 'Book_title':
        similarities = cosine_similarity([title_vectors[book_index]], title_vectors).flatten()
        similar_indices = similarities.argsort()[-2::-1]  # Exclude the book itself
        similar_books = df.iloc[similar_indices][['Book_title', 'Rating', 'Price']]
    else:
        similar_books = df.sort_values(by=mode, ascending=False).head(10)[['Book_title', 'Rating', 'Price']]
    
    return similar_books

# When the button is clicked, show recommendations
if st.button('Search'):
    if book:
        st.success(f'Recommending books similar to {book}')
        recommendations = get_recommendations(book, 'Book_title', df, title_vectors, feat)
        st.dataframe(recommendations, width=700, height=400)
    else:
        st.error('Please enter a book name.')
