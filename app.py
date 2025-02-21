import requests
from bs4 import BeautifulSoup
import streamlit as st
import pickle
from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
import re
import nltk
import string
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from textblob import TextBlob

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load pre-trained models and vectorizers
sentiment_model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

with open('keyword extraction/count_vectorizer.pkl', 'rb') as file:
    keyword_cv = pickle.load(file, encoding='utf-8')

with open('keyword extraction/feature_names.pkl', 'rb') as file:
    keyword_feature_names = pickle.load(file, encoding='utf-8')

with open('keyword extraction/tfidf_transformer.pkl', 'rb') as file:
    keyword_tfidf_transformer = pickle.load(file, encoding='utf-8')

# Initialize NLP tools
ws = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
punc = string.punctuation

# Text Preprocessing Functions
def remove_url(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text) if isinstance(text, str) else text

def preprocessing(text):
    text = re.sub(r'Read more', '', text)
    text = BeautifulSoup(text, 'html.parser').get_text()
    if not text.strip():
        return 'empty_text'
    text_list = [ws.lemmatize(word.lower().strip()) for word in word_tokenize(text) if word not in stop_words and word not in punc]
    return ' '.join(text_list) if text_list else 'empty_text'

# Fuzzy Matching Features
def fetch_fuzzy_features(row):
    q1, q2 = str(row['char_len']), str(row['word_len'])
    return [fuzz.QRatio(q1, q2), fuzz.partial_ratio(q1, q2), fuzz.token_sort_ratio(q1, q2), fuzz.token_set_ratio(q1, q2)]

# Web Scraping for Amazon Reviews
def scrape_reviews(url, review_type, sort_by, num_reviews):
    reviews, page_number = [], 1
    while len(reviews) < num_reviews:
        response = requests.get(f"{url}?reviewerType={review_type}&sortBy={sort_by}&pageNumber={page_number}",
                                headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            break
        soup = BeautifulSoup(response.text, "html.parser")
        review_elements = soup.find_all("span", {"data-hook": "review-body"})
        reviews.extend([element.get_text().strip() for element in review_elements])
        if len(reviews) >= num_reviews or not soup.find("a", {"data-hook": "see-all-reviews-link-foot"}):
            break
        page_number += 1
    return reviews[:num_reviews]

# Sentiment Analysis Functions
def calculate_overall_sentiment(reviews):
    df = pd.DataFrame({'Review': reviews})
    df['Review'] = df['Review'].apply(preprocessing)
    review_transform = tfidf.transform(df['Review']).toarray()
    df['char_len'] = df['Review'].str.len()
    df['word_len'] = df['Review'].apply(lambda x: len(x.split()))
    fuzzy_features = df.apply(fetch_fuzzy_features, axis=1).tolist()
    df[['fuzz_ratio', 'fuzz_partial_ratio', 'fuzz_token_sort_ratio', 'fuzz_token_set_ratio']] = pd.DataFrame(fuzzy_features)
    final_df = pd.concat([df.drop(columns=['Review']), pd.DataFrame(review_transform)], axis=1)
    predictions = sentiment_model.predict(final_df)
    return np.bincount(predictions).argmax()

def analyze_sentiment(review):
    polarity = TextBlob(review).sentiment.polarity
    return "😊 Satisfied" if polarity > 0.1 else "🙁 Dissatisfied" if polarity < -0.1 else "😐 Neutral"

# Sentiment Graph
def plot_sentiment_graph(sentiment_counts):
    sentiments = ["Satisfied", "Neutral", "Dissatisfied"]
    frequencies = [sentiment_counts.get("😊 Satisfied", 0), 
                   sentiment_counts.get("😐 Neutral", 0), 
                   sentiment_counts.get("🙁 Dissatisfied", 0)]
    colors = ['green', 'orange', 'red']
    
    plt.figure(figsize=(8, 5))
    plt.bar(sentiments, frequencies, color=colors, alpha=0.7)
    plt.xlabel('Sentiment', fontsize=14)
    plt.ylabel('Number of Reviews', fontsize=14)
    plt.title('Sentiment Analysis Results', fontsize=16)
    st.pyplot(plt)

# Streamlit UI
st.title("Amazon Review Sentiment Analyzer")

# User Input Fields
url = st.text_input("Enter the Amazon product review URL:")
review_type = st.selectbox("Review Type:", ("all_reviews", "avp_only_reviews"))
sort_by = st.selectbox("Sort By:", ("helpful", "recent"))
num_reviews = st.number_input("Number of Reviews:", min_value=1, max_value=100, value=10)

# Analyze Button Action
if st.button("Analyze Reviews"):
    if url:
        st.write("Fetching reviews from Amazon...")
        reviews = scrape_reviews(url, review_type, sort_by, num_reviews)
        if reviews:
            st.write(f"Fetched {len(reviews)} reviews.")
            df = pd.DataFrame({'Review': reviews})
            df['Sentiment'] = df['Review'].apply(analyze_sentiment)
            sentiment_counts = Counter(df['Sentiment'])
            plot_sentiment_graph(sentiment_counts)
            st.table(df[['Review', 'Sentiment']])
        else:
            st.write("No reviews found.")
    else:
        st.write("Please enter a valid URL.")
