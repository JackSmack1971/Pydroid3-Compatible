import requests
from bs4 import BeautifulSoup
import nltk
from textblob import TextBlob
from gensim import corpora, models
import pandas as pd
import os

# Check and download necessary NLTK data
def ensure_nltk_data():
    necessary_data = ['punkt', 'stopwords', 'brown']
    for dataset in necessary_data:
        if not nltk.data.find(f'tokenizers/{dataset}'):
            print(f"Downloading NLTK data: {dataset}...")
            nltk.download(dataset, quiet=True)

# Fetch page content with refined error handling
def fetch_page_content(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.HTTPError as e:
        print(f"HTTP Error: {e}")
    except requests.RequestException as e:
        print(f"Request Error: {e}")
    return None

# Extract text from HTML
def extract_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text()

# Preprocess text by filtering stopwords efficiently
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text.lower())
    return [word for word in words if word.isalpha() and word not in stop_words]

# Extract entities using TextBlob
def extract_entities(text):
    blob = TextBlob(text)
    return blob.noun_phrases

# Build LSI model and dictionary
def build_lsi_model(texts):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lsi_model = models.LsiModel(corpus, id2word=dictionary, num_topics=5)  # Optimize number of topics
    return lsi_model, dictionary

# Create DataFrame with improved logic for equal length lists
def create_data_frame(entities, lsi_keywords):
    max_len = max(len(entities), len(lsi_keywords))
    data = {
        "Entities": entities + [""] * (max_len - len(entities)),
        "LSI Keywords": lsi_keywords + [""] * (max_len - len(lsi_keywords))
    }
    return pd.DataFrame(data)

def main(url):
    ensure_nltk_data()

    html_content = fetch_page_content(url)
    if html_content:
        text = extract_text(html_content)
        preprocessed_text = preprocess_text(text)
        entities = extract_entities(text)
        
        lsi_model, _ = build_lsi_model([preprocessed_text])
        
        topics = lsi_model.show_topics(num_topics=5, formatted=False)
        lsi_keywords = [word for topic in topics for word, _ in topic[1]]
        
        df = create_data_frame(list(entities), lsi_keywords)
        df.to_excel("lsi_keywords_and_entities.xlsx", index=False)
        print("Analysis complete. Results saved to 'lsi_keywords_and_entities.xlsx'.")
    else:
        print("Failed to fetch content. Exiting...")

if __name__ == "__main__":
    url = "https://example.com"
    main(url)
