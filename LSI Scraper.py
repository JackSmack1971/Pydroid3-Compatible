import requests
from bs4 import BeautifulSoup
import nltk
from textblob import TextBlob
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import os

def ensure_nltk_data():
    print("Checking and downloading necessary NLTK data...")
    necessary_data = ['punkt', 'stopwords', 'brown']
    for dataset in necessary_data:
        nltk.download(dataset, quiet=True)
    print("NLTK data is ready.")

def fetch_page_content(url):
    try:
        print(f"Fetching content from {url}...")
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        print("Content fetched successfully.")
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage content: {e}")
        return None

def extract_text(html_content):
    if html_content:
        print("Extracting text from HTML...")
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text()
        print("Text extraction complete.")
        return text
    return ""

def preprocess_text(text):
    print("Preprocessing text...")
    stop_words = set(word.lower() for word in stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    print("Text preprocessing complete.")
    return filtered_words

def extract_entities(text):
    print("Extracting noun phrases (entities)...")
    blob = TextBlob(text)
    entities = blob.noun_phrases
    print("Entity extraction complete.")
    return entities

def build_lsi_model(texts):
    print("Building LSI model...")
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lsi_model = models.LsiModel(corpus, id2word=dictionary, num_topics=10)
    print("LSI model built successfully.")
    return lsi_model, dictionary

def create_data_frame(entities, lsi_topics):
    print("Creating DataFrame...")
    data = {"Entities": entities, "LSI Keywords": lsi_topics}
    df = pd.DataFrame(data)
    print("DataFrame created successfully.")
    return df

def main(url):
    ensure_nltk_data()
    print("Current Working Directory:", os.getcwd())

    html_content = fetch_page_content(url)
    if html_content:
        text = extract_text(html_content)
        preprocessed_text = preprocess_text(text)
        entities = extract_entities(text)
        
        lsi_model, _ = build_lsi_model([preprocessed_text])
        
        # Extract keywords from LSI topics
        topics = lsi_model.show_topics(num_topics=10, formatted=False)
        lsi_keywords = [word for topic in topics for word, _ in topic[1]]
        
        # Ensure both lists have the same length
        max_len = max(len(entities), len(lsi_keywords))
        entities.extend([""] * (max_len - len(entities)))
        lsi_keywords.extend([""] * (max_len - len(lsi_keywords)))
        
        # Create DataFrame
        df = create_data_frame(entities, lsi_keywords)
        df.to_excel("lsi_keywords_and_entities.xlsx", index=False)
        print("Analysis complete. Results saved to 'lsi_keywords_and_entities.xlsx'.")
    else:
        print("Failed to fetch content. Exiting...")

if __name__ == "__main__":
    url = "https://example.com"  # Placeholder URL
    main(url)
