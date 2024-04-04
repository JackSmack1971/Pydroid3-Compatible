import requests
from bs4 import BeautifulSoup
import nltk
from textblob import TextBlob
from gensim import corpora, models
import pandas as pd

# Streamlined check and download for necessary NLTK data
def ensure_nltk_data():
    necessary_data = ['punkt', 'stopwords', 'brown']
    for dataset in necessary_data:
        try:
            nltk.data.find(f'tokenizers/{dataset}')
        except LookupError:
            print(f"Downloading NLTK data: {dataset}...")
            nltk.download(dataset, quiet=True)

# Consolidated error handling in content fetching
def fetch_page_content(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except (requests.HTTPError, requests.RequestException) as e:
        print(f"Error fetching page content: {e}")
    return None

# Optimized text extraction from HTML
def extract_text(html_content):
    return BeautifulSoup(html_content, 'html.parser').get_text()

# Optimized text preprocessing to remove stopwords and tokenize
def preprocess_text(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    return [word for word in nltk.word_tokenize(text.lower()) if word.isalpha() and word not in stop_words]

# Improved entity extraction to minimize redundant processing
def extract_entities(text):
    return list(set(TextBlob(text).noun_phrases))

# Optimized LSI model building and dictionary creation
def build_lsi_model(texts):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lsi_model = models.LsiModel(corpus, id2word=dictionary, num_topics=5)
    return lsi_model, dictionary

# Refined DataFrame creation to handle entity and keyword alignment
def create_data_frame(entities, lsi_keywords):
    max_len = max(len(entities), len(lsi_keywords))
    padded_entities = entities + [""] * (max_len - len(entities))
    padded_keywords = lsi_keywords + [""] * (max_len - len(lsi_keywords))
    return pd.DataFrame({"Entities": padded_entities, "LSI Keywords": padded_keywords})

def main(url):
    ensure_nltk_data()

    html_content = fetch_page_content(url)
    if html_content:
        text = extract_text(html_content)
        preprocessed_text = preprocess_text(text)
        entities = extract_entities(' '.join(preprocessed_text))
        
        lsi_model, _ = build_lsi_model([preprocessed_text])
        topics = lsi_model.show_topics(num_topics=5, formatted=False)
        lsi_keywords = [word for _, words in topics for word, _ in words]
        
        df = create_data_frame(entities, lsi_keywords)
        df.to_excel("lsi_keywords_and_entities.xlsx", index=False)
        print("Analysis complete. Results saved to 'lsi_keywords_and_entities.xlsx'.")
    else:
        print("Failed to fetch content. Exiting...")

if __name__ == "__main__":
    url = "https://example.com"
    main(url)
