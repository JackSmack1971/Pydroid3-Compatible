import requests
from bs4 import BeautifulSoup
import nltk
from textblob import TextBlob
from gensim import corpora, models
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

# Ensure necessary NLTK data is available
def ensure_nltk_data():
    necessary_data = ['punkt', 'stopwords', 'brown', 'averaged_perceptron_tagger', 'wordnet']
    for dataset in necessary_data:
        try:
            nltk.data.find(f'tokenizers/{dataset}')
        except LookupError:
            nltk.download(dataset, quiet=True)

# Map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wn.ADJ, "N": wn.NOUN, "V": wn.VERB, "R": wn.ADV}
    return tag_dict.get(tag, wn.NOUN)

# Fetch page content with robust error handling
def fetch_page_content(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except (requests.HTTPError, requests.RequestException) as e:
        print(f"Error fetching page content: {e}")
    return None

# Extract text content from HTML
def extract_text(html_content):
    if html_content and html_content.strip():
        return BeautifulSoup(html_content, 'html.parser').get_text()
    return "No content found."

# Preprocess text, including lemmatization
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = nltk.word_tokenize(text.lower())
    return [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words if word.isalpha() and word not in stop_words]

# Extract unique entities
def extract_entities(text):
    return list(set(TextBlob(text).noun_phrases))

# Build and return an LSI model and its dictionary
def build_lsi_model(texts):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    num_topics = max(5, len(dictionary.token2id) // 2)  # Dynamic number of topics
    lsi_model = models.LsiModel(corpus, id2word=dictionary, num_topics=num_topics)
    return lsi_model, dictionary

# Create a DataFrame for entities and LSI keywords
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
        topics = lsi_model.show_topics(num_topics=-1, formatted=False)  # Display all topics
        lsi_keywords = [word for _, words in topics for word, _ in words]
        
        df = create_data_frame(entities, lsi_keywords)
        df.to_excel("lsi_keywords_and_entities.xlsx", index=False)
        print("Analysis complete. Results saved to 'lsi_keywords_and_entities.xlsx'.")
    else:
        print("Failed to fetch content. Exiting...")

if __name__ == "__main__":
    url = "https://example.com"
    main(url)
