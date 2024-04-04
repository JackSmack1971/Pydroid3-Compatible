# LSI Scraper

## Overview
LSI Scraper is a Python tool designed to fetch and analyze web page content to identify and extract latent semantic indexing (LSI) keywords and entities. It utilizes natural language processing (NLP) techniques and machine learning algorithms to process text data efficiently.

## Features
- Fetches web page content using HTTP requests.
- Extracts textual content from HTML.
- Preprocesses text to remove stopwords and tokenize.
- Identifies entities using TextBlob.
- Extracts LSI keywords using Gensim's LSI model.
- Outputs the results to an Excel file.
- Enhanced error handling for more robust web scraping.
- Optimized text processing for improved efficiency.

## Installation
Ensure you have Python 3.6+ installed. Then, install the required packages:

```bash
pip install requests bs4 nltk textblob gensim pandas
```

Additionally, run the following Python command to download necessary NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('brown')
```

## Usage
Update the `main` function's `url` variable in `LSIScraper.py` to the target web page URL. Then, run the script:

```bash
python LSIScraper.py
```

The analysis results will be saved to `lsi_keywords_and_entities.xlsx`.

## Contributing
Contributions to enhance LSI Scraper are welcome. Please follow the standard fork-and-pull request workflow.

## License
LSI Scraper is released under the MIT License. See the LICENSE file for more details.
