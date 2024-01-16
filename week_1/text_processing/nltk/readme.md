# Text Preprocessing with NLTK

NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries

## Requirements

Before we begin, we need to install the NLTK library.
Install using pip

```bash
pip install nltk
```

Additionally, we also need to download NLTK NLTK data for tokenizers, stopwords, and named entity recognition.

```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")
```

## Usage

### Tokenization

Tokenization is the process of breaking a text into individual sentences or words.

1. **Tokenizing Sentences**
   Tokenizing sentences helps divide a block of text into its constituent sentences, which is essential for many natural language processing tasks. Sentence tokenization typically involves identifying sentence boundaries based on punctuation and capitalization.

   ```python
   from nltk.tokenize import sent_tokenize

    def tokenize_sentences(text):
        return sent_tokenize(text)
   ```

2. **Tokenizing Words**
   Tokenizing words involves splitting a sentence or text into individual words or tokens.

   ```python
   from nltk.tokenize import word_tokenize

   def tokenize_words(text):
   return word_tokenize(text)
   ```

### Lowercasing

Lowercasing converts all text to lowercase.
Lowercasing ensures uniformity and consistency in text data. It prevents the model from treating words in different cases (e.g., "apple" and "Apple") as distinct, which can lead to improved text analysis results.

```python
def lowercase(text):
    return text.lower()
```

### Removing Stopwords

Stopwords are common words (e.g., "the," "and," "is") that are often removed from text because they do not carry significant meaning.
Removing stopwords reduces the dimensionality of text data and eliminates noise. It helps focus on the most meaningful words, improving the efficiency and effectiveness of NLP tasks.

```python
from nltk.corpus import stopwords

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_words = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_words)
```

### Stemming

Stemming is a process of transforming a word to its root form.
Stemming reduces the words "chocolates", "chocolatey", "choco" to the root word, "chocolate", and "retrieval", "retrieved", "retrieves" to the stem "retrieve".

```python
from nltk.stem import PorterStemmer

def stem_words(text):
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    return [(word, stemmer.stem(word)) for word in words]

```

**Errors in Stemming**

- Over stemming
  - When a much larger part of a word is chopped off than what is required, which in turn leads to two or more words being reduced to the same root word or stem incorrectly when they should have been reduced to two or more stem words.
  - Example:
    - University and universe
    - Some stemming algorithm may reduce both the words to the stem univers, which would imply both the words mean the same thing, and that is clearly wrong.
- Under Stemming
  - When two or more words could be wrongly reduced to more than one root word, when they actually should be reduced to the same root word.
  - Example:
    - consider the words "data" and "datum."
    - Some algorithms may reduce these words to dat and datu respectively, which is obviously wrong.
    - Both of these have to be reduced to the same stem dat.

### Lemmatization

Lemmatization is the process of reducing words to their dictionary or base form (lemma). Lemmatization is more linguistically informed compared to stemming. Lemmatization relies on accurately determining the intended part-of-speech and the meaning of a word based on its context. We also need find out the correct POS tag for each word, map it to the right input character that the WordnetLemmatizer accepts and pass it as the second argument to lemmatize()

```python
from nltk.stem import WordNetLemmatizer

def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    token_lemma_list = [(token, lemmatizer.lemmatize(token, get_wordnet_pos(token))) for token in tokens]
    return token_lemma_list

```

### Named Entity Recognition

Named Entity Recognition (NER) identifies and categorizes named entities (e.g., names of people, places, organizations) in text.
NER is essential for information extraction and understanding the structure of textual information. It helps identify and classify entities, allowing NLP models to recognize and work with specific pieces of information within text, such as dates, locations, and names.

```python
from nltk import ne_chunk

def named_entity_recognizer(text):
    words = word_tokenize(text)
    tagged = nltk.pos_tag(words)
    named_entities = ne_chunk(tagged)
    return named_entities
```

## References

- [NLP using NLTK](https://github.com/thapaliya123/nlp-using-nltk/tree/master)
- [Natural Language Processing with Python](https://www.nltk.org/book/)
- [NLTK: A Beginners Hands-on Guide to Natural Language Processing](https://www.analyticsvidhya.com/blog/2021/07/nltk-a-beginners-hands-on-guide-to-natural-language-processing/)
