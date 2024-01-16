import spacy

def tokenize_sentences_spacy(text):
    """
    Tokenizes input text into sentences using spaCy.

    Args:
        text (str): The input text.

    Returns:
        list: A list of tokenized sentences.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def tokenize_words_spacy(text):
    """
    Tokenizes input text into words using spaCy.

    Args:
        text (str): The input text.

    Returns:
        list: A list of tokenized words.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [token.text for token in doc]

def lowercase_spacy(text):
    """
    Converts input text to lowercase using spaCy.

    Args:
        text (str): The input text.

    Returns:
        str: The input text in lowercase.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return doc.text.lower()

def remove_stopwords_spacy(text):
    """
    Removes stopwords from input text using spaCy.

    Args:
        text (str): The input text.

    Returns:
        str: The input text with stopwords removed.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return ' '.join([token.text for token in doc if not token.is_stop])

def lemmatize_words_spacy(text):
    """
    Lemmatizes words in the input text using spaCy.

    Args:
        text (str): The input text.

    Returns:
        list: A list of tuples containing the original word and lemma.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [(token.text, token.lemma_) for token in doc]

def named_entity_recognizer_spacy(text):
    """
    Extracts named entities from input text using spaCy.

    Args:
        text (str): The input text.

    Returns:
        list: A list of named entities.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

if __name__ == "__main__":

    text = '''Oil refineries are industrial facilities that play a crucial role in the petroleum industry.
    These complexes are designed to process crude oil into various valuable products, meeting the diverse 
    energy and chemical needs of modern society. The refining process involves a series of complex
    operations that transform crude oil
    into refined products such as gasoline, diesel, jet fuel, and various petrochemicals.
    '''
    # Tokenize sentences using spaCy
    sentences_spacy = tokenize_sentences_spacy(text)
    print("Sentences (spaCy): ")
    print(sentences_spacy)

    # Tokenize words using spaCy
    words_spacy = tokenize_words_spacy(text)
    print("\nWords (spaCy):")
    print(words_spacy)

    # Lowercase using spaCy
    lowercased_text_spacy = lowercase_spacy(text)
    print("\nLowercased Text (spaCy):")
    print(lowercased_text_spacy)

    # Remove stopwords using spaCy
    without_stopwords_spacy = remove_stopwords_spacy(text)
    print("\nText after Stopword Removal (spaCy):")
    print(without_stopwords_spacy)

    # Lemmatization using spaCy
    lemmatized_words_spacy = lemmatize_words_spacy(text)
    print("\nLemmatized Words (spaCy):")
    print(lemmatized_words_spacy)

    # Named Entity Recognition using spaCy
    named_entities_spacy = named_entity_recognizer_spacy(text)
    print("\nNamed Entities (spaCy):")
    print(named_entities_spacy)
