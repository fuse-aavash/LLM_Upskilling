# BERT

**BERT** (Bidirectional Encoder Representation From Transformer) is a transformers model pre-trained on a large corpus of English data in a self-supervised fashion. 

BERT makes use of Transformer, an attention mechanism that learns contextual relations between words (or sub-words) in a text. Transformer includes two separate mechanisms — an encoder that reads the text input and a decoder that produces a prediction for the task.

BERT only uses the encoder as its goal is to generate a language model.

**BERT works in two steps:**

1. **Pre-training:** In the pretraining phase, the model is trained on unlabeled data over different pre-training tasks to learn a language representation. Hence in the pre-training phase, we will train a model using a combination of **Masked Language Modelling (MSM)** and **Next Sentence Prediction (NSP)** on a large corpus.
    1. **Masked Language Modeling (MLM)** involves training the BERT model on a large corpus of text  by masking certain words in the input text and training the model to predict the masked words based on the context of the surrounding words. 
    2. **Next sentence prediction (NSP)**: The model is trained to predict whether two sentences follow each other in the original text or not. 
2. **Fine-tuning:** In Fine-tuning phase, we will use learned weights from pre-trained models in a supervised fashion using a small number of labeled textual datasets.

### Example: **Spam Classification**

The most straight-forward way to use BERT is to use it to classify a single piece of text.

![http://jalammar.github.io/images/BERT-classification-spam.png](http://jalammar.github.io/images/BERT-classification-spam.png)

To train such a model, you mainly have to train the classifier, with minimal changes happening to the BERT model during the training phase. This training process is called Fine-Tuning.

# Model Architecture

The paper presents two model sizes for BERT:

- BERT BASE – Comparable in size to the OpenAI Transformer in order to compare performance
- BERT LARGE – A ridiculously huge model which achieved the state of the art results reported in the paper

![https://miro.medium.com/v2/resize:fit:720/format:webp/0*ws9kRab0kLculhYx.png](https://miro.medium.com/v2/resize:fit:720/format:webp/0*ws9kRab0kLculhYx.png)

BERT is a Transformer encoder stack.

## Model Inputs

Because BERT is a pretrained model that expects input data in a specific format, we will need:

1. A **special token, `[SEP]`,** to mark the end of a sentence, or the separation between two sentences
2. A **special token, `[CLS]`,** at the beginning of our text. This token is used for classification tasks, but BERT expects it no matter what your application is.
3. Tokens that conform with the fixed vocabulary used in BERT
4. The **Token IDs** for the tokens, from BERT’s tokenizer
5. **Mask IDs** to indicate which elements in the sequence are tokens and which are padding elements
6. **Segment IDs** used to distinguish different sentences
7. **Positional Embeddings** used to show token position within the sequence

### ****Special Tokens****

- BERT can take as input either one or two sentences, and uses the special token `[SEP]` to differentiate them.
- The `[CLS]` token always appears at the start of the text, and is specific to classification tasks.

Both tokens are *always required,* even if we only have one sentence, even if we are not using BERT for classification.

****************Example:****************

`[CLS] The man went to the store. [SEP] He bought a gallon of milk.`

### ****Tokenization****

BERT uses WordPiece Tokenization. This model greedily creates a fixed-size vocabulary of individual characters, subwords, and words that best fits our language data. The generated vocabulary contains:

1. Whole words
2. Subwords occuring at the front of a word or in isolation (“em” as in “embeddings” is assigned the same vector as the standalone sequence of characters “em” as in “go get em” )
3. Subwords not at the front of a word, which are preceded by ‘##’ to denote this case
4. Individual characters

To tokenize a word under this model:

- tokenizer first checks if the whole word is in the vocabulary.
- If not, it tries to break the word into the largest possible subwords contained in the vocabulary, and as a last resort will decompose the word into individual characters.

because of this, we can always represent a word as, at the very least, the collection of its individual characters. As a result, rather than assigning out of vocabulary words to a catch-all token like ‘OOV’ or ‘UNK,’ words that are not in the vocabulary are decomposed into subword and character tokens that we can then generate embeddings for.

### **Formalizing Input**

Each input token to the BERT is the sum of Token embeddings, Segment embeddings, and Position embeddings.

- **Position Embeddings:** Similar to the transformer, we will feed all the word sequences in the input sentence at once to the BERT model. So to identify the position of words in an input sentence i.e. where each word is located in the input sentence, we will generate position embeddings.
- **Segment Embeddings:** Segment embeddings help to indicate whether a sentence is first or second. Segment Embeddings is important because we also accomplish the `Next Sentence Prediction` task in BERT pretraining phase. So, if you want to process two sentences, assign each word in the first sentence plus the `[SEP]` token a series of 0’s, and all tokens of the second sentence a series of 1’s.
- **Token Embeddings:** Initial low-level embedding of tokens. Initial low-level embedding is essential because the machine learning model doesn’t understand textual data.

# ****BERT: Pretraining****

### ****Masked Language Modeling (MLM):****

In masked language modeling, the model randomly chooses 15% of the words in the input sentence and among those randomly chosen words, masks them 80% of the time (i.e. using **`[MASK]`** token from vocabulary), replace them with a random token 10% of the time, or keep as is 10% of the time and the model has to predict the masked words in the output.

![http://jalammar.github.io/images/BERT-language-modeling-masked-lm.png](http://jalammar.github.io/images/BERT-language-modeling-masked-lm.png)

### ****Next Sentence Prediction (NSP):****

In the Next Sentence Prediction task, Given two input sentences, the model is then trained to recognize if the second sentence follows the first one or not. helps the BERT model handle relationships between multiple sentences which is crucial for a downstream task like Q/A and ***Natural language inference***.

![http://jalammar.github.io/images/bert-next-sentence-prediction.png](http://jalammar.github.io/images/bert-next-sentence-prediction.png)

# BERT: Fine-tuning

Fine-tuning in BERT involves using pre-learned weights from the initial training phase in downstream tasks with minimal adjustments to the architecture. This allows for inexpensive training relative to the pre-training phase.

![https://miro.medium.com/v2/resize:fit:640/format:webp/0*S2YP63OvbKHI4CzT.png](https://miro.medium.com/v2/resize:fit:640/format:webp/0*S2YP63OvbKHI4CzT.png)

# References

- [Google BERT](https://medium.com/@thapaliyanish123/google-bert-8e990b64f570)
- [Illustrated BERT](http://jalammar.github.io/illustrated-bert/)
- [BERT Fine-Tuning Tutorial with PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)