# Transformers

**The Transformer** – a model that uses attention to boost the speed with which these models can be trained. The transformer model is based solely on self-attention mechanism without any recurrent or convolutional structures. Introduced in the 2017 paper titled "Attention is All You Need" by a team of researchers at Google. The key innovation of the Transformer is its attention mechanism, which allows the model to weigh the importance of different words in a sentence when processing it.

# High Level Look

![https://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png](https://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png)

The Transformer architecture uses an encoder-decoder structure. Both the encoder and decoder components are a stack of encoders and decoders.

**Encoders**

All encoders are identical in structure though they don’t share the weights. Each of the encoders is broken down into two sub layers:

- an self attention layer
- and a feed forward neural network

The input to the encoder first goes through a self attention layer.

- attention layer helps the encoder to look at other words in the input sentence as it encodes a specific word.

The outputs of the self-attention layer are fed to a feed-forward neural network.

![https://jalammar.github.io/images/t/Transformer_decoder.png](https://jalammar.github.io/images/t/Transformer_decoder.png)

**Decoders**

The decoder has both the layers of encoders and between them another attention layer that helps the decoder focus on relevant parts of the input sentence.

# Embeddings

Like other NLP applications, we begin by turning each input word into a vector using an [embedding algorithm](https://medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca). The embedding happens only in the the bottom most encoder.

The encoders receive a list of vectors each of size 512.

- The bottom encoder receives word embeddings.
- Other encoders receives the output of the encoder that’s directly below.

After embedding the words in our input sequence, each of them flows through each of the two layers of the encoder.

![https://jalammar.github.io/images/t/encoder_with_tensors.png](https://jalammar.github.io/images/t/encoder_with_tensors.png)

The word in each position flows through its own path in the encoder. There are dependencies between these paths in the self-attention layer but not in feed-forward layer.

Thus, various paths can be executed in parallel while flowing through the feed-forward layer.

# Encoder

As mentioned above, encoder receives a list of vectors as input. Encoder processes this list of vectors by passing them

- into a self-attention layer
- then into a feed-forward neural network
- then sends out the output upwards to the next encoder.

![https://jalammar.github.io/images/t/encoder_with_tensors_2.png](https://jalammar.github.io/images/t/encoder_with_tensors_2.png)

## Self-Attention

Let’s take an input sentence for translation:

”`The animal didn't cross the street because it was too tired`”

When the model is processing the word “it”, self-attention allows it to associate “it” with “animal”.

As model processes each word (each position in the input sequence),

- self attention allows it to look at other positions in the input sequence for clues that can help to a better encoding for this word

### How to calculate:

**Query, Key, Value Vectors**

- Create three vectors from each of the encoder’s input vectors. We call them, Query vector, Key vector, and Value Vector.
- They are created by multiplying the embeddings by three matrices that we train during training process.
- The new vectors are smaller in dimension than the embedding vectors.

![https://jalammar.github.io/images/t/transformer_self_attention_vectors.png](https://jalammar.github.io/images/t/transformer_self_attention_vectors.png)

****\*\*****Score****\*\*****

- If we are calculating for the word ‘thinking’ we need to score each word against this word.
- The score determines how much focus to place on other parts of input sentence as we encode a word at a certain position.
  - Calculate dot product of the query vector with the key vector of the respective word we’re scoring.
    - if we’re processing the self-attention for the word in position #1, the first score would be the dot product of q1 and k1. The second score would be the dot product of q1 and k2.
  ![https://jalammar.github.io/images/t/transformer_self_attention_score.png](https://jalammar.github.io/images/t/transformer_self_attention_score.png)
- Now divide the scores by the square root of the dimension of the key vectors (8).
  - For stable gradients.
- Then, pass the result through a softmax operation.
  - Softmax normalizes the scores so they’re all positive and add up to 1.
  ![https://jalammar.github.io/images/t/self-attention_softmax.png](https://jalammar.github.io/images/t/self-attention_softmax.png)
- Multiply each value vector by the softmax score.
  - to keep intact the values of the word(s) we want to focus on, and drown-out irrelevant words
- Sum up the weighted value vectors. This produces the output of the self-attention layer at this position (for the first word).
  ![https://jalammar.github.io/images/t/self-attention-output.png](https://jalammar.github.io/images/t/self-attention-output.png)

![https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png](https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png)

## Multi-Headed Attention.

With multi-headed attention, we have multiple sets of Query/Key/Value weight matrices. Each of these sets is randomly initialized. Then, after training, each set is used to project the input embeddings into a different representation subspace.

This improves the performance of the attention layer in two ways:

1. It expands the model’s ability to focus on different positions.
2. It gives the attention layer multiple “representation subspaces”.

![With multi-headed attention, we maintain separate Q/K/V weight matrices for each head resulting in different Q/K/V matrices](https://jalammar.github.io/images/t/transformer_attention_heads_qkv.png)

With multi-headed attention, we maintain separate Q/K/V weight matrices for each head resulting in different Q/K/V matrices

Doing the above outlined calculation for self-attention, we would end up with eight different z matrices. But the feed-forward layer is not expecting eight matrices.

So, we concat the matrices then multiply them by an additional weights matrix WO.

![https://jalammar.github.io/images/t/transformer_attention_heads_weight_matrix_o.png](https://jalammar.github.io/images/t/transformer_attention_heads_weight_matrix_o.png)

![https://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png](https://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)

## Positional Encoding

- a way to account for the order of the words in the input sequence.
- the transformer adds a vector to each input embedding.

These vectors follow a specific pattern that the model learns, which helps it determine the position of each word, or the distance between different words in the sequence.

these values provides meaningful distances between the embedding vectors once they’re projected into Q/K/V vectors and during dot-product attention.

![https://jalammar.github.io/images/t/transformer_positional_encoding_example.png](https://jalammar.github.io/images/t/transformer_positional_encoding_example.png)

## Residual connection and normalization layer

Each sub-layer in each encoder has a residual connection around it, and is followed by a layer-normalization step.

![https://jalammar.github.io/images/t/transformer_resideual_layer_norm.png](https://jalammar.github.io/images/t/transformer_resideual_layer_norm.png)

This goes for the sub-layers of the decoder as well. If we’re to think of a Transformer of 2 stacked encoders and decoders, it would look something like this:

![https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png](https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png)

# Decoder side

Most of the concept is the same as in the encoder side. Let’s take a look at how they work together.

The encoder start by processing the input sequence. The output of the top encoder is then transformed into a set of attention vectors K and V. These are used by each decoder in its “encoder-decoder attention” layer which helps the decoder focus on appropriate places in the input sequence.

![https://jalammar.github.io/images/t/transformer_decoding_1.gif](https://jalammar.github.io/images/t/transformer_decoding_1.gif)

The following steps repeat the process until a special symbol is reached which indicates the transformer has completed its output.

- The output of each step is fed to the bottom decoder in the next time step
- We also embed and add positional encoding to those decoder inputs to indicate the position of each word.

![https://jalammar.github.io/images/t/transformer_decoding_2.gif](https://jalammar.github.io/images/t/transformer_decoding_2.gif)

In the decoder, the self-attention layer is only allowed to attend to earlier positions in the output sequence.

- This is done by masking future positions before the softmax step in the self-attention calculation.

The “Encoder-Decoder Attention” layer works just like multiheaded self-attention, except

- it creates its Queries matrix from the layer below it, and takes the Keys and Values matrix from the output of the encoder stack.

# Final Linear and Softmax Layer

- The decoder stack outputs a vector of floats.
- The Linear layer projects the vector produced by the stack of decoders, into a much, much larger vector called a logits vector.
- The softmax layer then turns those scores into probabilities.
- The cell with the highest probability is chosen, and the word associated with it is produced as the output for this time step.

# References

- [The Illustrated Transformer - Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)
