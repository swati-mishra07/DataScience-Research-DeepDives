# Attention Is All You Need

**Paper:** *Attention Is All You Need*  
**Authors:** Ashish Vaswani et al.  
**Published:** NeurIPS 2017  
**Area:** Deep Learning, Natural Language Processing, Transformer Architecture

**Original Paper:** [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

## 📄 About the Paper

This paper introduces the Transformer, a neural network architecture that relies on attention instead of recurrence or convolution to process sequences.

Before reading this paper, I knew that Transformers were used in modern language models, but I wanted to understand the basic idea behind their architecture and why attention is so important.

My goal was to understand how the components work together rather than just memorize the attention equation.

## ❓ What Problem Does It Solve?

Earlier sequence models, such as RNNs and LSTMs, process information sequentially. This makes it difficult to parallelize computations across positions in a sequence during training.

It can also be challenging to model relationships between distant tokens efficiently.

The paper explores a different approach: **allowing tokens to interact through attention without relying on recurrence**.

The authors evaluate the Transformer mainly on machine translation tasks.

## 💡 The Main Idea

The Transformer uses attention to determine how information from different tokens contributes to each token's representation.

For example, in the sentence:

> The animal crossed the street because it was tired.

To understand the word *it*, a model may need information about other words in the sentence.

Self-attention allows a token to compare its representation with other tokens and gather information based on the resulting attention weights.

The main equation introduced in the paper is:

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

I understand the basic process as:

**Query → Match with Keys → Calculate Attention Weights → Combine Values**

---

## 🏗️ Transformer Architecture

The original Transformer follows an encoder-decoder architecture. The base configuration in the paper uses six encoder layers and six decoder layers.

```text
Input Tokens
     |
     v
Token Embeddings + Positional Encoding
     |
     v
+---------------------------+
|          ENCODER          |
|                           |
|   Multi-Head Attention    |
|            |              |
|   Feed-Forward Network    |
|                           |
|   Repeated for 6 Layers   |
+---------------------------+
     |
     v
Encoder Representations
     |
     v
+---------------------------+
|          DECODER          |
|                           |
|   Masked Self-Attention   |
|            |              |
|   Encoder-Decoder         |
|   Attention               |
|            |              |
|   Feed-Forward Network    |
|                           |
|   Repeated for 6 Layers   |
+---------------------------+
     |
     v
Linear Layer + Softmax
     |
     v
Output Token Probabilities
```

The encoder builds representations of the input sequence. The decoder uses previously generated output tokens and information from the encoder to generate the output sequence.

I also learned that six layers are a design choice in the original configuration, not a requirement for every Transformer.

---

## 🔍 Understanding Query, Key and Value

This was one of the concepts I needed to correct while explaining the paper.

The input representation \(X\) is transformed into three representations:

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

Here, \(W_Q\), \(W_K\), and \(W_V\) are learned weight matrices.

My current understanding is:

- **Query (Q):** Represents what information a token is looking for.
- **Key (K):** Represents information that can be matched against a query.
- **Value (V):** Contains the information that contributes to the output when its key receives attention.

I initially thought of Q, K, and V as a normal key-value storage system. I now understand them as learned vector representations with different roles in the attention mechanism.

My way of remembering them is:

> **Query asks → Key matches → Value provides information.**

## 📐 How Scaled Dot-Product Attention Works

### 1. Calculate \(QK^T\)

The first step is:

\[
QK^T
\]

The \(T\) means transpose.

This operation calculates dot-product scores between queries and keys. These scores indicate how strongly each query aligns with each key.

If there are \(n\) tokens, the resulting attention score matrix has shape \(n \times n\).

### 2. Scale the Scores

Next, the scores are divided by \(\sqrt{d_k}\):

\[
S=\frac{QK^T}{\sqrt{d_k}}
\]

Here, \(d_k\) is the dimension of the key vectors.

Initially, I thought this division reduced the dimension. I corrected that understanding: **it controls the magnitude of the dot-product scores**.

As the key dimension increases, dot products can become large in magnitude. Very large scores can make softmax extremely peaked, resulting in small gradients for many positions.

Scaling helps control the scores before softmax.

### 3. Apply Softmax

The scaled scores are passed through softmax:

\[
A=\operatorname{softmax}(S)
\]

Softmax converts the scores into normalized attention weights.

For example, the scores

\[
[2,1,0]
\]

produce approximately

\[
[0.665,0.245,0.090]
\]

The weights are non-negative and sum to one. They indicate how much attention is assigned to the corresponding keys.

### 4. Combine the Values

Finally, the attention weights are multiplied by the value matrix:

\[
O=AV
\]

For example, if the attention weights are \([0.1,0.7,0.2]\), the output is a weighted combination:

\[
O=0.1V_1+0.7V_2+0.2V_3
\]

The second value contributes the most in this example because its weight is the largest.

The complete process is:

\[
\boxed{
O=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
}
\]

This is the central equation I wanted to understand instead of simply memorizing it.

---

## 🔀 Multi-Head Attention

The Transformer uses multiple attention heads instead of relying on one attention operation.

\[
\operatorname{MultiHead}(Q,K,V)
=
\operatorname{Concat}
(\text{head}_1,\ldots,\text{head}_h)W^O
\]

Each head has its own learned projections and can learn different relationships or representation subspaces.

The outputs of the heads are concatenated and passed through another learned projection.

I initially thought multiple heads simply meant that the model gets more context. I now understand that the key idea is to let the model learn different attention patterns in parallel.

I still want to investigate what different heads actually learn during training and how their behaviors differ.

---

## 🔗 Three Uses of Attention

The paper describes three important uses of attention in the Transformer.

### 1. Encoder Self-Attention

Queries, keys, and values come from the encoder representations. Each input token can attend to other input tokens.

### 2. Encoder-Decoder Attention

Queries come from the decoder, while keys and values come from the encoder output. This allows the decoder to use information from the input sequence.

### 3. Masked Decoder Self-Attention

The decoder uses a mask to prevent a position from attending to future output positions. This ensures that a token is not allowed to use future target tokens when predicting the next token during training.

---

## 📍 Positional Encoding

Self-attention does not inherently represent the order of tokens. However, word order can change the meaning of a sentence.

For example:

- The dog chased the cat.
- The cat chased the dog.

Therefore, the Transformer needs information about token positions.

The paper adds positional encodings to token embeddings. It uses sine and cosine functions:

\[
PE_{(pos,2i)}
=
\sin
\left(
\frac{pos}{10000^{2i/d_{\text{model}}}}
\right)
\]

\[
PE_{(pos,2i+1)}
=
\cos
\left(
\frac{pos}{10000^{2i/d_{\text{model}}}}
\right)
\]

Here, \(pos\) represents the token position, \(i\) represents the dimension index, and \(d_{\text{model}}\) is the embedding dimension.

My main understanding is that token embeddings represent token information, while positional encodings provide information about where tokens occur in the sequence.

I still want to understand more deeply why sinusoidal functions were chosen and how their mathematical properties help represent positions.

---

## ⚙️ Feed-Forward Network

After attention, each position passes through a feed-forward network:

\[
\operatorname{FFN}(x)
=
\max(0,xW_1+b_1)W_2+b_2
\]

The attention mechanism allows tokens to exchange information. The feed-forward network then applies a learned nonlinear transformation independently at each position.

I initially thought the feed-forward network was simply there to improve the final result. I now understand that it performs a different role from attention.

My current mental model is:

- **Attention:** Which information should this token gather from other tokens?
- **Feed-forward network:** How should the representation at this position be transformed?

---

## ➕ Residual Connections and Layer Normalization

The Transformer uses residual connections around its sublayers.

The basic idea is:

\[
y=x+\operatorname{Sublayer}(x)
\]

This allows the input representation to pass forward alongside the transformation learned by the sublayer.

Layer Normalization is also used around these sublayers to help stabilize activations and optimization.

I understand their general purpose, but I still want to investigate more deeply why residual connections make deep networks easier to optimize and how Layer Normalization contributes to stable training.

---

## 🏋️ Training and Results

The paper discusses training data and batching, hardware and training schedules, the Adam optimizer, and regularization techniques.

Some of my initial explanations were incorrect, so I corrected them while studying:

- **Adam** is an optimization algorithm that uses estimates derived from gradients to update model parameters. It is not an outlier-detection method.
- **Batching** allows multiple training examples to be processed efficiently and helps utilize parallel hardware.
- The paper discusses techniques including **dropout and label smoothing** as part of its training setup.

The authors evaluate the Transformer on machine translation and report results using BLEU scores. They also investigate different model configurations to understand how architectural choices affect performance.

The experiments showed that the Transformer could achieve strong translation results while benefiting from greater parallelization than recurrent approaches.

---

## 🔧 What I Improved While Reading

These are the main corrections I made to my own understanding.

| My initial understanding | What I understand now |
|---|---|
| Q, K, and V are a normal key-value storage system. | They are learned vector representations with different roles. |
| \(QK^T\) is just a vector operation. | It calculates dot-product scores between queries and keys. |
| \(K^T\) means a power of T. | \(T\) means transpose. |
| Dividing by \(\sqrt{d_k}\) reduces dimensions. | It controls the magnitude of attention scores. |
| Softmax simply makes high scores higher. | It converts scores into normalized attention weights. |
| Multiple heads simply mean more context. | They allow different learned attention patterns in parallel. |
| Positional encoding is mainly about sine and cosine. | It provides sequence-position information. |
| The FFN mainly improves the final result. | It transforms each position's representation after attention. |
| Adam is related to detecting outliers. | Adam updates model parameters using gradient information. |
| A Transformer uses an RNN-style hidden state. | It builds contextual representations through attention. |

These corrections helped me move from remembering component names to understanding their individual roles in the architecture.

---

## 🤔 Questions I Still Have

Reading the paper helped me understand the basic architecture, but it also raised questions I want to explore further.

### Attention

- Why are query, key, and value represented by three separate learned transformations?
- How does the model learn which tokens are relevant to one another?
- What do different attention heads learn during training?
- What happens mathematically to the gradients when softmax becomes extremely peaked?

### Architecture

- Why did the original configuration use six encoder layers and six decoder layers?
- How does changing the number of layers affect model capacity and training?
- Why are residual connections important when stacking many layers?
- What does the feed-forward network learn that attention alone does not?

### Positional Encoding

- Why did the authors choose sinusoidal positional encoding?
- How do sine and cosine functions represent different positions?
- What are the limitations of this positional encoding?

### Modern Language Models

- How did the original Transformer evolve into encoder-only models such as BERT and decoder-only models such as GPT?
- Which components of the original architecture are retained in modern language models?
- How does positional information relate to the context window of a modern LLM?

These are questions I want to investigate through further reading rather than assume I already know the answers.

---

## 🧠 What I Learned

The biggest thing I learned is that a Transformer is not just an attention mechanism. It combines attention with positional information, feed-forward networks, residual connections, and normalization.

The attention equation became much clearer when I broke it into individual operations:

\[
Q,K,V
\rightarrow
QK^T
\rightarrow
\frac{QK^T}{\sqrt{d_k}}
\rightarrow
\operatorname{softmax}
\rightarrow
\operatorname{softmax}(\cdot)V
\]

My current way of remembering it is:

> **Query asks → Key matches → Softmax assigns weights → Value provides information.**

I also learned that the Transformer does not process tokens through an RNN-style hidden state passed sequentially from one position to the next. Instead, it builds contextual representations through attention.

I still have questions about why specific design choices work, but I now have a better foundation for reading papers about BERT, GPT, LoRA, and modern language models.

This is a learning-based breakdown of the paper, written in my own words. It is not an implementation or reproduction of the authors' experiments.

---
## 📖 Sections I Read

I went through the main sections of the paper to understand the Transformer architecture, the motivation behind it, and how its components work together.

| Paper Section | What I Focused On |
|---|---|
| **Abstract** | The main idea of the Transformer and its use of attention instead of recurrence and convolution. |
| **1. Introduction** | The limitations of recurrent sequence models and the motivation for a different architecture. |
| **2. Background** | Previous sequence-to-sequence approaches and the motivation for reducing sequential computation. |
| **3. Model Architecture** | The overall encoder-decoder architecture and how its components work together. |
| **3.1 Encoder and Decoder Stacks** | The structure of the encoder and decoder, including the six-layer configuration used in the original model. |
| **3.2 Attention** | Scaled dot-product attention, query, key, value, and multi-head attention. |
| **3.3 Applications of Attention in Our Model** | Encoder self-attention, masked decoder self-attention, and encoder-decoder attention. |
| **3.4 Position-wise Feed-Forward Networks** | Why the Transformer uses feed-forward networks after attention. |
| **3.5 Embeddings and Softmax** | How token embeddings and the output projection are used in the model. |
| **3.6 Positional Encoding** | Why token positions are needed and how sinusoidal encodings represent position. |
| **4. Why Self-Attention** | The comparison between self-attention, recurrent layers, and convolutional layers. |
| **5. Training** | Training data and batching, hardware and schedule, the Adam optimizer, and regularization. |
| **6. Results** | Machine translation results and experiments involving different model configurations. |
| **7. Conclusion** | The authors' main conclusions and the potential of attention-based sequence models. |

### My Reading Approach

I focused on understanding the main ideas and explaining them in my own words. While reading, I also tried to connect the equations to the purpose of each component.

Some concepts, especially the mathematical reasoning behind attention scaling, positional encoding, residual connections, and Layer Normalization, still need further study.

This README documents my current understanding. It does not mean that I have mastered every mathematical detail or reproduced the experiments.

## 📚 Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I.

**Attention Is All You Need.** NeurIPS 2017.

[Read the original paper on arXiv](https://arxiv.org/abs/1706.03762)
