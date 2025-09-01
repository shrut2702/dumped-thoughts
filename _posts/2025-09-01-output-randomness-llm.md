---
layout: post
title: "Understanding Temperature and Top-K in Large Language Models"
date: 2025-09-01
excerpt: "First attempt at blogging my learnings and projects."
---

What if every time we query the LLM a prompt, it responds with the same output? It would sound mundane and redundant. It would always generate same set of tokens, the tokens which has highest probability computed by the model. The task of a transformer-based LLM is to generate probability distribution for the vocabulary (set of all tokens/words known to a LLM) based on input sequence, from which it will predict next-word that is most related to input sequence or has the highest probability. This means for same input sequence, the LLM will generate exact same next-word.

So, now the question is how can we get more creative, unique but still coherent and correct output. In order to get control over the output randomness of an LLM there are temperature and top-k parameters.

Before we dive into understanding how these parameters work, let us explore the output of an LLM. The output of an LLM is a 3d tensor with shape (batch_size, num_tokens, vocab_size).

- batch_size: number of input sequences in a batch
- num_tokens: number of tokens in each input sequence
- vocab_size: vocabulary size (total distinct tokens from which model will generate next token)

For each token in the input sequence, the LLM will predict the next token. Since we want the LLM to generate coherent text based on whole input sequence, we will only focus on last logits which corresponds to last input token's output.

```example
e.g.
Input text: "the weather is good"
vocab = ['the', 'weather', 'is', 'good', 'today', 'tomorrow', 'never']

here, batch_size = 1
      num_tokens = 4
      vocab_size = 7 (for example purpose)

input: tensor([[0, 1, 2, 3]])
# shape (batch_size, num_tokens)

output: tensor([[[-1.1126,  0.1530,  0.6608, -0.7732, -0.3803, -1.1147,  1.4993], # logits of 'the
                 [-1.3186,  0.9240, -0.3326, -0.4618, -2.4549, -1.5394,  0.8870], # logits of 'weather'
                 [ 1.2215, -0.8649,  1.7561,  1.9612,  0.7086, -0.9581, -0.0143], # logits of 'is'
                 [-1.1442, -0.7637, -0.2789,  1.5958, -0.8117,  0.1906,  1.8612]]])  # logits of 'good' <------- WE WILL ONLY FOCUS ON THIS
# shape (batch_size, num_tokens, vocab_size)
```

Usually, we select the index with the highest probability after applying softmax to last logits. This index corresponds to token_id in the vocabulary, which is our next token/word.

However, to achieve randomness in the output instead of selecting token_id based on the highest probability, we will sample token_id proportional to its probability.

<!--more-->
