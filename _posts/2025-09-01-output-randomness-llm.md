---
layout: post
title: "Understanding Temperature and Top-K in Large Language Models"
date: 2025-09-01
excerpt: "What if every time we query the LLM a prompt, it responds with the same output?"
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
                 [ 0.3271,  0.0117, -1.5605, -0.0686,  0.7606,  1.2529, -0.8271]]])  # logits of 'good' <------- WE WILL ONLY FOCUS ON THIS
# shape (batch_size, num_tokens, vocab_size)
```

Usually, we select the index with the highest probability after applying softmax to last logits. This index corresponds to token_id in the vocabulary, which is our next token/word.

However, to achieve randomness in the output instead of selecting token_id based on the highest probability, we will sample token_id proportional to its probability.

Probability distribution after applying softmax to last logits:

```code
last_logits = logits[:,-1,:]
softmax_prob = torch.softmax(last_logits, dim=-1)
```

output:

```output
tensor([[0.1442, 0.1052, 0.0218, 0.0971, 0.2224, 0.3639, 0.0455]])
```

Earlier, the output would be always "tomorrow" (token_id 5). Now, if we sample from the output 1000 times then we get following distribution:

```output
the: 147
weather: 110
is: 23
good: 83
today: 244
tomorrow: 351
never: 42
```

We can control this probability distribution to be more uniform or spiked with the help of temperature parameter. To do that, we divide the logits by temperature before we apply softmax function. By default, the temperature value is set to 1, as dividing the logits by 1 would result into same values. If the temperature is less than 1, the logits will increase than the original values. And since the softmax is an exponential function, even the slight increase in the input to it can lead to huge differences in probability, and thus, making high values higher and low values lower. Therefore, the sampling probability also increases considerably, resulting almost constant token when temperature is less than 1.

Probability distribution with temperature = 0.2

```code
def temp_scaled_softmax(logits, temp=1):
  scaled_values = logits/temp
  prob = torch.softmax(scaled_values, dim=-1)
  return prob

temp_scaled_softmax(last_logits, temp=0.2)
```

output:

```output
tensor([[8.8888e-03, 1.8364e-03, 7.0796e-07, 1.2294e-03, 7.7659e-02, 9.1036e-01, 2.7703e-05]])
```

See how spiked the distribution is, the earlier high values are even higher now and low values are almost 0.

On the other hand, when the temperature increases, the resulting logits decreases, thus, softmax function will return nearly uniform distribution. Therefore, each token in the vocabulary has almost similar chances of getting picked, resulting into varying and creative output. But this also means generating text which doesn't make any sense, because each token, whether relevant or not, has similar probability.

Probability distribution with temperature = 5

```code
temp_scaled_softmax(last_logits, temp=5)
```

output:

```output
tensor([[0.1507, 0.1415, 0.1033, 0.1392, 0.1643, 0.1813, 0.1196]])
```

As can be seen above, every token has nearly equal probabilities, this might lead to incoherent text generation.

![Alt text]({{ site.baseurl }}/assets/images/temp_prob_dist.png)

To overcome this drawback, the top-k parameter comes in handy. It limits the token sampling from the output probability distribution to the top 'k' values, i.e., if k=10, only indices with top 10 highest probabilities across the output will be considered. This ensures the top 10 most relevant tokens will the output be sampled from, generating coherent text even with the high temperature value.

Probability distribution with temperature = 5 and top-k = 3

```code
top_k=3
top_logits, top_idx = torch.topk(last_logits.squeeze(0), top_k)
masked_logits = torch.where(
    condition = last_logits < top_logits[-1],
    input = torch.tensor(float('-inf')),
    other = last_logits
)

temp_scaled_softmax(masked_logits, temp=5)
```

output:

```output
tensor([[0.3036, 0.0000, 0.0000, 0.0000, 0.3311, 0.3653, 0.0000]])
```

Here, the probability of every token other than 0,4 and 5 is 0, meaning the output will be sampled from only these three tokens. Consequently, the generated text will be more coherent and non-redundant.

Depending on the use-case, we can tune these parameters to get desired output. For example, if we want to generate a poem or story, we can set temperature to a high value like 3 and top-k to 50. But if we want to generate a technical document or article, we can set temperature to a low value like 0.5 and top-k to 10.
