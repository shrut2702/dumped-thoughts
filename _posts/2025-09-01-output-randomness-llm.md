---
layout: post
title: "Understanding Temperature and Top-K in Large Language Models"
date: 2025-08-31
excerpt: "First attempt at blogging my learnings and projects."
---

What if every time we query the LLM a prompt, it responds with the same output? It would sound mundane and redundant. It would always generate same set of tokens, the tokens which has highest probability computed by the model. The task of a transformer-based LLM is to generate probability distribution for the vocabulary (set of all tokens/words known to a LLM) based on input sequence, from which it will predict next-word that is most related to input sequence or has the highest probability. This means for same input sequence, the LLM will generate exact same next-word.

<!--more-->
