# LAB 1.1 — BPE Tokenizer from Scratch

**Difficulty:** ⭐⭐☆☆☆ · **Time:** 3-4h · **Stack:** Python 3.11, collections, tiktoken, Jupyter

## Goal

Understand how an LLM transforms text into tokens. Implement BPE (Byte Pair Encoding) from A to Z using pure Python, no external libraries.

## Steps

1. Create a corpus of 100 sentences. Implement `get_stats(vocab)` — count all adjacent symbol pairs.
2. Implement `merge_vocab(pair, vocab)` — merge the most frequent pair into a new token.
3. Loop 50 merges — observe the vocabulary evolving from characters to full words.
4. Write `encode(text)` and `decode(tokens)` — verify a perfect round-trip.
5. Compare with tiktoken (`cl100k_base`) on 5 sentences — explain the differences.
6. Push to GitHub branch `feature/tokenizer-bpe` — clean notebook with visible outputs.

## Success Criteria

- `decode(encode("Hello world")) == "Hello world"` — perfect round-trip
- You can explain BPE in 90 seconds without notes
- Notebook with outputs visible on GitHub

## Recruiter Demo

Open the notebook, tokenize a sentence live, explain each cell.
