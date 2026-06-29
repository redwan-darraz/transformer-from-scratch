# BPE Tokenizer — from scratch

Pure Python implementation of Byte Pair Encoding, the tokenization algorithm used by GPT-2, GPT-3 and GPT-4.

No external libraries used for the core algorithm — only `collections`. tiktoken is used at the end for comparison only.

## What it does

Transforms raw text into integer token IDs, the same way a real LLM tokenizer works:

```python
"the model learns"  →  [35, 17, 62, 9, 55, 16, 78, 18, 32]
[35, 17, 62, 9, 55, 16, 78, 18, 32]  →  "the model learns"
```

## How it works

BPE starts with individual characters and repeatedly merges the most frequent adjacent pair into a single token. After 50 merges on a 100-sentence corpus:

- `"the"` goes from 4 tokens → 1 token (merged at iteration 7)
- `"training"` goes from 9 tokens → 3 tokens (`tra` + `in` + `ing`)
- `"backpropagation"` stays split — too rare to fully merge

## Implementation

| Function        | Role                                                          |
| --------------- | ------------------------------------------------------------- |
| `build_vocab()` | Character-level word frequency dictionary with `</w>` markers |
| `get_stats()`   | Count adjacent symbol pairs, weighted by word frequency       |
| `merge_vocab()` | Apply one merge rule across the entire vocabulary             |
| `encode()`      | Text → list of integer IDs                                    |
| `decode()`      | List of integer IDs → text                                    |

## Results

50 merges on 100 sentences produces a ~2x token compression ratio vs character-level encoding.

Compared against tiktoken `cl100k_base` (GPT-4's tokenizer): same algorithm, but 100,000 merge rules trained on hundreds of billions of tokens. The core difference is scale, not the algorithm itself.

## Stack

Python 3.11 · collections · tiktoken · Jupyter Notebook
