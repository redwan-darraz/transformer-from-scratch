# tok — Tokenizer CLI

A command-line tool that tokenizes any text and compares three tokenizers side by side: my BPE implementation from lab01, tiktoken (GPT-4), and a HuggingFace tokenizer.

Useful for understanding how different tokenizers handle the same text, and for estimating token counts before making API calls.

## Usage

```bash
# Tokenize text with all three tokenizers
python -m tok "Your text here"

# Show detailed token breakdown
python -m tok "Your text here" --verbose

# Compare compression stats only
python -m tok "Your text here" --stats
```

## Output example

```
Input: "the transformer architecture uses self attention"

┌─────────────────┬────────┬──────────────────────────────────────────┐
│ Tokenizer       │ Tokens │ Top tokens                               │
├─────────────────┼────────┼──────────────────────────────────────────┤
│ BPE (lab01)     │ 18     │ the, er, arch, it, ect, ure, ...         │
│ tiktoken        │  7     │ the, transformer, architecture, ...      │
│ HuggingFace     │  9     │ the, transform, ##er, architecture, ...  │
└─────────────────┴────────┴──────────────────────────────────────────┘

Compression ratio vs characters:
  BPE (lab01)  :  2.4x
  tiktoken     :  6.1x   ← best
  HuggingFace  :  4.9x
```

## Stack

Python 3.11 · tiktoken · transformers (HuggingFace) · collections
