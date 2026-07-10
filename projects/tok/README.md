# tok — Tokenizer CLI

A command-line tool that tokenizes any text with three tokenizers side by side: a BPE implementation built from scratch (see [lab01](../../lab01_tokenizer)), tiktoken (GPT-4's `cl100k_base`), and Mistral's tokenizer.

Useful for understanding how different tokenizers split the same text, and for estimating the cost of a prompt before sending it to an API.

## Usage

```bash
tok "your text here"                              # compare all three tokenizers
tok "your text here" --compare                    # same, as a side-by-side table
tok "your text here" --cost --model gpt4o          # estimate API cost
tok "your text here" --compare --cost --model mistral
```

## Output example

```
Input: "gradient descent minimizes the loss"  (35 characters)

┌────────────────────────┬────────┬─────────────┬──────────────────────────────────────────┐
│ Tokenizer              │ Tokens │ Chars/token │ Preview                                  │
├────────────────────────┼────────┼─────────────┼──────────────────────────────────────────┤
│ BPE (from scratch)     │ 19     │ 1.84        │ g ra d i ent de s c ent m in i m iz e... │
│ tiktoken (cl100k_base) │ 6      │ 5.83        │ gradient  descent  minim izes  the  loss │
│ HuggingFace (mistral)  │ 6      │ 5.83        │ ▁gradient ▁descent ▁minim izes ▁the ▁... │
└────────────────────────┴────────┴─────────────┴──────────────────────────────────────────┘

Estimated cost on gpt4o (6 tokens): 0.0015 cents ($0.000015)
```

A from-scratch BPE trained on 16 sentences needs ~3x more tokens than tiktoken or Mistral, both trained at production scale — same algorithm, wildly different results depending on training data size. See [`benchmark.py`](benchmark.py) for a visual comparison across several sentences: [`tokenizer_comparison.png`](tokenizer_comparison.png).

## Real usage

Before any large API call, check the actual cost first:

```bash
tok "$(cat my_long_prompt.txt)" --cost --model gpt4o
```

## Install as a global command

Add a `tok` function to your PowerShell profile (`$PROFILE`) so it's available in any terminal, from any directory:

```powershell
function tok {
    & "<path-to-repo>\transformer-from-scratch\.venv\Scripts\python.exe" `
      "<path-to-repo>\transformer-from-scratch\projects\tok\tok.py" @args
}
```

Reload with `. $PROFILE`, then `tok "hello world"` works from anywhere.

## Structure

| File | Role |
| --- | --- |
| `bpe.py` | BPE tokenizer from scratch, trained on import |
| `engines.py` | The three tokenizers behind one common interface |
| `display.py` | Terminal rendering — colored tokens, comparison table |
| `pricing.py` | 2026 pricing table and cost estimation |
| `tok.py` | Entry point — argparse and orchestration only |
| `benchmark.py` | Runs a batch of sentences and plots a comparison chart |
| `mistral_tokenizer.json` | Mistral tokenizer, bundled locally for instant, offline loading |

## Why it's fast

The first version loaded Mistral's tokenizer through `transformers.AutoTokenizer`, which imports torch under the hood — about 5 seconds just to start up. Switched to the lightweight `tokenizers` library and bundled the tokenizer file locally instead of fetching it from the Hub on every run. Total runtime is now under 0.5s.

## Stack

Python 3.11 · tiktoken · tokenizers · colorama
