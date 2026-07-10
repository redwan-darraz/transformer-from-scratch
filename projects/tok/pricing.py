"""
Estimated API cost per model, based on real tokenizers.
"""

from engines import run_tiktoken, run_huggingface

# Input pricing, USD per 1M tokens (2026 published rates).
# gpt4o / gpt4o-mini are counted with tiktoken (their real tokenizer).
# mistral is counted with the HuggingFace mistral tokenizer (its real tokenizer).
# claude-haiku has no public pip tokenizer, so tiktoken is used as an approximation.
PRICING = {
    "gpt4o":        {"tokenizer": "tiktoken",     "usd_per_million": 2.50},
    "gpt4o-mini":   {"tokenizer": "tiktoken",     "usd_per_million": 0.15},
    "mistral":      {"tokenizer": "huggingface",  "usd_per_million": 2.00},
    "claude-haiku": {"tokenizer": "tiktoken",     "usd_per_million": 0.80},
}


def estimate_cost(text, model):
    """Estimated cost in USD cents for tokenizing `text` as input to `model`."""
    info = PRICING[model]
    tokens = run_tiktoken(text) if info["tokenizer"] == "tiktoken" else run_huggingface(text)
    usd = len(tokens) / 1_000_000 * info["usd_per_million"]
    return len(tokens), usd * 100
