# transformer-from-scratch

Hands-on implementation of the core Transformer building blocks — BPE tokenization and the attention mechanism — built from scratch in pure Python and PyTorch.

## Labs

### LAB 1.1 — BPE Tokenizer from Scratch

Implement Byte Pair Encoding from A to Z, verify a perfect encode/decode round-trip, compare with tiktoken. → [`lab01_tokenizer/`](lab01_tokenizer/)

### LAB 1.2 — Multi-Head Attention from Scratch

Scaled dot-product attention in NumPy → Multi-Head Attention in PyTorch, verified against `nn.MultiheadAttention`. → [`lab02_attention/`](lab02_attention/)

## Stack

Python 3.11 · NumPy · PyTorch · matplotlib · Jupyter

## Resources

- [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
