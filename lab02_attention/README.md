# LAB 1.2 — Multi-Head Attention from Scratch

**Difficulty:** ⭐⭐⭐☆☆ · **Time:** 4-5h · **Stack:** NumPy, PyTorch, matplotlib, Jupyter

## Goal

Implement the attention mechanism — the core of the Transformer. First in NumPy for the math, then in PyTorch. Visualize what the model "looks at" with a heatmap. After this lab, *Attention is All You Need* will have no more secrets.

## Steps

1. In NumPy: `scaled_dot_product_attention(Q, K, V)` = softmax(QKᵀ / √d_k) · V
2. Visualize the attention matrix on 5 words with `matplotlib.imshow` — which tokens attend to which?
3. Multi-Head: split Q, K, V into h heads, run attention on each head, concatenate, project.
4. In PyTorch: `MultiHeadAttention(nn.Module)` class with a clean `forward()`.
5. Verify your implementation matches `nn.MultiheadAttention` (diff < 1e-4).
6. Add an ASCII architecture diagram to this README. Open a PR onto main.

## Success Criteria

- Attention matrix: values in [0, 1], each row sums to 1.0
- Match `nn.MultiheadAttention` with diff < 1e-4
- You can explain why we divide by √d_k without hesitating
- Attention heatmap visible in the README

## Recruiter Demo

Show the heatmap. "This token attends particularly to that token because..." — intuitive explanation.
