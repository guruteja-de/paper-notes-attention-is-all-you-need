# paper-notes-attention-is-all-you-need
Detailed notes and code walkthrough of the Attention Is All You Need paper (Vaswani et al., 2017) with numerical examples

# 📄 Attention Is All You Need — Paper Notes & Code Walkthrough

> **Paper:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., Google Brain (NIPS 2017)  
> **Studied by:** Guru Teja  
> **Purpose:** Foundation study for Zero-Shot Navigation with CLIP + RL project

---

## 🎯 Why I Studied This Paper

The Transformer architecture introduced in this paper (2017) is the direct foundation of **CLIP** (2021), which I use in my navigation project. Before building the RL agent, I needed to understand every component of the Transformer from first principles — attention mechanisms, positional encoding, encoder-decoder structure, and training techniques.

This repository documents that learning with detailed explanations, numerical examples computed step by step, and runnable PyTorch code.

---

## 📁 Repository Structure

```
paper-notes-attention-is-all-you-need/
│
├── notebook/
│   └── Attention_Is_All_You_Need.ipynb   ← full walkthrough with code
│
├── notes/
│   ├── 01_problem_with_rnns.md           ← why RNNs were slow
│   ├── 02_attention_formula.md           ← scaled dot-product attention
│   ├── 03_multi_head_attention.md        ← why 8 heads beat 1 head
│   ├── 04_positional_encoding.md         ← sine/cosine encoding
│   ├── 05_encoder_decoder.md             ← full translation pipeline
│   └── 06_results_and_impact.md          ← BLEU scores, training costs
│
├── README.md
└── requirements.txt
```

---

## 📓 What the Notebook Covers

The notebook walks through the entire paper section by section. Every concept has:
- Plain English explanation
- Real-world analogy
- Numerical example computed by hand
- PyTorch code that verifies the math

| Section | Topic | Key Concept |
|---------|-------|-------------|
| 1 | Problem with RNNs | Sequential bottleneck, vanishing gradients |
| 2 | The Big Idea | Replace recurrence entirely with Attention |
| 3 | Tokenisation & Embeddings | Words → 512-dim vectors |
| 4 | Positional Encoding | Sinusoidal fingerprints that inject word order |
| 5 | Scaled Dot-Product Attention | `Attention(Q,K,V) = softmax(QKᵀ/√d_k)V` |
| 6 | Multi-Head Attention | 8 parallel heads, each specialising differently |
| 7 | Feed-Forward Network | Per-token non-linear reasoning |
| 8 | Residual Connections & LayerNorm | Training stability across 6 deep layers |
| 9 | The Encoder | All 6 pieces combined, processes tokens in parallel |
| 10 | The Decoder | Masked attention + cross-attention + generation loop |
| 11 | Full Translation Example | 'The cat sat' → 'Die Katze sass' traced step by step |
| 12 | Why Transformers Beat RNNs | Complexity table with real numbers |
| 13 | Training Details | Adam optimizer, LR warmup, label smoothing |
| 14 | Results | BLEU scores and training cost analysis |
| 15 | Connection to CLIP | How this paper leads to my navigation project |

---

## 🔑 Key Concepts Explained

### The One Formula That Changed AI

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**In plain English:**
1. Compute how similar my Query is to every Key
2. Divide by √d_k — prevents softmax from becoming too sharp
3. softmax converts scores to probabilities (attention weights)
4. Weighted sum of Values — each token gets context from all others

### Why Multi-Head Instead of Single Head?

A single 512-dim head produces one blurry answer trying to serve all purposes. Eight 64-dim heads each produce one sharp answer to a different question — coreference, syntax, position, semantics — and the final concatenation gives all answers simultaneously.

```
8 heads × 64 dims = 512 dims  (same total cost, much richer representation)
```

### Why Residual Connections?

Without residual connections, signals shrink at every layer:

```
Without residual (×0.8 per layer):  5.0 → 4.0 → 3.2 → ... → 0.34  ← signal GONE
With    residual (x + SubLayer(x)): 5.0 → 9.0 → 16.2 → ... → 6341  ← signal PRESERVED
```

The skip connection creates a direct gradient highway back to early layers — every layer learns equally fast.

---

## 📊 Results from the Paper

| Model | EN-DE BLEU | Training Cost |
|-------|-----------|---------------|
| GNMT + RL (Google's system) | 24.6 | 2.3 × 10¹⁹ FLOPs |
| ConvS2S (Facebook) | 25.16 | 9.6 × 10¹⁸ FLOPs |
| ConvS2S Ensemble (8 models) | 26.36 | 7.7 × 10¹⁹ FLOPs |
| **Transformer base (1 model)** | **27.3** | **3.3 × 10¹⁸ FLOPs** |
| **Transformer big (1 model)** | **28.4** | 2.3 × 10¹⁹ FLOPs |

**One Transformer outperforms ensembles of 8 previous models at 7× lower training cost.**

Training time: **12 hours on 8 NVIDIA P100 GPUs.**

---

## 🔗 Connection to My Project

```
Attention Is All You Need (2017)
        │  Transformer architecture invented
        ▼
ViT — Vision Transformer (2020)
        │  Same Transformer blocks applied to image patches
        ▼
CLIP (2021) — Contrastive Language-Image Pretraining
        │  Image encoder (ViT) + Text encoder (Transformer)
        │  Trained on 400M image-text pairs
        │  Both encoders output 512-dim vectors in the same space
        ▼
Zero-Shot Navigation with CLIP + RL  ← my project
        │
        ├── Goal: "find the red apple"
        │     → CLIP Text Encoder → 512-dim text embedding
        │
        ├── Agent camera frame
        │     → CLIP Vision Encoder → 512-dim image embedding
        │
        └── reward = dot(image_embedding, text_embedding)
              → PPO agent learns to navigate toward the target
              → ZERO-SHOT: works for any object in natural language
```

Every component I use — the 512-dim embedding space, multi-head attention, the encoder architecture, causal masking in the text encoder — comes directly from this 2017 paper.

---


## 📦 Requirements

```
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
jupyter>=1.0.0
```

---

## 📚 References

| Resource | Link |
|---------|------|
| Original Paper | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| The Illustrated Transformer | [Jay Alammar's blog](http://jalammar.github.io/illustrated-transformer/) |
| CLIP Paper | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) |
| My Navigation Project | [Zero-Shot Navigation with CLIP + RL](https://github.com/guruteja-de) |

---

## 📌 Related Work

This study is part of my broader research project on zero-shot object navigation using vision-language models and reinforcement learning. The Transformer understanding built here directly informs:

- How CLIP's text encoder processes navigation goals
- Why the 512-dim shared embedding space enables zero-shot generalisation
- How causal masking in the text encoder ensures EOT captures full goal intent
- Why the Vision Transformer (ViT) applies the same blocks to image patches

---
