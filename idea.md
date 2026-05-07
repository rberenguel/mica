# Mica Transformer: Low-Data Regime Architecture and Training Spec

**Objective:** Train a custom, highly constrained Transformer (~10M-20M parameters) on a micro-corpus (10M words) locally on Apple Silicon (M1, 16GB Unified RAM, MPS backend). 

## 1. Architectural Blueprint (The "Mica" Configuration)
Do not use standard GPT-2 or LLaMA configurations. Implement the following strictly constrained hyperparameter set to prevent immediate overfitting and respect the 10-12GB VRAM training limit.

* **Total Parameters:** ~10M to 20M
* **Tokenizer:** Custom BPE (Byte Pair Encoding) or WordPiece trained *strictly* on the target author's corpus. 
    * `vocab_size`: 8192 or 10000 (Do not use standard 50k+ vocabs).
* **Embedding Dimension:** `d_model = 256` or `384`. (Forces semantic compression).
* **Transformer Layers:** `n_layers = 6`. (Limits representational capacity and residual accumulation).
* **Attention Heads:** `n_heads = 6` (if d_model=384) or `n_heads = 4` (if d_model=256). Ensure `d_model / n_heads == 64` for numerical stability in the softmax.
* **Context Window:** `block_size` or `max_seq_len = 256` or `512`. 

## 2. Architectural Tweaks (Regularization & Geometric Priors)
Implement these specific modifications to the standard attention mechanism to make the learning process harder and prevent exact sequence memorization.

* **Positional Encoding:** Implement **Rotary Position Embeddings (RoPE)** instead of absolute sinusoidal or learned absolute embeddings. This provides a strict inductive bias for relative grammatical distances.
* **Aggressive Dropout:** Set `dropout = 0.2`. Apply this to the residual pathways, the MLP outputs, and specifically the attention probability weights to force distributed learning.
* **Optimizer:** Use `AdamW`.
* **Weight Decay:** Set a high weight decay (e.g., `weight_decay = 0.1`) to heavily penalize large weights and maintain a tight matrix spectrum.

## 3. The "Kickstart" Training Pipeline (Transfer Learning)
Execute training in two distinct, sequential phases to solve the low-data constraint.

* **Phase 1: The Grammar Engine (Pre-training)**
    * **Dataset:** Simple Wikipedia (~200M words).
    * **Goal:** Teach the model basic English syntax, noun/verb structures, and general semantic relationships.
    * **Hyperparameters:** Standard learning rate schedule (e.g., 1e-3 max with cosine decay and warmup). Train until validation loss plateaus.
* **Phase 2: The Style Transfer (Fine-tuning)**
    * **Dataset:** Target Micro-Corpus (e.g., 10M words of Edgar Allan Poe).
    * **Goal:** Overwrite the generic style with the specific vocabulary, cadence, and conceptual manifold of the target author.
    * **Hyperparameters:** Drop the maximum learning rate by an order of magnitude (e.g., 1e-4) to gently shift the weights without destroying the Phase 1 grammatical foundation. Train until target style emerges, monitoring strictly for memorization/overfitting.