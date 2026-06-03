# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Assignment Overview

CSE 142 Spring 2026 — Build a decoder-only transformer language model from scratch on the TinyStories dataset. Five parts graded by unit tests and a staff retrain. **Due: 6/5/2026.**

Only these six files may be edited — the autograder extracts exactly them:
- `transformer_lm/model.py` — Part 2: all model components
- `transformer_lm/nn_utils.py` — Parts 2–3: `softmax`, `silu`, `cross_entropy_loss`
- `transformer_lm/training_utils.py` — Part 4: `get_batch`, `generate`
- `transformer_lm/tokenizer.py` — Part 1: BPE tokenizer
- `transformer_lm/lr_schedule.py` — LR schedule (default: cosine with warmup)
- `config.yaml` — architecture and training hyperparameters

Do **not** modify anything in `tests/`, `scripts/`, or any other file.

## Commands

```bash
# One-time data setup
PYTHONPATH=. python scripts/prepare_data.py

# Run all tests
PYTHONPATH=. python3 -m pytest tests/ -x -q

# Run tests for one part
PYTHONPATH=. python3 -m pytest tests/test_tokenizer.py -x -q   # Part 1
PYTHONPATH=. python3 -m pytest tests/test_model.py     -x -q   # Part 2
PYTHONPATH=. python3 -m pytest tests/test_nn_utils.py  -x -q   # Part 3
PYTHONPATH=. python3 -m pytest tests/test_training.py  -x -q   # Part 4

# Run one model component at a time
PYTHONPATH=. python3 -m pytest tests/test_model.py::TestLinear -v
PYTHONPATH=. python3 -m pytest tests/test_model.py::TestEmbedding -v
PYTHONPATH=. python3 -m pytest tests/test_model.py::TestRMSNorm -v
PYTHONPATH=. python3 -m pytest tests/test_model.py::TestScaledDotProductAttention -v
PYTHONPATH=. python3 -m pytest tests/test_model.py::TestCausalMultiHeadSelfAttention -v
PYTHONPATH=. python3 -m pytest tests/test_model.py::TestFeedForward -v
PYTHONPATH=. python3 -m pytest tests/test_model.py::TestTransformerBlock -v
PYTHONPATH=. python3 -m pytest tests/test_model.py::TestTransformerLM -v

# Run one function-level test (Parts 1, 3, 4)
PYTHONPATH=. python3 -m pytest tests/test_nn_utils.py::test_softmax_correctness -v
PYTHONPATH=. python3 -m pytest tests/test_training.py::test_get_batch_correctness -v

# Match by keyword
PYTHONPATH=. python3 -m pytest tests/ -k "softmax or silu" -v

# Sanity check (overfits a tiny batch, ~10s; run after Parts 2–3)
PYTHONPATH=. python3 scripts/sanity_check.py

# Train (reads config.yaml; auto-detects CUDA > MPS > CPU)
PYTHONPATH=. python3 scripts/train.py
PYTHONPATH=. python3 scripts/train.py --max_steps 500 --d_model 40 --n_layers 4 --n_heads 5 --d_ff 160

# Pre-submission gate (7 checks, stops at first failure)
PYTHONPATH=. python3 scripts/final_check.py submission.zip

# Create submission ZIP
zip -r submission.zip transformer_lm/model.py transformer_lm/nn_utils.py \
    transformer_lm/training_utils.py transformer_lm/tokenizer.py \
    transformer_lm/lr_schedule.py config.yaml
```

## Architecture

Pre-LN decoder-only transformer. Data flow: `token_emb → blocks × n_layers → ln_final → lm_head`. Output is raw logits `(B, T, V)`.

**Component specs (model.py):**
- `Linear`: `y = xW^T + b`. Weight uniform in `[-1/sqrt(d_in), 1/sqrt(d_in)]`, bias zero. All transformer projections use `bias=False`.
- `Embedding`: lookup table, weight init `N(0, 0.02)`.
- `RMSNorm`: `x / sqrt(mean(x²) + eps)`. **No learnable scale/gain parameter.** Zero parameters.
- `scaled_dot_product_attention`: `softmax(QK^T/sqrt(d_k) + mask) V`. Mask is **additive** (0 = attend, −10⁹ = masked out).
- `CausalMultiHeadSelfAttention`: fused QKV projection (`d_model → 3*d_model`), split into heads `(B, n_heads, T, d_head)`, apply RoPE to Q and K **after** reshaping into heads, build additive causal mask, compute attention, concat heads, output projection + dropout.
- `FeedForward` (SwiGLU): `w_down(silu(w_gate(x)) ⊙ w_up(x))`.
- `TransformerBlock` (Pre-LN): `x = x + attn(ln1(x))`, then `x = x + ffn(ln2(x))`.
- `TransformerLM`: weight tying — `self.lm_head.weight = self.token_emb.weight` (same tensor, set in `__init__`). Zero-init `attn.o_proj.weight` and `ffn.w_down.weight` for **every block** in `TransformerLM.__init__` (not in `TransformerBlock.__init__`). `forward` must raise `AssertionError` if sequence length exceeds `context_length`; store `self.context_length`.

**`RotaryPositionEmbedding` is provided in `model.py` — do not modify it.**

## Hard Constraints (autograder enforced)

| Constraint | Value |
|---|---|
| Maximum parameters | 500,000 (deduplicated; weight tying counts once) |
| Maximum training steps | 5,000 |
| Vocabulary size | 512 (fixed) |
| Context length | 256 (fixed) |
| Sanity gate | best val_loss < 3.0 |

Parameter formula (no bias, with weight tying): `params = 512 × d_model + n_layers × (4*d_model² + 3*d_model*d_ff)`

Architecture constraints: `d_model` divisible by `n_heads`; `d_head = d_model / n_heads` must be even.

## What You May NOT Use

**Allowed PyTorch surface** (`import torch.nn as nn` is fine — you need `nn.Module`, `nn.Parameter`, etc.):
- `torch.nn.Parameter`, container classes (`nn.Module`, `nn.ModuleList`, `nn.Sequential`), `nn.Dropout`
- All standard `torch.Tensor` operations and `torch.autograd`

**Banned specific APIs** (enforced by `TestAntiCheat` in `test_model.py` and `test_nn_utils.py`):
- `torch.nn` classes: `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.RMSNorm`, `nn.GELU`, `nn.SiLU`, `nn.Softmax`, `nn.MultiheadAttention`
- `torch.nn.functional (F.)`: `softmax`, `log_softmax`, `silu`, `gelu`, `cross_entropy`, `nll_loss`, `embedding`, `linear`, `scaled_dot_product_attention`, `rms_norm`
- `torch` / `Tensor` methods: `torch.softmax`, `torch.log_softmax`, `torch.logsumexp`, `torch.cross_entropy`, `Tensor.softmax`, `Tensor.log_softmax`
- Private dispatch: `torch._C`, `torch._C._nn`, `torch._VF`, `torch.ops.*`

**Banned standard-library imports** in `model.py`, `nn_utils.py`, `training_utils.py`, `lr_schedule.py`:
- `os`, `pathlib`, `sys`, `inspect`, `gc`, `subprocess`, `shutil`, `tempfile`, `urllib`, `requests`, `http`, `socket`, `pickle`, `marshal`, `open`, `exec`, `eval`, `getattr` on dynamic strings, `__import__`

**Forbidden architecture modifications:**
- GQA/MQA, Mixture of Experts, layer sharing beyond `lm_head`/`token_emb`, factored embeddings, learned RMSNorm gain vectors, additional skip connections or residual scaling, changes to RoPE.

**Hard-coded hyperparameters** (staff-fixed, cannot go in `config.yaml`): `vocab_size=512`, `context_length=256`, `max_steps=5000`, `seed=42`, optimizer=AdamW.

## config.yaml Allowed Keys

Only these 13 keys are accepted; unknown keys are rejected as errors during grading:

`d_model`, `n_layers`, `n_heads`, `d_ff`, `dropout`, `learning_rate`, `min_lr`, `batch_size`, `warmup_steps`, `weight_decay`, `gradient_clip_norm`, `beta1`, `beta2`

## Grading

Parts 1–4 are all-or-nothing (95% total). Part 5 (5%) and bonus tiers are scored on staff-retrained loss.

| Loss threshold | Effect |
|---|---|
| val_loss < 3.0 | Sanity gate (required for Parts 1–4 credit) |
| val_loss ≤ 1.70 | Part 5 (additional 5%) |
| hidden test loss ≤ 1.53 | Bonus: Pass |
| hidden test loss ≤ 1.47 | Bonus: Good |
| hidden test loss ≤ 1.43 | Bonus: Excellent |

## Recommended Implementation Order

1. `nn_utils.py`: `softmax`, `silu`
2. `model.py` bottom-up: `Linear` → `Embedding` → `RMSNorm` → `scaled_dot_product_attention` → `CausalMultiHeadSelfAttention` → `FeedForward` → `TransformerBlock` → `TransformerLM`
3. `nn_utils.py`: `cross_entropy_loss`
4. `training_utils.py`: `get_batch`, `generate`
5. Sanity check, then train and tune
6. `tokenizer.py`: BPE (independent, can be done anytime)

## Common Bugs

- **MHA shape**: track `(B,T,d_model) → (B,T,3d_model) → split → (B,n_heads,T,d_head) → attention → (B,T,d_model)`
- **RoPE before reshape**: RoPE must be applied to Q,K **after** reshaping into heads, not on flat `d_model` vectors
- **Multiplicative mask**: the causal mask must be **additive** (−10⁹ above diagonal), not multiplicative
- **`generate` logits**: sample from `logits[:, -1, :]` (last position only), not all positions
- **`cross_entropy_loss` reduction**: mean over all BT positions, not sum; use `keepdim=True` consistently in log-sum-exp
- **`get_batch` targets**: `y[i] = data[start+1 : start+ctx+1]`, not `data[start : start+ctx]`
- **Weight tying**: set `self.lm_head.weight = self.token_emb.weight` in `__init__`, not in `forward`
- **Zero-init placement**: zero `o_proj.weight` and `w_down.weight` in `TransformerLM.__init__`, not `TransformerBlock.__init__`
- **RMSNorm gain vector**: RMSNorm has **no** learnable parameters in this assignment
- **Context-length assertion**: `TransformerLM.forward` must raise `AssertionError` on sequences longer than `context_length`

## Test Counts by Component

| File | Class/Group | Tests |
|---|---|---|
| `test_model.py` (35 total) | `TestLinear` | 3 |
| | `TestEmbedding` | 2 |
| | `TestRMSNorm` | 6 |
| | `TestScaledDotProductAttention` | 4 |
| | `TestCausalMultiHeadSelfAttention` | 3 |
| | `TestFeedForward` | 2 |
| | `TestTransformerBlock` | 2 |
| | `TestTransformerLM` | 10 |
| | `TestAntiCheat` | 3 |
| `test_nn_utils.py` (11 total) | softmax | 3 + 1 anticheat |
| | silu | 1 + 1 anticheat |
| | cross_entropy | 4 + 1 anticheat |
| `test_tokenizer.py` | — | 12 |
| `test_training.py` | get_batch | 5 |
| | generate | 9 |
| | smoke/other | 1 |

## AI Assistant Policy

You may use LLMs to explain concepts and debug, but must not use them to write the implementation bodies of any required function/class. All submitted code must be understandable and re-derivable by the student. The experiment report must disclose any AI assistant usage (required section, even if none used).
