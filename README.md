# AttnSeq-PPI
Overview

AttnSeq‑PPI is a PyTorch implementation of a two‑channel hybrid attention network that combines multi‑head self‑attention and multi‑head cross‑attention to capture both intra‑protein and inter‑protein contextual cues. Input sequences are first embedded residue‑wise with ProtT5‑XL (encoder‑only, fp16) and passed through local feature extractors (1‑D CNN + max‑pool) before the hybrid attention stack. A lightweight fully‑connected head outputs an interaction probability.

The framework achieves 99 % accuracy on Human and Multi‑Species datasets and ≥ 0.94 MCC on Yeast, outperforming 15 state‑of‑the‑art baselines while retaining fast inference suitable for high‑throughput screening.