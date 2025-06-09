AttnSeq-PPI
============
Overview

AttnSeq‑PPI is a PyTorch implementation of a two‑channel hybrid attention network that combines multi‑head self‑attention and multi‑head cross‑attention to capture both intra‑protein and inter‑protein contextual cues. Input sequences are first embedded residue‑wise with ProtT5‑XL (encoder‑only, fp16) and passed through local feature extractors (1‑D CNN + max‑pool) before the hybrid attention stack. A lightweight fully‑connected head outputs an interaction probability.

The framework achieves 99 % accuracy on Human and Multi‑Species datasets and, outperforming 15 state‑of‑the‑art baselines while retaining fast inference suitable for high‑throughput screening.

Complete workflow of AttnSeq‑PPI.
-

![Figure_2-01](https://github.com/user-attachments/assets/e6812e75-7c54-4941-8785-3aaba887e29f)


🌐 Web Tool
-
An online GUI version with pairwise and network prediction as well as interactive visualisation is freely available at:
https://compbiosysnbu.in/attnseqppi/

License
-
This project is distributed under the MIT License. See LICENSE for details.

Contac
-
Dipayan Sarkar
Research Scholar
Computational Systems Biology Lab
Department of Bioinformatics
Email: dipayansarkar26@gmail.com
