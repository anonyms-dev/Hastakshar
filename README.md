# HASTAKSHAR: A Generalized Framework for Deciphering Cross-Script Signatures via Multimodal Fusion

## Installation

```bash
pip install -r requirements.txt
```

## Abstract

The growing ubiquity of digitization has intensified the use of online signature verification, while offline signature verification has existed for years. Online signatures capture structural (shape) and behavioral modalities such as velocity, pressure, and stroke sequence. This increasing prevalence has also enabled AI-generated forgeries, demanding a generalized and robust verification system resilient against both manual and synthetic forgeries across diverse linguistic writing styles.

In this work, we propose Hastakshar (Hindi word for Signature), a multimodal framework that fuses Sigma-Lognormal stroke modeling, Ensemble Empirical Mode Decomposition (EEMD), and Hilbert–Huang Transform (HHT) based pressure analysis inside a unified Transformer architecture.

The trajectory is decomposed into lognormal stroke tokens that normalize neuromotor variability, while pressure signals are transformed via Hilbert analysis into instantaneous amplitude and instantaneous frequency tokens, capturing detailed surface dynamics. A multi-stream Transformer with cross-attention fuses these modalities to generate discriminative embeddings capable of identifying genuine and forged signatures.

To evaluate script-conditioned generalization, the framework is validated on datasets covering English and Dutch (Latin), Chinese (Hanzi), and Hindi (Devanagari). We also introduce the first Hindi Online Signature dataset — HinSig, with device metadata and baseline protocols to accelerate Indic script signature research. Hastakshar demonstrates improved verification accuracy along with interpretable decomposition via sigma-lognormal parameters and Hilbert spectral patterns.

## Performance of EER (80:20 vs 1:4)

| Database            | No. of Signatures | Train (80:20) | EER (80:20)<br>Strokes + Sigma-Lognormal + HHT | Train (1:4) | EER (1:4)<br>Strokes + Sigma-Lognormal + HHT |
|--------------------|-------------------|---------------|-----------------------------------------------|-------------|----------------------------------------------|
| SigComp11 Chinese  | 130               | 104           | **1.2615**                                     | 1 / 1       | **1.633**                                    |
| SigComp11 Dutch    | 120               | 95            | **0.2518**                                     | 1 / 1       | **1.9089**                                   |
| DeepSign DB        | 1526              | 1220          | **1.5986**                                     | 1 / 4       | **1.6035**                                   |
| Hindi (HinSig)     | 2500              | 200           | **1.4827**                                     | 3 / 2       | **1.8817**                                   |

## Figures

### Figure 1 — English (Latin Script) — DeepSignDB
![English Signature Analysis](figures/english_signature_analysis.png)

### Figure 2 — Hindi (Devanagari Script) — HinSig
![Hindi Signature Analysis](figures/hindi_signature_analysis.png)

### Figure 3 — Chinese (Hanzi Script) — SigComp11 Chinese
![Chinese Signature Analysis](figures/chinese_signature_analysis.png)

### Figure 4 — Dutch (Latin Script) — SigComp11 Dutch
![Dutch Signature Analysis](figures/dutch_signature_analysis.png)
