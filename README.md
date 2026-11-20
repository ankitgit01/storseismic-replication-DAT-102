# StorSeismic: Reproduction & Extended Experiments

PyTorch reproduction and extension of **StorSeismic** (*Harsuko & Alkhalifah, 2022*) for self-supervised seismic pre-training and downstream processing using **OpenFWI** (synthetic) and **Mobil AVO Viking Graben Line 12** (field) data.

📄 **Full Project Report:** [Link to Report](./report.pdf) *(see report for methodology, detailed analysis, and all figures)*

---

## Repository Structure

```text
storseismic/
├── data/
│   ├── mobil/seismic.segy            # Mobil AVO Line 12 field gathers
│   └── openfwi/
│       ├── data1.npy                 # OpenFWI synthetic shot gathers
│       └── model1.npy                # OpenFWI 2D velocity models
├── nb3_mobil_openfwi_section4.ipynb  # Stage 1 pre-training + Stage 2 denoising & velocity fine-tuning
├── nb4_storseismic_experiments.ipynb # Ablations (A1, A2, A3, A5), zero-shot interpolation (B2), attention maps (B5)
└── results/
    ├── mobil_openfwi_section4/       # Saved nb3 checkpoints & training history
    └── nb4_experiments/              # Saved nb4 checkpoints, JSON metrics & 300-DPI figures
```

---

## Setup

Requires **Python 3.10+** and a CUDA-enabled GPU (tested with PyTorch 2.5+ / CUDA 12.1).

```bash
# Using uv (recommended)
uv venv --python 3.10
uv pip install torch numpy scipy segyio matplotlib jupyter

# Or using pip
pip install torch numpy scipy segyio matplotlib jupyter
```

---

## Usage

1. **Main Pipeline (`nb3_mobil_openfwi_section4.ipynb`):**
   - Set `RUN_TRAINING = True` on the first run to pre-train (`BERT_MINI`, 50/50 OpenFWI + Mobil) and fine-tune for Denoising and 1D Velocity Prediction.
   - Set `RUN_TRAINING = False` afterwards to reload saved checkpoints and regenerate plots without re-training.

2. **Extended Experiments (`nb4_storseismic_experiments.ipynb`):**
   - Automatically reuses `nb3` checkpoints and caches all ablation checkpoints/metrics under `results/nb4_experiments/`.
   - Covers:
     - **A1:** Pre-training vs. From-Scratch sample efficiency (10%–100% labels)
     - **A2:** Partial layer freezing sweep (`0`–`4` frozen layers)
     - **A3:** Pre-training data mix ablation (100% OpenFWI vs. 50/50 Mixed vs. 100% Mobil)
     - **A5:** Model capacity scaling (`BERT_TINY`, `BERT_MINI`, `BERT_SMALL`)
     - **B2:** Zero-shot Mobil dead-trace interpolation vs. Linear/Cubic baselines
     - **B5:** Layer-wise self-attention map interpretability

---

## Reference

- Harsuko, R., & Alkhalifah, T. A. (2022). *StorSeismic: A New Paradigm in Deep Learning for Seismic Processing*. IEEE Transactions on Geoscience and Remote Sensing.