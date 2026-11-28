# StorSeismic: Reproduction & Multi-Dataset Field Study (DAT-102)

PyTorch implementation, reproduction, and extended experimental evaluation of **StorSeismic** (*Harsuko & Alkhalifah, 2022*) for self-supervised seismic pre-training and downstream geophysical processing. Alongside the upstream **SNIST** reproduction notebooks (Track A), this repository evaluates the framework across four synthetic–field dataset pairs (**Stratton 3D**, **Volve 3D OBC**, **Mobil AVO**, and **USGS Alaska Line 31-81**, paired with **OpenFWI CurveVel-A** and **SEG/EAGE 3-D Salt**), comparing pure-synthetic vs. 25% field-mixed Stage 2A denoising, plus systematic ablations in `nb4` (Track B).

---

## Key Experimental Findings (Track B)

### 1. Multi-Dataset Field Workflow (`nb3_1`–`nb3_4`): Pure-Synthetic (`0%`) vs. Field-Mixed (`25%`) Stage 2A Denoising

| Notebook | Dataset Pair (Synth + Field) | Stage 2A Mix | Synth SNR | Synth MSE | Field Corr ($r$) | Rem. Energy | Vel MAE | Vel $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nb3_1_stratton_segsalt.ipynb`** | SEG Salt + Stratton 3D (3D Land) | `0%` | `4.46 dB` | `1.70e-2` | `0.5775` | `67.91%` | `162.5 m/s` | `0.6833` |
| **`nb3_2_volve_segsalt.ipynb`** | SEG Salt + Volve 3D OBC (3D Marine) | `0%` | `4.52 dB` | `1.74e-2` | `0.5854` | `66.95%` | **`117.1 m/s`** | **`0.7741`** |
| **`nb3_3_failed_mobil_openfwi.ipynb`** | OpenFWI + Mobil AVO (2D Marine) | `0%` | **`16.18 dB`** | `4.45e-5` | `-0.0008` *(collapse)* | `100.82%` | `307.0 m/s` | `0.6917` |
| **`nb3_3_mobil_openfwi_final.ipynb`** | OpenFWI + Mobil AVO (2D Marine) | `25%` | `15.49 dB` | `4.77e-5` | **`0.4494`** | `80.97%` | `298.0 m/s` | `0.6927` |
| **`nb3_4_alaska_openfwi.ipynb`** | OpenFWI + Alaska 31-81 (2D Land) | `25%` | `9.67 dB` | `1.59e-4` | **`0.5890`** | `65.67%` | `296.0 m/s` | `0.7141` |

> **Key Takeaway:** Pure-synthetic Stage 2A fine-tuning (`0%` mix) transfers to 3D post-stack/OBC surveys (`nb3_1`, `nb3_2`), but collapses on 2D marine streamer data (`nb3_3_failed`, $r \approx 0$, `~100%` energy removed). Mixing **25% field gathers + Gaussian noise** in Stage 2A (`nb3_3_mobil_openfwi_final`, `nb3_4_alaska_openfwi`) eliminates this domain collapse while preserving synthetic accuracy.

### 2. Ablations & Interpretability (`nb4_storseismic_experiments.ipynb` on Mobil + OpenFWI)

| Experiment / Task | Baseline / From Scratch | StorSeismic (Pre-Trained) | Key Finding & Geophysical Takeaway |
| :--- | :---: | :---: | :--- |
| **A1: Denoising (10% Labels / 40 Gathers)** | -0.83 dB (Scratch) | **+9.17 dB** (StorSeismic) | **+10.00 dB gain**: Pre-training prevents failure in low-data regimes. |
| **A1: Velocity MAE (10% Labels)** | 388.2 m/s (Scratch) | **315.1 m/s** (StorSeismic) | **73.1 m/s error reduction** using only 40 training gathers. |
| **A2: Partial Layer Freezing ($k=2$)** | 15.05 dB (Full FT, $k=0$) | **16.18 dB** (Frozen, $k=2$) | Freezing 2 layers retains accuracy while protecting backbone features. |
| **B1: Zero-Shot Mobil Trace Interp. (20%)** | -2.71 dB (Linear Interp.) | **+0.09 dB** (StorSeismic) | Outperforms classical linear/cubic interpolation zero-shot on missing field traces. |

---

## Repository Structure

```text
storseismic/
├── nb0_1_data_prep_pretrain.ipynb       # Track A: SNIST pre-training data preparation
├── nb0_2_data_prep_finetune.ipynb       # Track A: SNIST fine-tuning data preparation
├── nb1_pretraining.ipynb                # Track A: Masked trace pre-training (SNIST)
├── nb2_1_finetuning_denoising.ipynb     # Track A: Denoising fine-tuning (SNIST)
├── nb2_2_finetuning_velpred.ipynb       # Track A: Velocity prediction fine-tuning (SNIST)
├── nb3_1_stratton_segsalt.ipynb         # Track B: SEG Salt + Stratton 3D (0% field mix)
├── nb3_2_volve_segsalt.ipynb            # Track B: SEG Salt + Volve 3D OBC (0% field mix)
├── nb3_3_failed_mobil_openfwi.ipynb     # Track B: OpenFWI + Mobil AVO (0% field mix, collapse demo)
├── nb3_3_mobil_openfwi_final.ipynb      # Track B: OpenFWI + Mobil AVO (25% field mix fix)
├── nb3_4_alaska_openfwi.ipynb           # Track B: OpenFWI + Alaska 31-81 (25% field mix fix)
├── nb4_storseismic_experiments.ipynb    # Track B: Ablations (A1–A4) & analyses (B1, B2)
├── storseismic/                         # Core neural network modules used by Track A
├── data/                                # Input datasets (gitignored)
│   ├── mobil/seismic.segy               # Mobil AVO Line 12 (2D Marine)
│   ├── alaska_31_81.segy                # USGS Alaska Line 31-81 (2D Land)
│   ├── Stratton3D_32bit.sgy             # Stratton 3D (3D Land)
│   ├── volve_3d.segy                    # Volve 3D OBC (3D Marine)
│   ├── openfwi/{data1,model1}.npy       # OpenFWI CurveVel-A gathers & velocity models
│   └── Salt_Model_3D/                   # SEG/EAGE 3-D Salt ClassicA SEG-Y & SALTF.ZIP velocity grid
└── results/                             # Saved checkpoints (.pt), metrics (.json/.npz), & figures (.png)
    ├── stratton_segsalt_nomix/
    ├── volve_segsalt_nomix/
    ├── alaska_openfwi/
    ├── mobil_openfwi_section4/
    └── nb4_experiments/
```

---

## Setup & Dependencies

Requires **Python 3.10+** and a CUDA-enabled GPU (tested on NVIDIA RTX / T4 / A100):

```bash
git clone <repository-url>
cd storseismic

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install torch numpy scipy segyio matplotlib jupyter
```

> **Note:** Track A notebooks (`nb0_1`–`nb2_2`) additionally require `transformers` and `scikit-learn`.

---

## Execution Guide

### Track B: Field Framework & Extended Experiments

- **Caching (`RUN_TRAINING`):** Set `RUN_TRAINING = False` to reload cached checkpoints and regenerate all metrics and figures without retraining; set `RUN_TRAINING = True` on an initial run to train locally.
- **3-Stage Pipeline (`nb3_*`):**
  1. **Stage 1:** Self-supervised masked trace pre-training (`BERT_MINI`: $L=4, H=256, A=4, d_{\text{ff}}=1024$) on a 50/50 synthetic–field mix.
  2. **Stage 2:** Supervised fine-tuning with partial layer freezing ($k=2$) for **Denoising** (MSE) and **1D Velocity Prediction** (L1, sigmoid-bounded in SEG Salt notebooks).
  3. **Stage 3:** Inference, $f-k$ spectral decomposition, 2D inverted velocity cross-sections, $V_p$ parity scatter plots, zero-shot trace interpolation, and 16-head attention visualizations.
- **Ablations (`nb4_storseismic_experiments.ipynb`):** Evaluates **A1** (sample efficiency `10%`–`100%`), **A2** (layer freezing $k \in \{0..4\}$), **A3** (pre-training corpus mix), **A4** (model capacity scaling), **B1** (zero-shot dead-trace interpolation), and **B2** (16-head attention dissection, rollout, and layer-4 task drift).

### Track A: Upstream SNIST Reproduction

Run sequentially: `nb0_1_data_prep_pretrain.ipynb` → `nb0_2_data_prep_finetune.ipynb` → `nb1_pretraining.ipynb` → `nb2_1_finetuning_denoising.ipynb` and `nb2_2_finetuning_velpred.ipynb`.

---

## References

- Harsuko, R., & Alkhalifah, T. A. (2022). *StorSeismic: A New Paradigm in Deep Learning for Seismic Processing*. IEEE Transactions on Geoscience and Remote Sensing, 60, 1–15.
- Original Repository: [swag-kaust/storseismic](https://github.com/swag-kaust/storseismic)
