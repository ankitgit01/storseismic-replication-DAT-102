# StorSeismic Replication (DAT-102 Course Project)

Replication of **StorSeismic** (*Harsuko & Alkhalifah, 2022*), a BERT-based self-supervised pre-training and fine-tuning framework for seismic data processing, built as a course project for **DAT-102** at IIT Roorkee.

Original repository: [swag-kaust/storseismic](https://github.com/swag-kaust/storseismic)

---

## Overview

StorSeismic adapts the BERT masked language modeling architecture (`transformers.BertForMaskedLM`) to continuous seismic shot gathers by treating each seismic trace as a token and using Pre-LayerNorm (`Pre-LN`) transformer blocks:

1. **Data Preparation (`nb0_1`, `nb0_2`)**: Generates masked pre-training shot gathers and downstream fine-tuning datasets from the [SNIST](https://github.com/LukasMosser/SNIST) synthetic seismic benchmark.
2. **Self-Supervised Pre-Training (`nb1_pretraining.ipynb`)**: Trains a 4-layer Pre-LN BERT model with `RAdam` and MSE loss on masked traces to reconstruct missing seismic waveforms.
3. **Downstream Fine-Tuning (`nb2_1`, `nb2_2`)**: Fine-tunes the pre-trained backbone for **seismic denoising** and **1D velocity estimation**.

---

## Repository Structure

```text
├── storseismic/
│   ├── modules.py          # Custom BertEmbeddings, BertSelfAttention, PreLNBert*, and task heads
│   ├── train.py            # Pre-training and fine-tuning loops
│   ├── utils.py            # SSDataset and SNISTMLM PyTorch Dataset classes
│   └── pytorchtools.py     # EarlyStopping utility
├── nb0_1_data_prep_pretrain.ipynb
├── nb0_2_data_prep_finetune.ipynb
├── nb1_pretraining.ipynb
├── nb2_1_finetuning_denoising.ipynb
├── nb2_2_finetuning_velpred.ipynb
├── data/                   # Generated pre-train and fine-tune .pt datasets
└── results/                # Saved model checkpoints and training/validation loss arrays
```

---

## Environment Setup (Windows + CUDA)

This replication was run locally in **VS Code** using **Python 3.10** (required for prebuilt `tokenizers==0.12.1` wheels compatible with `transformers==4.20.1`) and **PyTorch 2.7.1 + CUDA 12.8** for RTX 50-series (Blackwell) GPU support.

### 1. Clone the repository

```powershell
git clone https://github.com/swag-kaust/storseismic.git
cd storseismic
```

### 2. Create a Python 3.10 virtual environment with `uv`

```powershell
uv venv --python 3.10 .venv
.\.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
uv pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers==4.20.1 numpy==1.26.4 pandas matplotlib tqdm scipy scikit-learn ipykernel
uv pip install git+https://github.com/LiyuanLucasLiu/RAdam
uv pip install git+https://github.com/LukasMosser/SNIST.git
```

---

## Reproducing the Workflow

Run the notebooks sequentially using the `.venv` (Python 3.10) kernel:

1. `nb0_1_data_prep_pretrain.ipynb` — Creates `./data/pretrain/train_data.pt` and `test_data.pt`.
2. `nb0_2_data_prep_finetune.ipynb` — Creates `./data/finetune/` datasets.
3. `nb1_pretraining.ipynb` — Pre-trains the StorSeismic model and saves checkpoints to `./results/pretrain/`.
4. `nb2_1_finetuning_denoising.ipynb` — Fine-tunes the pre-trained model for seismic denoising.
5. `nb2_2_finetuning_velpred.ipynb` — Fine-tunes the pre-trained model for velocity prediction.

### Compatibility Notes (PyTorch 2.6+ & VS Code)

Two minimal updates were applied to the original notebooks for modern PyTorch and VS Code execution:
- **`torch.load(..., weights_only=False)`**: Required in PyTorch 2.6+ when loading pickled `SSDataset` and full `BertForMaskedLM` checkpoint files (`train_data.pt`, `test_data.pt`, `starting_model.pt`, `model.pt`).
- **`%matplotlib inline`**: Replaced `%matplotlib notebook` in the training loss plot cells for compatibility with VS Code's notebook renderer.

---

## References

- Harsuko, R., & Alkhalifah, T. A. (2022). *StorSeismic: A new paradigm in deep learning for seismic processing.* IEEE Transactions on Geoscience and Remote Sensing, 60, 1–15.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). *BERT: Pre-training of deep bidirectional transformers for language understanding.* arXiv:1810.04805.
- Xiong, R., et al. (2020). *On layer normalization in the transformer architecture.* ICML, 10524–10533.
