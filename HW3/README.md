# HW3 — Transformer is All You Need (Tiny Shakespeare)

Public course submission folder. **Code:** [COMS4995-Applied-Machine-Learning-HW/HW3](https://github.com/zz3231/COMS4995-Applied-Machine-Learning-HW/tree/main/HW3)

## Contents

| Item | Description |
|------|-------------|
| `HW3.ipynb` | Runnable notebook: data, BPE (vocab ≤ 500), from-scratch causal Transformer, training, plots, ablations |
| `run_hw3_train.py` | Optional headless script (same hyperparameters; env vars `HW3_EPOCHS`, etc.) |
| `requirements.txt` | Python dependencies |
| `report.tex` / `report.pdf` | Write-up for Gradescope (figures embedded) |
| `fig_01_*.png` … `fig_05_*.png` | Figures used by the report |
| `data/input.txt` | Tiny Shakespeare (optional cache if download fails) |
| `COLAB_SETUP.txt` | Notes for Google Colab GPU runs |

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook HW3.ipynb
```

## Rebuild the PDF

```bash
pdflatex report.tex
pdflatex report.tex
```

## Syncing to your GitHub clone

From the root of [COMS4995-Applied-Machine-Learning-HW](https://github.com/zz3231/COMS4995-Applied-Machine-Learning-HW):

```bash
rsync -a --delete /path/to/this/HW3/ ./HW3/
git add HW3
git status   # verify only intended files
git commit -m "Update HW3 submission"
git push origin main
```

Ensure the repository is **public** before the assignment deadline.
