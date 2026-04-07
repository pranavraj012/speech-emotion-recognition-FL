
# Privacy-Preserving Speech Emotion Recognition (Federated)

This repo contains a federated learning simulation for speech emotion recognition using HuBERT embeddings and Flower.

The project goal is simple: keep raw audio local, share model updates only, and still hit strong classification accuracy.

---

## What is inside

- Flower-based FL training loop (`flwr_simulation.py`, `run_flwr.py`)
- HuBERT-based inference (`emotion_inference.py`, `predict.py`)
- Streamlit demo app for upload + microphone testing (`streamlit_app.py`)
- Visual and PPT generation scripts (`generate_visuals.py`, `build_ppt.py`)
- Technical writeups (`FL_TECHNICAL_REPORT.md`, `FL_OVERVIEW.md`)

---

## Model and setup at a glance

- Features: HuBERT-Large embeddings (1024-dim)
- Classifier: MLP (1024 -> 256 -> 6)
- FL strategy: FedAvg in Flower
- Dataset: RAVDESS (6 emotions)
- Split: global holdout split first, then training partition across clients

---

## Quick start (replication)

### 1. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

### 3. Prepare data artifacts

Run these after you have the RAVDESS audio available locally:

```powershell
python parse_ravdess.py
python extract_hubert.py
python prepare_scaler.py
```

---

### 4. Train federated model (example: 80/20 split)

```powershell
python run_flwr.py --num-clients 5 --rounds 10 --local-epochs 100 --lr 0.005 --batch-size 32 --hidden-dim 256 --test-size 0.2 --epsilon 0.0
```

---

### 5. Run prediction from a file

```powershell
python predict.py path\to\audio.wav
```

---

### 6. Run Streamlit demo (upload or record voice)

```powershell
streamlit run streamlit_app.py
```

---

### 7. Generate visuals and PPT

```powershell
python generate_visuals.py
python build_ppt.py
```

Outputs will be generated in `presentation_assets/` and as `SER_FL_Presentation.pptx`.

---

## Notes

* If CUDA is available, training and inference will use GPU.
* `--epsilon 0.0` disables output-perturbation noise for maximum accuracy.
* Increase epsilon only if you intentionally want stronger privacy noise with a likely accuracy tradeoff.

---

## Minimal file flow

1. Build dataset metadata and embeddings
2. Train with Flower
3. Evaluate via generated visuals and metrics history
4. Demo using Streamlit

---

## First-time setup

If you are cloning this repo for the first time, start with the Quick Start section and run commands in the same order.


