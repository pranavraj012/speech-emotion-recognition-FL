# Speech Emotion Recognition FL Overview

This project uses federated learning on HuBERT embeddings from the RAVDESS speech emotion dataset.

## 1. What data was identified

The dataset is the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).

We keep 6 emotion classes:
- neutral
- happy
- sad
- angry
- fearful
- disgusted

We exclude the other two RAVDESS classes:
- calm
- surprised

Why:
- calm is too close to neutral acoustically
- surprised is small and often overlaps with fearful or happy

The raw inputs are WAV audio clips from 24 actors. Each clip is transformed into a 1024-dimensional embedding using HuBERT-Large.

## 2. How the audio was transformed

The raw audio is not fed directly into the classifier.

Pipeline:
1. Load WAV audio.
2. Resample to 16 kHz if needed.
3. Convert stereo to mono.
4. Extract HuBERT-Large hidden states.
5. Mean-pool the final hidden state over time.
6. Save the resulting 1024-dim vector.

That gives us a fixed-length feature vector per clip.

## 3. Train/test splitting in federated learning

Yes, the data is split before client training.

The correct FL pattern is:
1. Create one global holdout test split from the full dataset.
2. Use only the remaining training split for client partitioning.
3. Partition that training split across clients.
4. Keep the test split untouched until evaluation.

In this codebase, the split is configurable with `--test-size`.

Current default:
- `--test-size 0.15`  => 85/15 train/test

If you want a standard 80/20 split, run:

```powershell
python run_flwr.py --num-clients 5 --rounds 10 --local-epochs 100 --lr 0.005 --test-size 0.2
```

Important FL note:
- The test set is global and shared for evaluation only.
- The training split is the only part that gets divided across clients.

## 4. Federated learning environment

Implementation:
- Flower federated learning
- FedAvg aggregation
- Local simulated clients on one machine
- gRPC-based training coordination
- PyTorch local model training

How it works:
1. Flower server starts and creates the global model.
2. Each client receives the current global parameters.
3. Each client trains locally on its own data shard.
4. Clients send model parameters back.
5. The server averages them using FedAvg.
6. The updated global model is written to `checkpoints_flwr/`.

This is a simulation of distributed FL, but everything runs locally.

## 5. Tech stack

- Python 3.13
- PyTorch
- TorchAudio
- Hugging Face Transformers
- Flower
- scikit-learn
- NumPy
- Joblib
- Streamlit
- streamlit-audiorecorder
- Pandas

## 6. Model being trained

The current training model is a small MLP:
- Input: 1024 HuBERT features
- Hidden layer: 256 units
- ReLU activation
- Dropout: 0.1
- Output: 6 emotion classes

Why this model:
- It is lightweight
- It trains quickly in federated settings
- It is more expressive than a single linear layer

## 7. Visuals to show

Recommended visuals for your report or presentation:
- Training accuracy vs round
- Training loss vs round
- Confusion matrix on the holdout test split
- Probability bars for a sample prediction
- Screenshot of the Streamlit tester

You can build these from the saved Flower history and checkpoint outputs.

## 8. Training commands

Prepare the scaler:

```powershell
python prepare_scaler.py
```

Run federated training:

```powershell
python run_flwr.py --num-clients 5 --rounds 10 --local-epochs 100 --lr 0.005
```

Run the Streamlit emotion tester:

```powershell
streamlit run streamlit_app.py
```

## 9. Result from the latest full run

The saved Flower history shows a final test accuracy of 94.64%, which clears the 85% target.

## 10. Practical summary

- If you want the current default, use the built-in split in the code.
- If you want an exact 80/20 split, pass `--test-size 0.2`.
- The training data is partitioned across clients after the global holdout split.
- The model is the small HuBERT-based MLP, not the old linear-only setup.
