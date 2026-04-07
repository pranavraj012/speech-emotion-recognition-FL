import glob
import os
from functools import lru_cache
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor

EMOTION_NAMES = ["neutral", "happy", "sad", "angry", "fearful", "disgusted"]
DEFAULT_CHECKPOINT_DIR = "checkpoints_flwr"
DEFAULT_SCALER_PATH = "scaler.pkl"
DEFAULT_HUBERT_MODEL = "facebook/hubert-large-ls960-ft"


@lru_cache(maxsize=1)
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def load_scaler(scaler_path: str = DEFAULT_SCALER_PATH):
    return joblib.load(scaler_path)


@lru_cache(maxsize=1)
def load_hubert_bundle(model_name: str = DEFAULT_HUBERT_MODEL):
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name).to(get_device()).eval()
    return extractor, model


def latest_checkpoint_path(checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR) -> Optional[str]:
    direct_path = os.path.join(checkpoint_dir, "global_round_10.pkl")
    if os.path.exists(direct_path):
        return direct_path

    candidates = glob.glob(os.path.join(checkpoint_dir, "global_round_*.pkl"))
    if not candidates:
        return None
    return sorted(candidates, key=os.path.getmtime)[-1]


def build_model(hidden_dim: int, input_dim: int = 1024, num_classes: int = 6) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(hidden_dim, num_classes),
    )


def load_flower_model(checkpoint_path: Optional[str] = None, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR):
    path = checkpoint_path or latest_checkpoint_path(checkpoint_dir)
    if not path:
        raise FileNotFoundError(f"No Flower checkpoint found in {checkpoint_dir}")

    checkpoint = joblib.load(path)
    if "parameters" in checkpoint:
        parameters = checkpoint["parameters"]
        hidden_dim = int(parameters[0].shape[0])
        model = build_model(hidden_dim)
        state_keys = list(model.state_dict().keys())
        state_dict = {key: torch.tensor(value) for key, value in zip(state_keys, parameters)}
        model.load_state_dict(state_dict, strict=True)
    else:
        model = nn.Linear(1024, 6)
        model.weight.data = torch.tensor(checkpoint["weights"]).float()
        model.bias.data = torch.tensor(checkpoint["bias"]).float()
        hidden_dim = 0

    model.to(get_device()).eval()
    return model, path, hidden_dim


def embed_audio(audio_path: str) -> np.ndarray:
    extractor, hubert = load_hubert_bundle()
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    waveform = waveform.squeeze().numpy()

    inputs = extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = hubert(inputs["input_values"].to(get_device()))
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def predict_audio(audio_path: str, checkpoint_path: Optional[str] = None, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR) -> Dict[str, object]:
    scaler = load_scaler()
    model, resolved_checkpoint, hidden_dim = load_flower_model(checkpoint_path=checkpoint_path, checkpoint_dir=checkpoint_dir)

    embedding = embed_audio(audio_path).reshape(1, -1)
    embedding_scaled = scaler.transform(embedding)

    with torch.no_grad():
        input_tensor = torch.tensor(embedding_scaled, dtype=torch.float32).to(get_device())
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        prediction = int(np.argmax(probabilities))

    return {
        "prediction": prediction,
        "emotion": EMOTION_NAMES[prediction],
        "probabilities": probabilities,
        "checkpoint_path": resolved_checkpoint,
        "device": get_device(),
        "hidden_dim": hidden_dim,
    }
