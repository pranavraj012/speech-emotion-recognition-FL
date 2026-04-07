# extract_hubert.py
# Step 2: Extract HuBERT-Large embeddings from audio → embeddings.npz
# GPU-intensive — run once, ~10 minutes on GTX 1650
import torch, torchaudio, numpy as np, pandas as pd
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# Load HuBERT-Large (downloads ~1.3GB first time, cached after)
print('Loading HuBERT-Large...')
extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-large-ls960-ft')
model = HubertModel.from_pretrained('facebook/hubert-large-ls960-ft')
model = model.to(DEVICE).eval()
print('HuBERT loaded successfully!')

df = pd.read_csv('ravdess_6class.csv')
embeddings = []
labels = []
actors = []

with torch.no_grad():
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Extracting HuBERT embeddings'):
        waveform, sr = torchaudio.load(row['path'])
        # Resample to 16kHz (HuBERT expects 16kHz)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        waveform = waveform.squeeze().numpy()

        # Feature extraction (normalises amplitude, pads/truncates)
        inputs = extractor(waveform, sampling_rate=16000, return_tensors='pt', padding=True)
        input_values = inputs['input_values'].to(DEVICE)

        # Forward pass — get last hidden state
        outputs = model(input_values)
        # Mean-pool across time dimension: (1, T, 1024) -> (1024,)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(embedding)
        labels.append(row['label'])
        actors.append(row['actor'])

X = np.array(embeddings)  # shape: (1152, 1024)
y = np.array(labels)      # shape: (1152,)
a = np.array(actors)      # shape: (1152,)

np.savez('embeddings.npz', X=X, y=y, actors=a)
print(f'\nSaved embeddings.npz — X shape: {X.shape}')
print(f'Labels shape: {y.shape}, Actors shape: {a.shape}')
