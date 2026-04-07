# parse_ravdess.py
# Step 1: Download RAVDESS via kagglehub + parse filenames → CSV (6 emotions)
import os, glob, pandas as pd
import kagglehub

# Download dataset via kagglehub
print("Downloading RAVDESS dataset via kagglehub...")
dataset_path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
print(f"Dataset downloaded to: {dataset_path}")

# Emotions we KEEP (drop calm=2, surprised=8)
KEEP = {1: 'neutral', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful', 7: 'disgusted'}

records = []
for wav in glob.glob(os.path.join(dataset_path, '**/*.wav'), recursive=True):
    parts = os.path.basename(wav).replace('.wav', '').split('-')
    modality, emotion_id, actor_id = int(parts[0]), int(parts[2]), int(parts[6])
    if modality != 3: continue         # audio-only
    if emotion_id not in KEEP: continue  # drop calm, surprised
    records.append({
        'path': wav,
        'emotion': KEEP[emotion_id],
        'label': list(KEEP.keys()).index(emotion_id),  # 0-5
        'actor': actor_id,
    })

df = pd.DataFrame(records)
print("\nEmotion distribution:")
print(df.groupby('emotion').size())
print(f'\nTotal: {len(df)} samples, {df.actor.nunique()} actors')
df.to_csv('ravdess_6class.csv', index=False)
print("Saved ravdess_6class.csv")
