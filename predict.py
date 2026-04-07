import os
import sys

import pandas as pd

from emotion_inference import EMOTION_NAMES, predict_audio


if __name__ == "__main__":
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "test_input.wav"
    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found.")
        sys.exit(1)

    print(f"Processing: {audio_path}")
    result = predict_audio(audio_path)

    print(f"\n*** Predicted emotion: {result['emotion'].upper()} ***\n")
    for emotion, probability in zip(EMOTION_NAMES, result["probabilities"]):
        bar = "|" * int(probability * 20)
        print(f"  {emotion:10s} {bar} {probability * 100:.1f}%")

    print(f"\nCheckpoint: {result['checkpoint_path']}")
    print(f"Device: {result['device']}")
