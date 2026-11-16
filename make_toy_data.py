"""Write a tiny synthetic dataset in the expected format, to check the pipeline runs end to end.

The clips are tones and noise, not speech, so the scores mean nothing. Use it to try the code
before pointing it at real data.

Usage:
    python make_toy_data.py toy_data
    python train.py toy_data toy_pred.csv --encoders openai/whisper-tiny --epochs 1
    python evaluate.py toy_data/test_labels.csv toy_pred.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.io import wavfile

EMOTIONS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
RATE = 16_000


def main(out_dir, n_train=48, n_test=16, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(out_dir, "audio"), exist_ok=True)
    rows = []
    for i in range(n_train + n_test):
        seconds = rng.uniform(1.5, 4.0)
        t = np.arange(int(seconds * RATE)) / RATE
        pitch = rng.uniform(90, 300)
        loud = rng.uniform(0.05, 0.6)
        wav = loud * np.sin(2 * np.pi * pitch * t) + rng.normal(0, 0.02, t.size)
        name = f"audio/clip_{i:04d}.wav"
        wavfile.write(os.path.join(out_dir, name), RATE, (np.clip(wav, -1, 1) * 32767).astype(np.int16))
        # votes lean towards anger for loud clips and sad for low ones, so there is something to learn
        logits = rng.normal(0, 0.5, len(EMOTIONS))
        logits[0] += 4 * loud
        logits[5] += 300 / pitch
        logits[4] += 1.0
        votes = rng.multinomial(10, np.exp(logits) / np.exp(logits).sum())
        rows.append({"id": f"c{i:04d}", "audio": name, **dict(zip(EMOTIONS, votes / votes.sum()))})
    df = pd.DataFrame(rows)
    df.iloc[:n_train].to_csv(os.path.join(out_dir, "train.csv"), index=False)
    df.iloc[n_train:][["id", "audio"]].to_csv(os.path.join(out_dir, "test.csv"), index=False)
    df.iloc[n_train:].drop(columns="audio").to_csv(os.path.join(out_dir, "test_labels.csv"), index=False)
    print(f"wrote {n_train} training and {n_test} test clips to {out_dir}/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "toy_data")
