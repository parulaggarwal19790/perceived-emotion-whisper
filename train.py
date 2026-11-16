"""Predict how a panel of listeners splits its votes over six emotions, from one speech clip.

Whisper audio encoders (general-purpose speech-recognition models, never trained on emotion) are
fine-tuned end to end on the training clips. Several encoders are trained one after another and
their predictions are averaged: different pretrained encoders make different mistakes, so their
average beats extra seeds of one model.

Per encoder:
  * only the encoder is kept (no decoder, no text is ever produced);
  * its final hidden states are pooled over the clip (mean and standard deviation), joined with
    two loudness features (log RMS and log peak), and mapped by a small MLP to six logits;
  * every layer is trained, with layer-wise learning-rate decay;
  * the loss is the Jensen-Shannon divergence between the predicted distribution and the
    listeners' vote shares, i.e. the model learns the whole split of votes, not one label.

At test time every clip is seen in three views (as recorded, 5% faster, 5% slower) and all
predictions are averaged.

Usage:
    python train.py DATA_DIR OUT_CSV
    python train.py DATA_DIR OUT_CSV --encoders openai/whisper-tiny --epochs 1   # quick check
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import wavfile
from transformers import WhisperFeatureExtractor, WhisperModel

EMOTIONS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
DEFAULT_ENCODERS = ["openai/whisper-large-v3", "openai/whisper-large",
                    "openai/whisper-medium", "openai/whisper-medium.en"]
SAMPLE_RATE = 16_000
MAX_LEN = 102_400   # 6.4 s; Whisper's 30 s window is cut down to what the clips need
HOP = 320           # audio samples per encoder frame


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_dir", help="folder with train.csv, test.csv and the audio they point to")
    p.add_argument("out_csv", help="where to write the predicted vote shares for test.csv")
    p.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5, help="learning rate of the top encoder layer")
    p.add_argument("--layer-decay", type=float, default=0.9, help="each lower layer learns this much slower")
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


ARGS = parse_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = dict(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(ARGS.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_wav(path):
    rate, x = wavfile.read(os.path.join(ARGS.data_dir, path))
    assert rate == SAMPLE_RATE and x.ndim == 1, f"{path}: expected 16 kHz mono"
    return x.astype(np.float32) / 32768


class Head(nn.Module):
    """Mean and std of the encoder output over the clip's frames, plus loudness, into an MLP."""

    def __init__(self, hidden, level_stats):
        super().__init__()
        self.register_buffer("level_mean", torch.tensor(level_stats[0], dtype=torch.float32))
        self.register_buffer("level_std", torch.tensor(level_stats[1], dtype=torch.float32))
        self.level = nn.Sequential(nn.Linear(2, 64), nn.GELU())  # louder clips draw anger votes
        self.mlp = nn.Sequential(nn.Linear(2 * hidden + 64, 512), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(512, len(EMOTIONS)))

    def forward(self, h, n_frames, wav, lengths):
        h = h.float()
        mask = (torch.arange(h.shape[1], device=h.device)[None] < n_frames[:, None]).float()[..., None]
        mean = (h * mask).sum(1) / mask.sum(1)
        std = (((h - mean[:, None]) ** 2 * mask).sum(1) / mask.sum(1) + 1e-5).sqrt()
        rms = ((wav ** 2).sum(1) / lengths).sqrt()
        level = torch.stack([torch.log10(rms + 1e-5), torch.log10(wav.abs().amax(1) + 1e-5)], 1)
        level = (level - self.level_mean) / self.level_std
        return self.mlp(torch.cat([mean, std, self.level(level)], 1))


class WhisperEmotion(nn.Module):
    def __init__(self, name, level_stats):
        super().__init__()
        self.enc = WhisperModel.from_pretrained(name).encoder.float()
        self.register_buffer("mel_fb", torch.tensor(
            WhisperFeatureExtractor.from_pretrained(name).mel_filters, dtype=torch.float32))
        n_pos = MAX_LEN // HOP
        self.enc.embed_positions = nn.Embedding.from_pretrained(
            self.enc.embed_positions.weight[:n_pos].clone(), freeze=True)
        self.enc.config.max_source_positions = n_pos
        self.head = Head(self.enc.config.d_model, level_stats)

    def forward(self, wav, lengths):
        wav = wav.float() * (torch.arange(wav.shape[1], device=wav.device)[None] < lengths[:, None])
        padded = F.pad(wav, (0, MAX_LEN - wav.shape[1]))
        # Whisper's log-mel front end, on the GPU, in full precision
        with torch.autocast(DEVICE.type, enabled=False):
            spec = torch.stft(padded, 400, 160, window=torch.hann_window(400, device=wav.device),
                              return_complex=True)
            mel = self.mel_fb.T @ (spec[..., :-1].abs() ** 2)
            log = mel.clamp(min=1e-10).log10()
            log = torch.maximum(log, log.amax(dim=(1, 2), keepdim=True) - 8.0)
            mel = (log + 4.0) / 4.0
        final = self.enc(mel, output_hidden_states=True).hidden_states[-1]
        return self.head(final, (lengths + HOP - 1) // HOP, wav, lengths)


def make_batch(wavs, idx):
    clips = [torch.from_numpy(wavs[i]) for i in idx]
    lengths = torch.tensor([len(x) for x in clips])
    out = torch.zeros(len(clips), int(lengths.max()))
    for j, x in enumerate(clips):
        out[j, :len(x)] = x
    return out.to(DEVICE), lengths.to(DEVICE)


@torch.no_grad()
def predict(model, wavs):
    model.eval()
    order = np.argsort([len(w) for w in wavs], kind="stable")
    out = np.zeros((len(wavs), len(EMOTIONS)))
    with torch.autocast(**AMP):
        for k in range(0, len(order), 16):
            idx = order[k:k + 16]
            x, n = make_batch(wavs, idx)
            out[idx] = torch.softmax(model(x, n).float(), 1).cpu().numpy()
    return out


def predict_views(model, wavs):
    """Average over three views of each clip: as recorded, 5% faster, 5% slower."""
    out = predict(model, wavs)
    for rate in (0.95, 1.05):
        view = [F.interpolate(torch.from_numpy(w)[None, None], size=min(MAX_LEN, int(len(w) / rate)),
                              mode="linear", align_corners=False)[0, 0].numpy() for w in wavs]
        out += predict(model, view)
    return out / 3


def jsd_loss(logits, target):
    """Jensen-Shannon divergence (bits) between softmax(logits) and the vote shares."""
    logq = F.log_softmax(logits.float(), 1)
    q = logq.exp()
    m = 0.5 * (q + target)
    js = 0.5 * (q * (logq - m.log())).sum(1) + 0.5 * (torch.xlogy(target, target) - target * m.log()).sum(1)
    return js.mean() / np.log(2)


def fine_tune(model, wavs, shares, seed):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    blocks = list(model.enc.layers)
    in_blocks = {id(p) for b in blocks for p in b.parameters()}
    # layer-wise learning-rate decay: the top block learns at lr, each block below 0.9x slower
    groups = [{"params": [p for p in model.enc.parameters() if p.requires_grad and id(p) not in in_blocks],
               "lr": ARGS.lr * ARGS.layer_decay ** len(blocks)}]
    for depth, block in enumerate(blocks):
        groups.append({"params": list(block.parameters()),
                       "lr": ARGS.lr * ARGS.layer_decay ** (len(blocks) - 1 - depth)})
    groups.append({"params": list(model.head.parameters()), "lr": ARGS.head_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)

    batch = min(ARGS.batch, len(wavs))
    steps = max(1, ARGS.epochs * (len(wavs) // batch))
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
    targets = torch.tensor(shares, dtype=torch.float32, device=DEVICE)

    for epoch in range(ARGS.epochs):
        model.train()
        perm = rng.permutation(len(wavs))
        # clips of similar length share a batch, so little padding is computed
        batches = []
        for k in range(0, len(perm), batch * 40):
            chunk = np.sort(perm[k:k + batch * 40])
            chunk = chunk[np.argsort([len(wavs[i]) for i in chunk], kind="stable")]
            batches += [chunk[j:j + batch] for j in range(0, len(chunk) - batch + 1, batch)]
        total = 0.0
        for b in rng.permutation(len(batches)):
            idx = batches[b]
            x, n = make_batch(wavs, idx)
            opt.zero_grad()
            # two half-batches with summed gradients: same update, half the activation memory,
            # so the large encoders fit on a 24 GB card
            half = max(1, len(idx) // 2)
            for part in (slice(0, half), slice(half, len(idx))):
                if part.start >= len(idx):
                    continue
                with torch.autocast(**AMP):
                    logits = model(x[part], n[part])
                loss = jsd_loss(logits, targets[idx[part]]) * (part.stop - part.start) / len(idx)
                loss.backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        print(f"  epoch {epoch}: train JSD {total / max(1, len(batches)):.4f}", flush=True)


def write_shares(ids, probs, path):
    """id + six decimals per row; the largest value absorbs rounding so every row sums to 1."""
    units = np.rint(probs / probs.sum(1, keepdims=True) * 1_000_000).astype(np.int64)
    units[np.arange(len(units)), units.argmax(1)] += 1_000_000 - units.sum(1)
    out = pd.DataFrame({"id": ids})
    for k, name in enumerate(EMOTIONS):
        out[name] = [f"{v // 1_000_000}.{v % 1_000_000:06d}" for v in units[:, k]]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out.to_csv(path, index=False, lineterminator="\n")


def main():
    train = pd.read_csv(os.path.join(ARGS.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(ARGS.data_dir, "test.csv"))
    shares = train[EMOTIONS].to_numpy(np.float64)
    train_wavs = [load_wav(f) for f in train.audio]
    test_wavs = [load_wav(f) for f in test.audio]
    print(f"{len(train_wavs)} training clips, {len(test_wavs)} test clips, device {DEVICE}", flush=True)

    levels = np.array([[np.log10(np.sqrt((w ** 2).mean()) + 1e-5), np.log10(np.abs(w).max() + 1e-5)]
                       for w in train_wavs])
    level_stats = (levels.mean(0), levels.std(0) + 1e-6)

    pred = np.zeros((len(test), len(EMOTIONS)))
    for k, name in enumerate(ARGS.encoders):
        print(f"model {k}: {name}", flush=True)
        model = WhisperEmotion(name, level_stats).to(DEVICE)
        fine_tune(model, train_wavs, shares, ARGS.seed + k)
        pred += predict_views(model, test_wavs) / len(ARGS.encoders)
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    write_shares(test.id, pred, ARGS.out_csv)
    print(f"wrote {len(test)} rows to {ARGS.out_csv}", flush=True)


if __name__ == "__main__":
    main()
