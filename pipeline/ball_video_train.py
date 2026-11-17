"""Fine-tune WASB on our match: start from the AdaBN weights (held-out fold excluded), train on the
tile cache with synthetic occlusion, keep the best checkpoint on a chunk-held-out validation set.

    caffeinate -dimsu .venv-video/bin/python -c "from pipeline import ball_video_train as t; t.train()"

Loss and target follow WASB: weighted BCE (focal-style) on a binary disc of radius 2.5 around the ball.
"""

import os
import time

import numpy as np
import pandas as pd
import torch

from pipeline.common import CACHE_DIR

# Params
EPOCHS, LR, BATCH = 3, 1e-4, 2
ACCUM = 4                           # steps per update: effective batch 8 (batch 8 runs out of MPS memory)
VAL_SHARE = 0.1
VAL_CHUNK_FRAMES = 10 * 60 * 25     # validation takes whole 10-minute stretches of video
VAL_EVERY = 0.5                     # epochs between validations
HIT_PX = 4                          # model px (5 source px): argmax this close to the ball is a hit
CONFIDENT = 0.5
LOG_EVERY = 200                     # steps between progress lines, so a stall shows within minutes
FINETUNED_PATH = os.path.join(CACHE_DIR, "ball_video", "finetuned.pt")


def wbce_loss(logits, target):
    """WASB's weighted binary cross-entropy on heatmap logits."""
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    return torch.mean(-((1 - p) ** 2 * target * torch.log(p) + p ** 2 * (1 - target) * torch.log(1 - p)))


def train_mode(model):
    """Train mode with BatchNorm frozen at the AdaBN statistics: batches of 2 give noisy batch stats."""
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    return model


def split_validation(samples, share=VAL_SHARE, seed=0):
    """(train, val): val is whole VAL_CHUNK_FRAMES stretches, about `share` of the samples."""
    chunk = samples["frame"] // VAL_CHUNK_FRAMES
    order = np.random.default_rng(seed).permutation(np.unique(chunk))
    sizes = chunk.value_counts()
    picked, n = [], 0
    for c in order:
        if n >= share * len(samples):
            break
        picked.append(c)
        n += sizes[c]
    is_val = chunk.isin(picked)
    return samples[~is_val].reset_index(drop=True), samples[is_val].reset_index(drop=True)


def tile_metrics(model, loader, device):
    """pos_hit: share of ball tiles whose middle-frame argmax is within HIT_PX of the ball with peak >=
    CONFIDENT. neg_fire: share of ball-free tiles with any peak >= CONFIDENT."""
    model.eval()
    hits = pos = fires = neg = 0
    with torch.no_grad():
        for x, y in loader:
            hm = torch.sigmoid(model(x.to(device))[0])[:, 1].cpu().numpy()
            target = y[:, 1].numpy()
            for h, t in zip(hm, target):
                if t.sum() > 0:
                    ty, tx = np.argwhere(t > 0).mean(axis=0)
                    py, px = np.unravel_index(np.argmax(h), h.shape)
                    hits += h.max() >= CONFIDENT and np.hypot(px - tx, py - ty) <= HIT_PX
                    pos += 1
                else:
                    fires += h.max() >= CONFIDENT
                    neg += 1
    return {"pos_hit": hits / max(pos, 1), "neg_fire": fires / max(neg, 1)}


def train(epochs=EPOCHS, lr=LR, batch=BATCH, accum=ACCUM, seed=0, val_every=VAL_EVERY, limit=None, out_path=FINETUNED_PATH):
    """Fine-tune from the held-out AdaBN weights; returns the path of the best checkpoint."""
    from pipeline import ball_video as bv, wasb
    from pipeline.ball_video_data import CACHE_DIR as TILES, TileDataset
    from pipeline.occlusion import load_donor_bank

    torch.manual_seed(seed)
    samples = pd.read_parquet(os.path.join(TILES, "samples.parquet"))
    train_s, val_s = split_validation(samples, seed=seed)
    if limit:
        train_s = train_s.sample(min(limit, len(train_s)), random_state=seed)
        val_s = val_s.sample(min(max(limit // 10, 50), len(val_s)), random_state=seed)
    train_ds = TileDataset(train_s, TILES, donors=load_donor_bank(), seed=seed)
    val_ds = TileDataset(val_s, TILES, donors=None, jitter=False, seed=seed)
    gen = torch.Generator().manual_seed(seed)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True, generator=gen)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch)

    device = wasb.pick_device()
    model = wasb.load_model(bv.ADABN_HELDOUT_PATH, device)
    opt = torch.optim.Adam(model.parameters(), lr)
    start = tile_metrics(model, val_dl, device)
    print(f"start: {start}", flush=True)
    best = start["pos_hit"] - start["neg_fire"]
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out_path)

    every = max(accum, int(val_every * len(train_dl)) // accum * accum)
    step, losses, t0 = 0, [], time.time()
    opt.zero_grad()
    for epoch in range(epochs):
        for x, y in train_dl:
            train_mode(model)
            loss = wbce_loss(model(x.to(device))[0], y.to(device))
            (loss / accum).backward()
            losses.append(loss.item())
            step += 1
            if step % accum == 0:
                opt.step()
                opt.zero_grad()
            if step % LOG_EVERY == 0:
                print(f"epoch {epoch} step {step}/{epochs * len(train_dl)} loss {np.mean(losses[-LOG_EVERY:]):.5f} "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % every == 0:
                m = tile_metrics(model, val_dl, device)
                if device == "mps":
                    torch.mps.empty_cache()
                score = m["pos_hit"] - m["neg_fire"]
                print(f"epoch {epoch} step {step} loss {np.mean(losses[-every:]):.5f} {m} "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
                if score > best:
                    best = score
                    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out_path)
    return out_path
