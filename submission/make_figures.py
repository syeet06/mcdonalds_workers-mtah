"""Figures for the technical document (submission/figures/), from our own pipeline outputs and IDSSE
results only. No video frames and no positions, so nothing here reconstructs the match data.

    .venv-video/bin/python -m submission.make_figures   # matplotlib is in the video environment
"""
import os, sys, numpy as np, pandas as pd
try:
    import matplotlib
except ModuleNotFoundError:
    sys.exit("matplotlib is missing: run this with the video environment, "
             ".venv-video/bin/python -m submission.make_figures")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pipeline import decoder as dc
from pipeline.common import load_public_gt, load_stage, load_tracking, half_of_frame, runs, FPS

OUT = "submission/figures"
os.makedirs(OUT, exist_ok=True)
COL = {"H": "#3b82f6", "A": "#ef4444", "D": "#9ca3af", "X": "#e5e7eb"}
INK, MUTED, ACCENT = "#1f2937", "#6b7280", "#111827"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
                     "axes.titlesize": 11, "figure.dpi": 150})
PUB = (11185, 18685)

from pipeline import submission as sub
c = sub.build_components()
lo, video, dead, ours = c["log_odds"], c["video"], c["dead"], c["labels"]
half = half_of_frame()
gt = load_public_gt()


def strip(ax, labels, a, b, y, h=0.8):
    for v, s, e in runs(labels[a:b]):
        ax.add_patch(plt.Rectangle(((s + a) / FPS / 60 - a / FPS / 60, y), (e - s) / FPS / 60, h, color=COL[str(v)], lw=0))


# 1. Public window, label strips -------------------------------------------------------------
base = pd.read_csv("evaluation/results/baseline_smooth50.csv").label.values.astype(str)
fig, ax = plt.subplots(figsize=(9, 2.2))
rows = [("Public ground truth", gt), ("Our pipeline (own labels)", ours), ("Baseline smooth50", base)]
for i, (name, lab) in enumerate(rows):
    strip(ax, lab, *PUB, y=len(rows) - 1 - i)
    n = sum(1 for v, s, e in runs(lab[PUB[0]:PUB[1]]) if v in "HA")
    ax.text(-0.08, len(rows) - 1 - i + 0.4, f"{name}  ({n} possessions)", ha="right", va="center", color=INK)
ax.set_xlim(0, (PUB[1] - PUB[0]) / FPS / 60); ax.set_ylim(-0.1, len(rows))
ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("minutes into the public window (video frames 11185-18684)")
ax.legend(handles=[Patch(color=COL[k], label=t) for k, t in (("H", "Barcelona (H)"), ("A", "Athletic (A)"), ("D", "dead ball (D)"))],
          loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
fig.savefig(f"{OUT}/public_window_strips.png", bbox_inches="tight"); plt.close(fig)

# 2. Switch cost: IDSSE sweep + possessions per live minute on our match ---------------------
sm = pd.read_csv("pipeline/model/results/smoothing.csv")
vit = sm[sm.method.str.startswith("viterbi")]
rates = []
live_min = ((half > 0) & ~dead).sum() / FPS / 60
for p in vit.p_switch:
    lab = np.full(len(lo), "X", dtype="<U1")
    score = lo + video
    for v, s, e in runs(half):
        if v:
            seg = np.where(dead[s:e], 0.0, score[s:e])
            lab[s:e] = dc.LETTERS[dc.viterbi(dc.emissions(seg, dead[s:e]), p)]
    rates.append(sum(1 for v, s, e in runs(lab) if v in "HA") / live_min)
g = gt[PUB[0]:PUB[1]]
gt_rate = sum(1 for v, s, e in runs(g) if v in "HA") / ((g != "D").sum() / FPS / 60)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.2))
keep = vit.p_switch >= 1
a1.plot(vit.p_switch[keep], vit.spell_f1_mean[keep], "o-", color=COL["H"], ms=4)
a1.axvline(6, color=MUTED, ls=":"); a1.text(6.6, 0.48, "best on IDSSE: 6", color=MUTED, fontsize=9)
a1.set_xscale("log"); a1.set_xticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"]); a1.set_xlabel("switch cost p_switch"); a1.set_ylabel("spellF1 (7 IDSSE matches, out of fold)")
a1.set_title("IDSSE prefers short possessions...")
rk = np.array(rates)[keep.to_numpy()]
a2.plot(vit.p_switch[keep], rk, "o-", color=COL["A"], ms=4)
a2.axhline(gt_rate, color=INK, ls="--", lw=1); a2.text(1.05, gt_rate - 0.7, f"public ground truth: {gt_rate:.1f} per live min", color=INK, fontsize=9)
a2.axvline(24, color=MUTED, ls=":"); a2.text(26, 7.5, "ours: 24", color=MUTED, fontsize=9)
a2.set_xscale("log"); a2.set_xticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"]); a2.set_ylim(2, 9); a2.set_xlabel("switch cost p_switch"); a2.set_ylabel("possessions per live minute, our match")
a2.set_title("...this competition's labels are coarser")
fig.tight_layout(); fig.savefig(f"{OUT}/switch_cost.png", bbox_inches="tight"); plt.close(fig)
print("rates", dict(zip(vit.p_switch, np.round(rates, 2))), "gt", round(gt_rate, 2))

# 3. Training data: clean vs degraded vs both --------------------------------------------------
tr = pd.read_csv("pipeline/model/results/train_six_tune_seventh.csv")
fig, ax = plt.subplots(figsize=(6, 3))
x = np.arange(len(tr)); w = 0.38
ax.bar(x - w / 2, tr.deg_acc_ball_seen, w, color=COL["H"], label="ball tracked")
ax.bar(x + w / 2, tr.deg_acc_ball_missing, w, color=COL["A"], label="ball missing")
for i, r in tr.iterrows():
    ax.text(i + w / 2, r.deg_acc_ball_missing + 0.003, f"{r.deg_acc_ball_missing:.3f}", ha="center", fontsize=8, color=INK)
    ax.text(i - w / 2, r.deg_acc_ball_seen + 0.003, f"{r.deg_acc_ball_seen:.3f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(x, [f"trained on {o}" for o in tr.option]); ax.set_ylim(0.8, 0.94)
ax.set_ylabel("frame accuracy, held-out match\n(degraded to look like ours)")
ax.legend(frameon=False, loc="upper left", ncol=2)
ax.set_title("Training on degraded copies teaches the model to cope without the ball")
fig.tight_layout(); fig.savefig(f"{OUT}/training_data.png", bbox_inches="tight"); plt.close(fig)

# 4. Dead mask stages ---------------------------------------------------------------------------
st = pd.DataFrame([("Stage 1 vote", 0.951, 0.821, 0.881), ("+ set-piece anchors", 0.951, 0.822, 0.882),
                   ("+ broadcast cut-aways", 0.943, 0.959, 0.951), ("+ ball over a line", 0.943, 0.975, 0.959),
                   ("+ whistled starts", 0.966, 0.961, 0.963), ("+ quiet players", 0.946, 0.984, 0.965),
                   ("Final mask", 0.948, 0.984, 0.966)], columns=["step", "precision", "recall", "f1"])
fig, ax = plt.subplots(figsize=(8, 3))
x = np.arange(len(st))
ax.plot(x, st.precision, "o-", color=COL["H"], label="precision", ms=4)
ax.plot(x, st.recall, "o-", color=COL["A"], label="recall", ms=4)
ax.plot(x, st.f1, "s-", color=INK, label="dead-ball F1", ms=4, lw=2)
ax.set_xticks(x, st.step, rotation=20, ha="right"); ax.set_ylim(0.8, 1.0)
ax.set_ylabel("vs public ground truth"); ax.legend(frameon=False, ncol=3, loc="lower right")
ax.set_title("Dead-ball mask, step by step (public 5 minutes, 4 stoppages)")
fig.tight_layout(); fig.savefig(f"{OUT}/dead_mask_steps.png", bbox_inches="tight"); plt.close(fig)

# 5. White pixels under the tracked ball ---------------------------------------------------------
white = load_stage("ball_seen")["white"]
wv = white[(half > 0) & (white >= 0)]
fig, ax = plt.subplots(figsize=(7, 2.8))
bins = np.arange(0, 121, 2)
ax.hist(np.clip(wv, 0, 120), bins=bins, color=COL["H"], alpha=0.85)
ax.axvline(6, color=INK, ls="--", lw=1)
ax.text(40, 3000, f"left of the dashed line (< 6 px): nothing white\nin the square, {(wv < 6).mean():.0%} of tracked graded frames", fontsize=9, color=INK)
ax.set_yscale("log"); ax.set_xlabel("white pixels in the 34 px square around the tracked ball (clipped at 120)")
ax.set_ylabel("tracked graded frames")
ax.set_title("Is there a ball under the tracked ball?")
fig.tight_layout(); fig.savefig(f"{OUT}/ball_square_white.png", bbox_inches="tight"); plt.close(fig)

# 6. Viewer-style signal panel around a public stoppage -------------------------------------------
from pipeline.dead_refine import ball_speed
from pipeline.whistle import whistle_onsets
a, b = 15600, 16600
t = (np.arange(a, b) - a) / FPS
trk = load_tracking()
bs = ball_speed(load_stage("ball_filled")["ball"].astype(float))[a:b]
ps = trk["player_speed"][a:b].astype(float); ps = np.where(ps > 15, np.nan, ps)
mean_ps = np.nanmean(ps, axis=1)
task4 = load_stage("dead_ball")["dead"]
wh = [f for f in whistle_onsets(half) if a <= f < b]
fig, axs = plt.subplots(4, 1, figsize=(9, 5), sharex=True, gridspec_kw={"height_ratios": [1.4, 1.4, 0.5, 1.2]})
axs[0].plot(t, bs, color=ACCENT, lw=1); axs[0].set_ylabel("ball m/s"); axs[0].set_ylim(0, 25)
axs[1].plot(t, mean_ps, color=COL["A"], lw=1); axs[1].set_ylabel("players'\nmean m/s")
axs[1].axhline(1.0, color=MUTED, ls=":", lw=1)
for f in wh:
    for k in (0, 1, 2):
        axs[k].axvline((f - a) / FPS, color="#f59e0b", lw=1.5)
axs[2].text(0.01, 0.4, "whistle onsets (audio)", transform=axs[2].transAxes, color="#b45309", fontsize=9)
axs[2].set_yticks([])
for i, (name, lab) in enumerate([("ground truth", gt), ("final mask + decoder", ours),
                                 ("Task 4 vote only", np.where(task4, "D", "X"))]):
    for v, s, e in runs(lab[a:b]):
        axs[3].add_patch(plt.Rectangle((s / FPS, 2 - i), (e - s) / FPS, 0.8, color=COL[str(v)], lw=0))
    axs[3].text(-0.5, 2 - i + 0.4, name, ha="right", va="center", fontsize=9, color=INK)
axs[3].set_ylim(-0.1, 3); axs[3].set_yticks([]); axs[3].set_xlim(0, (b - a) / FPS)
axs[3].set_xlabel("seconds (video frames 15600-16600, inside the public window)")
for ax in axs[2:]: ax.spines["left"].set_visible(False)
axs[0].set_title("What the dead-ball steps see around a stoppage (the viewer's dead-mask band)")
fig.tight_layout(); fig.savefig(f"{OUT}/stoppage_signals.png", bbox_inches="tight"); plt.close(fig)
print("whistles in panel", wh)
print("done")
