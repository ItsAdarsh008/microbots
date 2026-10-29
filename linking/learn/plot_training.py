"""Draw the training curves from policy_log.json (written by dagger.py).

    python linking/learn/plot_training.py              # -> media/linking_training.png

Blue: the loss on each batch (and its running average). Orange: how often the
network picks the same move as the rules on decisions it never trained on.
Shaded bands mark the DAgger rounds, where new data (the student driving)
comes in, which is why the loss jumps up at the start of each band. The
held-out decisions all come from the teacher's own runs, so agreement on them
drops a little in DAgger: the network is now also learning to recover from
its own mistakes, situations the teacher never gets into. What matters is
whether it builds (evaluate.py).
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--log", default=os.path.join(HERE, "policy_log.json"))
ap.add_argument("--out", default=os.path.join(ROOT, "media", "linking_training.png"))
a = ap.parse_args()

log = json.load(open(a.log))
loss = np.array(log["loss"])
ends = np.array(log["epoch_end"])
epoch_of_step = np.searchsorted(ends, np.arange(len(loss)), side="right") + np.arange(len(loss)) * 0.0
# x axis in epochs: step k sits inside its epoch in proportion
starts = np.concatenate([[0], ends[:-1]])
x = np.zeros(len(loss))
for e, (s0, s1) in enumerate(zip(starts, ends)):
    x[s0:s1] = e + (np.arange(s1 - s0) + 0.5) / (s1 - s0)
avg = np.convolve(loss, np.ones(25) / 25, mode="same")

fig, ax = plt.subplots(figsize=(9, 4.6))
ax2 = ax.twinx()
stages = log["stage"]
for st in sorted(set(stages)):
    if st.startswith("dagger"):
        es = [i for i, s in enumerate(stages) if s == st]
        ax.axvspan(es[0], es[-1] + 1, color="#eef1f6", zorder=0)
        ax.axvline(es[0], color="#b8c0cc", lw=1, ls="--", zorder=0)
        ax.text(es[0] + 0.15, 0.98, st.replace("dagger", "DAgger round"), transform=ax.get_xaxis_transform(),
                va="top", fontsize=9, color="#555")
ax.plot(x, loss, color="#a9c1df", lw=0.6)
ax.plot(x[12:-12], avg[12:-12], color="#2a5ea8", lw=2, label="training loss")
ax2.plot(np.arange(1, len(log["agree"]) + 1), np.array(log["agree"]) * 100, "o-", color="#d9622b", lw=2)
ax.set_xlabel("epoch")
ax.set_ylabel("cross-entropy loss", color="#2a5ea8")
ax2.set_ylabel("% same move as the rules (held out)", color="#d9622b")
lo, hi = min(log["agree"]) * 100, max(log["agree"]) * 100
ax2.set_ylim(lo - 4, hi + 4)
ax.set_title("Training the shared bot policy (262 → 128 → 128 → 19): imitation, then DAgger")
for s in ("top",):
    ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(a.out, dpi=150, facecolor="white")
print("saved", a.out)
