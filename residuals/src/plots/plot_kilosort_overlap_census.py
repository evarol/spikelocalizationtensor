import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS = os.path.join(
    HERE, "..", "..", "runs", "dataset1_p1", "kilosort_overlap_census"
)
OUT = os.path.join(HERE, "..", "..", "out", "kilosort_overlap_census")
os.makedirs(OUT, exist_ok=True)

rej = json.load(open(os.path.join(CENSUS, "rejection_census.json")))
good_time_recall = json.load(open(os.path.join(CENSUS, "good_time_recall.json")))
rows = rej["rows"]
by_run = {r["run"]: r for r in rows}
for r in rows:
    r["recall_good"] = good_time_recall.get(r["run"], r["good_covered_accepted"])


def trimmed_parse(name):
    parts = name.split("_")
    if "trimmed" not in parts:
        return None
    rule = parts[3]
    f0 = int(parts[4][2:]) / 100.0
    return rule, f0


RULES = ["min", "mean", "kofn", "flat"]
RULE_COLORS = {"min": "#1f77b4", "mean": "#d62728", "kofn": "#2ca02c", "flat": "#9467bd"}
RULE_LABELS = {"min": "min (worst channel)", "mean": "mean channel", "kofn": "k-of-n (7/8)", "flat": "flat bar"}

fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor="white")
ceiling = np.mean([r["good_covered_by_proposals"] for r in rows if r["run"].startswith("0019_allchannel_trimmed")])
for rule in RULES:
    pts = sorted(
        (trimmed_parse(r["run"])[1], r["recall_good"])
        for r in rows
        if trimmed_parse(r["run"]) and trimmed_parse(r["run"])[0] == rule
    )
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, "o-", color=RULE_COLORS[rule], label=RULE_LABELS[rule], lw=2, ms=5)
ax.axhline(ceiling, color="k", ls="--", lw=1.2, label="proposal ceiling (~0.95)")
ax.axhline(by_run["0019_allchannel_pass3_round1_fraction20_step10_fitted8"]["recall_good"],
           color="#ff7f0e", ls=":", lw=1.5, label="untrimmed 20% bar (0.48)")
ax.set_xlabel("base acceptance bar $f_0$")
ax.set_ylabel("Kilosort good-spike time coverage")
ax.set_title("Under-detection is the acceptance bar: trimmed Q32 $f_0$ sweep vs Kilosort good spikes")
ax.set_ylim(0, 1.02)
ax.legend(fontsize=8, loc="lower left")
fig.savefig(os.path.join(OUT, "f0_sweep_coverage.png"), dpi=800, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 5.4), facecolor="white")
for fam, mask_fn, color, marker in [
    ("0019 threshold detector", lambda n: n.startswith("0019"), "#1f77b4", "o"),
    ("024 convolving detector", lambda n: n.startswith("024"), "#d62728", "^"),
]:
    xs = [by_run[r["run"]]["n_events"] for r in rows if mask_fn(r["run"])]
    ys = [r["recall_good"] for r in rows if mask_fn(r["run"])]
    ax.scatter(xs, ys, s=18, c=color, marker=marker, alpha=0.75, label=fam)
for name, label, dx in [
    ("0019_allchannel_trimmed_mean_f005_q32", "mean $f_0$ 0.05", (5, -10)),
    ("0019_allchannel_pass3_round1_fraction20_step10_fitted8", "production 20% bar", (5, 6)),
    ("024_convolving_perchannel_lockout5_q32", "perchannel5 q32", (5, -10)),
]:
    r = by_run[name]
    ax.annotate(label, (r["n_events"], r["recall_good"]), textcoords="offset points",
                xytext=dx, fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("accepted events")
ax.set_ylabel("Kilosort good-spike time coverage")
ax.set_title("Coverage vs yield across all 57 post-0018 runs (±0.5 ms, any channel)")
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.25)
ax.legend(fontsize=8)
fig.savefig(os.path.join(OUT, "coverage_vs_yield.png"), dpi=800, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.4, 4.8), facecolor="white")
trimmed = [r for r in rows if trimmed_parse(r["run"])]
trimmed.sort(key=lambda r: (RULES.index(trimmed_parse(r["run"])[0]), trimmed_parse(r["run"])[1]))
x = np.arange(len(trimmed))
acc = np.array([r["recall_good"] for r in trimmed])
rej_only = np.array([r["good_only_rejected"] for r in trimmed])
uncovered = np.clip(1.0 - acc - rej_only, 0, None)
labels = ["%s\n%.2f" % (trimmed_parse(r["run"])[0], trimmed_parse(r["run"])[1]) for r in trimmed]
ax.bar(x, acc, color="#2ca02c", label="accepted (covered)")
ax.bar(x, rej_only, bottom=acc, color="#d62728", label="proposed, bar rejected")
ax.bar(x, uncovered, bottom=acc + rej_only, color="#bbbbbb", label="never proposed")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylim(0, 1.02)
ax.set_ylabel("share of Kilosort good-spike times")
ax.set_title("Where each trimmed run's good-spike coverage goes (rule, $f_0$)")
ax.legend(fontsize=8, loc="lower right")
fig.savefig(os.path.join(OUT, "acceptance_funnel.png"), dpi=800, bbox_inches="tight")
plt.close(fig)

print("wrote", sorted(os.listdir(OUT)))
