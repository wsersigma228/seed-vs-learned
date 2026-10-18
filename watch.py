# Live view of running experiments: re-reads the CSV files every few seconds.
# Run: python watch.py   (close the window to stop; it never touches the results)

import os
import matplotlib.pyplot as plt
import pandas as pd

GRID, PPL = "results/grid.csv", "results/ppl.csv"
TOTAL = int(os.environ.get("TOTAL", (3 * 3 + 1) * 7 * 2 * 4))   # (layers x seeds + control) x matrices x rot x B
COLORS = {"random": "tab:red", "kmeans": "tab:blue", "generator": "tab:green",
          "RTN2": "0.6", "RTN3": "0.4", "RTN4": "0.2"}


def read(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None


fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
while plt.fignum_exists(fig.number):
    g, p = read(GRID), read(PPL)
    for ax in axes:
        ax.clear()
    if g is not None and len(g):
        done = len(g.drop_duplicates(["layer", "matrix", "rotated", "B", "seed"]))
        last = g.iloc[-1]
        fig.suptitle(f"grid: {done}/{TOTAL} configs ({100 * done / TOTAL:.0f}%), "
                     f"last: layer {last.layer} {last.matrix} rotated={last.rotated} B={last.B}")
        real = g[g.layer >= 0]
        for rot, ax in zip((False, True), axes[:2]):
            d = real[real.rotated == rot]
            for m, c in COLORS.items():
                x = d[d.method == m]
                if len(x):
                    ax.scatter(x.bpw, x.rel_err, s=8, c=c, label=m, alpha=0.6)
            ax.set(title=f"{'rotated' if rot else 'not rotated'}: error vs bits per weight",
                   xlabel="bits per weight (incl. table / network)", ylabel="relative error", yscale="log", xlim=(1.9, 4.5))
            ax.legend(fontsize=8)
    else:
        fig.suptitle("grid: no results yet")
    if p is not None and len(p):
        lab = [f"{m}{' rot' if r else ''}" for m, r in zip(p.method, p.rotated)]
        axes[2].barh(lab, p.ppl, color=[COLORS.get(m, "k") for m in p.method])
        axes[2].set(title="WikiText-2 perplexity (lower is better)", xscale="log")
    else:
        axes[2].set_title("perplexity: not started")
    fig.tight_layout()
    plt.pause(5)
