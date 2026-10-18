# Tables and figures from results/grid.csv and results/ppl.csv -> results/tables.md, figures/fig_*.png
# Run: python analyze.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

g = pd.read_csv("results/grid.csv")
g["hidden"] = g.hidden.fillna(0).astype(int)
g["name"] = g.method + g.hidden.map(lambda h: f" h={h}" if h else "")
# RTN does not depend on B: keep one copy per (layer, matrix, rotated, seed)
g = pd.concat([g[~g.method.str.startswith("RTN")],
               g[g.method.str.startswith("RTN") & (g.B == g.B.min())]])
g["sq"] = g.rel_err * g.numel   # matrices are compared by relative error; aggregate weighted by size
real, ctrl = g[g.layer >= 0], g[g.layer == -1]
out = []


def per_seed(d, keys):
    """Size-weighted mean error and bpw over matrices, per seed."""
    return d.groupby(keys + ["seed"]).apply(
        lambda x: pd.Series(dict(bpw=(x.bpw * x.numel).sum() / x.numel.sum(),
                                 err=x.sq.sum() / x.numel.sum())), include_groups=False).reset_index()


def agg(d, keys):
    """per_seed, then mean and std over seeds."""
    return per_seed(d, keys).groupby(keys).agg(bpw=("bpw", "mean"), err=("err", "mean"), err_std=("err", "std"),
                               n_seeds=("seed", "nunique")).reset_index()


def md(df, floatfmt=".4f"):
    return df.to_markdown(index=False, floatfmt=floatfmt)


# ---- table 1: whole grid aggregated (3 layers x 7 matrices) ----
t1 = agg(real, ["rotated", "name", "B"]).sort_values(["rotated", "name", "B"])
out += ["## Weight error over the whole grid (3 layers x 7 matrices, mean over 3 seeds)\n", md(t1), ""]

# ---- fig 1: error vs bpw ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for rot, ax in zip((False, True), axes):
    d = t1[t1.rotated == rot]
    for name in sorted(d.name.unique()):
        x = d[d.name == name].sort_values("bpw")
        ls = "o--" if name.startswith("RTN") else "o-"
        ax.errorbar(x.bpw, x.err, yerr=x.err_std, fmt=ls, ms=4, capsize=2, label=name)
    ax.set(title="rotated" if rot else "not rotated", xlabel="bits per weight (incl. table / network)",
           yscale="log")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("relative weight error ||W-Ŵ||²/||W||²")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/fig_err_vs_bpw.png", dpi=130)

# ---- table 2 / fig 2: gain over random, real weights vs Gaussian control ----
def gains(d, keys):
    """Gain over random of the size-weighted error, per seed (keys select the aggregation level)."""
    p = per_seed(d[d.method.isin(["random", "kmeans", "generator"])], keys + ["rotated", "B", "name"])
    p = p.pivot_table(index=keys + ["rotated", "B", "seed"], columns="name", values="err").reset_index()
    for c in [c for c in p.columns if c.startswith(("kmeans", "generator"))]:
        p["gain " + c] = 100 * (1 - p[c] / p["random"])
    return p


def gain_table(d, keys):
    p = gains(d, keys)
    cols = [c for c in p.columns if c.startswith("gain")]
    m = p.groupby(keys + ["rotated", "B"])[cols].mean()
    s = p.groupby(keys + ["rotated", "B"])[cols].std().add_suffix(" std")
    return pd.concat([m, s], axis=1).reset_index()


gr = gain_table(real, [])
gc = gain_table(ctrl, [])
out += ["## Gain over random, % (1 - err/err_random), real weights\n", md(gr, ".2f"), "",
        "## Same on Gaussian noise of the same shape (control, 1 seed)\n", md(gc, ".2f"), ""]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for rot, ax in zip((False, True), axes):
    for d, lab, ls in ((gr, "weights", "-"), (gc, "noise (control)", ":")):
        x = d[d.rotated == rot]
        for h, c in ((32, "tab:olive"), (64, "tab:green"), (128, "tab:cyan")):
            ax.plot(x.B, x[f"gain generator h={h}"], ls, color=c, marker="o", ms=4, label=f"generator h={h}, {lab}")
        ax.plot(x.B, x["gain kmeans"], ls, color="tab:blue", marker="s", ms=4, label=f"kmeans, {lab}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title="rotated" if rot else "not rotated", xlabel="B, bits per codeword index")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("error reduction vs random, %")
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig("figures/fig_gain.png", dpi=130)

# ---- table 3: structure beyond noise, per layer x matrix (B=12, h=64) ----
pl = gains(real[real.B == 12], ["layer", "matrix"])
pc = gains(ctrl[ctrl.B == 12], ["matrix"])[["matrix", "rotated", "gain generator h=64", "gain kmeans"]]
pl = pl.groupby(["layer", "matrix", "rotated"])[["gain generator h=64", "gain kmeans"]].mean().reset_index()
pl = pl.merge(pc.rename(columns={"gain generator h=64": "ctrl gen", "gain kmeans": "ctrl kmeans"}),
              on=["matrix", "rotated"])
pl["gen - ctrl"] = pl["gain generator h=64"] - pl["ctrl gen"]
t3 = pl.pivot_table(index=["layer", "matrix"], columns="rotated",
                    values=["gain generator h=64", "gen - ctrl"]).round(2)
t3.columns = [f"{a} ({'rot' if b else 'no rot'})" for a, b in t3.columns]
out += ["## Per layer and matrix, B=12, h=64: generator gain over random, and gain minus noise control, %\n",
        t3.reset_index().to_markdown(index=False), ""]

# ---- per-matrix bpw overhead (shows the cost of tables on small matrices) ----
ov = real[(real.layer == 12) & (real.seed == 0) & real.method.isin(["kmeans", "generator"])]
ov = ov.pivot_table(index=["matrix", "numel"], columns=["name", "B"], values="overhead_bpw")
out += ["## Extra bits per weight for storing the table / network (layer 12)\n", ov.round(3).to_markdown(), ""]

# ---- perplexity ----
try:
    p = pd.read_csv("results/ppl.csv")
    p = p.sort_values(["bpw", "ppl"], ascending=[False, True])
    out += ["## WikiText-2 test perplexity (first windows of 1024 tokens)\n",
            md(p[["method", "rotated", "B", "hidden", "bpw", "ppl", "windows"]], ".3f"), ""]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lab = [f"{m}{' + rotation' if r else ''} ({b:.2f} b/w)" for m, r, b in zip(p.method, p.rotated, p.bpw)]
    ax.barh(lab, p.ppl, color=["tab:green" if m == "generator" else "tab:blue" if m == "kmeans"
                               else "tab:red" if m == "random" else "0.5" for m in p.method])
    ax.set(xscale="log", xlabel="perplexity (log scale, lower is better)")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig("figures/fig_ppl.png", dpi=130)
except FileNotFoundError:
    out.append("_no results/ppl.csv_")

open("results/tables.md", "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
