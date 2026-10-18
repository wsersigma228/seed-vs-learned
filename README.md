# seed-vs-learned

Does a learned codeword generator beat SeedLM's pseudo-random one when compressing LLM weights? I tested this on Qwen2.5-0.5B. Short answer: not in this setup.

## Idea

[SeedLM](https://arxiv.org/abs/2410.10714) stores a block of weights as a PRNG seed plus a few coefficients, so the codebook costs nothing to store. My idea was to replace the PRNG with a small network trained on the weights of the model (codeword index in, vector out) and get a lower error for the same number of bits.

There is an obvious counter-argument. Methods like QuIP# and QTIP rotate the weights first, after which they look a lot like Gaussian noise. If that is true, a learned generator has nothing model-specific to learn. So I ran everything with and without rotation, and I also ran it on pure Gaussian noise as a control.

## Results

![weight error vs bits per weight](figures/fig_err_vs_bpw.png)

- A learned codebook (generator or k-means) has about 4% lower weight error than a random one at B=12. On Gaussian noise of the same shape the gap is the same, about 4%. So what the generator learns is a better way to cover the sphere, not structure that is specific to Qwen weights.
- Rotation barely matters: the gain over random is 4.2% without rotation and 3.9% with it. The one exception is `v_proj` without rotation (layers 2 and 22), where the gain is 5-8 points above the noise control. Rotation removes it.
- When the generator's own parameters are counted in bits per weight, the gain mostly disappears. Random with the same extra bits does as well or slightly better.
- At ~2.5 bits per weight every codebook method breaks the model. WikiText-2 perplexity is 14.5 for the original model, 42-70 for 3-bit RTN, and 2185-3543 for the generator. The generator does not beat 3-bit RTN.

![gain over random: weights vs noise](figures/fig_gain.png)

Numbers, per-matrix tables, limitations and ideas for next steps are in [report.md](report.md).

## Setup

Every weight matrix is split into blocks of 8 values. A block is stored as a codeword index (B bits) plus a gain (8 bits). The codewords come from one of:

- `random`: a seeded PRNG (simplified SeedLM, one codeword per block)
- `kmeans`: a table fitted to the matrix
- `generator`: an MLP from the B index bits to 8 values, hidden size 32/64/128

RTN at 2/3/4 bits (asymmetric, group size 128) is the baseline. The rotation is a random two-sided orthogonal transform. The k-means table and the generator weights (fp16) are included in bits per weight.

The grid is layers 2/12/22 × all 7 linear layers × rotation on/off × B ∈ {8, 10, 12, 14} × hidden ∈ {32, 64, 128} × 3 seeds, plus the Gaussian control.

## Running it

Needs a CUDA GPU; I used an RTX 3060 Laptop (6 GB) with Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python seed_vs_learned.py   # one matrix, ~1 min (SYNTHETIC=1 skips the model download)
python ppl.py               # perplexity with all linear layers compressed, ~15 min
python grid.py              # full grid, ~45 min
python analyze.py           # results/tables.md and figures/
```

`watch.py` shows live plots while `grid.py` and `ppl.py` run, and `run_all.ps1` runs both with a log. Settings are environment variables (`LAYER`, `MATRIX`, `B`, `HIDDEN`, `SEED`, `GEN_STEPS`, `N_WINDOWS`, ...), listed at the top of each script.

## Files

```
seed_vs_learned.py   codebooks, encoding, RTN, rotation; runs on one matrix
grid.py              the grid -> results/grid.csv
ppl.py               WikiText-2 perplexity -> results/ppl.csv
analyze.py           tables and figures
watch.py             live plots during a run
report.md            full write-up
results/             raw results and summary tables
figures/             plots
```

## License

MIT
