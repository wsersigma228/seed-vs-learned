# Random (SeedLM-style) vs. learned codebooks for compressing LLM weights.
#
# One weight matrix of Qwen is cut into blocks of 8 values, and every block is coded with the
# same number of bits in three ways:
#   random    - codewords come from a seeded PRNG, only the seed is stored (SeedLM-like)
#   kmeans    - codewords are fitted to this matrix and stored as a table (upper bound for "learned")
#   generator - codewords come from a tiny MLP trained on this matrix (the hypothesis being tested)
# Plain round-to-nearest (RTN) quantization is the reference. Everything is repeated with and
# without a random rotation of the weights (as in QuIP#/QTIP).
#
# Simplification vs. the SeedLM paper: one pseudo-random codeword + one gain per block
# (gain-shape VQ), not a combination of several random vectors with several coefficients.
#
# Run: python seed_vs_learned.py   (env: SYNTHETIC=1, LAYER, MATRIX, B, HIDDEN, SEED)

import os, time
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- settings ----------------
MODEL = "Qwen/Qwen2.5-0.5B"
LAYER = int(os.environ.get("LAYER", 12))
MATRIX = os.environ.get("MATRIX", "mlp.down_proj")
D = 8                      # block size
B = int(os.environ.get("B", 12))   # bits per codeword index -> 2^12 = 4096 codewords
HIDDEN = int(os.environ.get("HIDDEN", 64))
SEED = int(os.environ.get("SEED", 0))
GAIN_BITS = 8              # bits per block gain (1 sign + 7 log-scale)
PARAM_BITS = 16            # kmeans tables and generator weights are stored as fp16
N_TRAIN = int(os.environ.get("N_TRAIN", 100_000))
TEST_FRAC = 0.2            # held-out share of blocks for measuring the error
KMEANS_ITERS = 10          # 25 iterations gave the same error
GEN_STEPS = int(os.environ.get("GEN_STEPS", 300))   # 2000 steps give ~2% lower error but take ~7x longer
RTN_GROUP = 128            # asymmetric RTN, fp16 scale + fp16 zero per group
SYNTHETIC = os.environ.get("SYNTHETIC") == "1"   # test run without downloading the model


# ---------------- weights ----------------
def synthetic_weight():
    # Gaussian noise plus a few outlier columns, roughly like real weights
    g = torch.Generator().manual_seed(0)
    W = torch.randn(896, 4864, generator=g) * 0.02
    cols = torch.randperm(4864, generator=g)[:20]
    W[:, cols] *= 8
    return W


def load_model():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)


def get_linear(model, layer, matrix):
    mod = model.model.layers[layer]
    for part in matrix.split("."):
        mod = getattr(mod, part)
    return mod


# ---------------- building blocks ----------------
def rel_err(X, Xh):
    return (((X - Xh) ** 2).sum() / (X ** 2).sum()).item()


class GainQuant:
    """Block gain: sign + log-scale magnitude. Shared by all codebook methods, so any
    difference between them comes from the codewords only."""
    def __init__(self, g_train):
        mag = g_train.abs().clamp_min(1e-12)
        self.lo = torch.quantile(mag[: 100_000].float(), 0.001).log()
        self.hi = mag.max().log()
        self.levels = 2 ** (GAIN_BITS - 1)

    def __call__(self, g):
        s = torch.sign(g)
        lg = g.abs().clamp_min(1e-12).log().clamp(self.lo, self.hi)
        q = torch.round((lg - self.lo) / (self.hi - self.lo) * (self.levels - 1))
        return s * torch.exp(self.lo + q / (self.levels - 1) * (self.hi - self.lo))


def normalize(C):
    return C / C.norm(dim=1, keepdim=True).clamp_min(1e-12)


def assign(X, C):
    """For every block, pick the codeword with the largest |dot product|."""
    chunk = max(1024, 2 ** 26 // C.shape[0])   # keep the dots matrix at <= 256 MB
    idx, g = [], []
    for i in range(0, X.shape[0], chunk):
        dots = X[i:i + chunk] @ C.T
        j = dots.abs().argmax(dim=1)
        idx.append(j)
        g.append(dots.gather(1, j[:, None]).squeeze(1))
    return torch.cat(idx), torch.cat(g)


def encode(X, C, gq):
    idx, g = assign(X, C)
    return gq(g)[:, None] * C[idx]


def code_error(Xtr, Xte, C):
    C = normalize(C)
    _, gtr = assign(Xtr, C)
    return rel_err(Xte, encode(Xte, C, GainQuant(gtr)))


# ---------------- three ways to get codewords ----------------
# Each returns (codebook, overhead_bits): bits that must be stored besides per-block codes.
def codebook_random(B, seed=1234):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(2 ** B, D, generator=gen).to(DEV), 32   # only the 32-bit seed


def codebook_kmeans(Xtr, B):
    K, n = 2 ** B, Xtr.shape[0]
    init = torch.randperm(n, device=DEV)[:K] if n >= K else torch.randint(0, n, (K,), device=DEV)
    C = normalize(Xtr[init].clone())
    for _ in range(KMEANS_ITERS):
        idx, g = assign(Xtr, C)
        newC = torch.zeros_like(C)
        # one power-iteration step on each cluster's scatter matrix: increases sum (x.c)^2,
        # i.e. lowers the residual ||x||^2 - (x.c)^2 that we measure
        newC.index_add_(0, idx, g[:, None] * Xtr)
        empty = newC.norm(dim=1) < 1e-12
        newC[empty] = C[empty]
        C = normalize(newC)
    return C, K * D * PARAM_BITS


class Generator(torch.nn.Module):
    """Codeword index (B bits) -> vector of 8 values. With B=12, hidden=64 that is ~5.5k
    parameters instead of a 4096 x 8 = 32768 entry table."""
    def __init__(self, B, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(B, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, D),
        )
        codes = torch.arange(2 ** B)[:, None] >> torch.arange(B)[None, :] & 1
        self.register_buffer("codes", codes.float() * 2 - 1)

    def forward(self):
        return normalize(self.net(self.codes))


def codebook_generator(Xtr, B, hidden=64):
    gen = Generator(B, hidden).to(DEV)
    opt = torch.optim.Adam(gen.parameters(), lr=3e-3)
    # Keep block norms: the measured error weights blocks by ||x||^2, so must the loss.
    Xn = Xtr / Xtr.pow(2).sum(dim=1).mean().sqrt()
    for step in range(GEN_STEPS):
        batch = Xn[torch.randint(0, Xn.shape[0], (8192,), device=DEV)]
        C = gen()
        # max() only passes gradient to the argmax codeword, so search it without autograd
        with torch.no_grad():
            j = (batch @ C.T).abs().argmax(dim=1)
        best = (batch * C[j]).sum(dim=1)
        loss = -(best ** 2).mean()   # maximising the match = minimising the residual
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Codewords are recomputed from fp16 weights, so the stored size is what we measure.
    gen.half().float()
    n_params = sum(p.numel() for p in gen.parameters())
    with torch.no_grad():
        return gen(), n_params * PARAM_BITS


# ---------------- reference: round-to-nearest ----------------
def rtn(W, bits, group=RTN_GROUP):
    """Asymmetric min-max RTN: all 2^bits levels are used."""
    Wg = W.reshape(-1, group)
    lo, hi = Wg.min(dim=1, keepdim=True).values, Wg.max(dim=1, keepdim=True).values
    scale = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-12)
    Wq = torch.round((Wg - lo) / scale).clamp(0, 2 ** bits - 1) * scale + lo
    return Wq.reshape(W.shape)


def rtn_bpw(bits):
    return bits + 2 * 16 / RTN_GROUP


def random_rotation(n):
    Q, R = torch.linalg.qr(torch.randn(n, n, device=DEV))
    return Q * torch.sign(torch.diagonal(R))   # sign fix -> Haar-distributed


def rotate(W):
    """Two-sided orthogonal rotation as in QuIP#/QTIP (they use randomized Hadamard).
    Weight error is the same in both bases, so errors are directly comparable."""
    return random_rotation(W.shape[0]) @ W @ random_rotation(W.shape[1])


def split_blocks(W):
    blocks = W.reshape(-1, D)
    perm = torch.randperm(blocks.shape[0], device=DEV)
    n_test = int(blocks.shape[0] * TEST_FRAC)
    return blocks[perm[n_test:n_test + N_TRAIN]], blocks[perm[:n_test]]


# ---------------- experiment ----------------
def run(W, rotated, B=B, hiddens=(HIDDEN,), seed=SEED, rtn_bits=(2, 3)):
    """Returns rows: dict(method, hidden, code_bpw, overhead_bpw, bpw, rel_err)."""
    torch.manual_seed(seed)
    if rotated:
        W = rotate(W)
    Xtr, Xte = split_blocks(W)
    code_bpw = (B + GAIN_BITS) / D
    rows = []

    def add(method, bpw_code, overhead_bits, err, hidden=None):
        ov = overhead_bits / W.numel()
        rows.append(dict(method=method, hidden=hidden, code_bpw=bpw_code,
                         overhead_bpw=ov, bpw=bpw_code + ov, rel_err=err))

    for bits in rtn_bits:
        add(f"RTN{bits}", rtn_bpw(bits), 0, rel_err(W, rtn(W, bits)))
    C, ov = codebook_random(B, seed=1234 + seed)
    add("random", code_bpw, ov, code_error(Xtr, Xte, C))
    C, ov = codebook_kmeans(Xtr, B)
    add("kmeans", code_bpw, ov, code_error(Xtr, Xte, C))
    for h in hiddens:
        C, ov = codebook_generator(Xtr, B, h)
        add("generator", code_bpw, ov, code_error(Xtr, Xte, C), hidden=h)
    return rows


if __name__ == "__main__":
    t0 = time.time()
    if SYNTHETIC:
        W = synthetic_weight()
    else:
        model = load_model()
        W = get_linear(model, LAYER, MATRIX).weight.detach().clone()
        del model
    W = W.to(DEV).float()
    print(f"matrix {MATRIX}, layer {LAYER}, shape {tuple(W.shape)}, device {DEV}, B={B}"
          + (" [synthetic]" if SYNTHETIC else ""))
    for rotated in (False, True):
        print(f"\n=== {'rotated' if rotated else 'not rotated'} ===")
        print(f"{'method':22s} {'bits/w':>8s} {'(overhead)':>13s} {'rel. error':>12s}")
        for r in run(W, rotated):
            name = r["method"] + (f" h={r['hidden']}" if r["hidden"] else "")
            print(f"{name:22s} {r['bpw']:8.3f} {r['overhead_bpw']:13.3f} {r['rel_err']:12.4f}")
    print(f"\ndone in {time.time() - t0:.0f} s")
