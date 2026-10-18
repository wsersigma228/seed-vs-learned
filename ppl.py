# Replace every linear layer of the transformer blocks (q,k,v,o,gate,up,down in all
# 24 layers) with its compressed version and measure WikiText-2 test perplexity.
# Embeddings / lm_head (tied in Qwen2.5-0.5B), norms and biases stay uncompressed.
# Compression is simulated: weights are decoded back to dense matrices before evaluation.
# Run: python ppl.py   (env: METHODS="fp,RTN3,generator", ROTATIONS="0,1", CTX, N_WINDOWS)

import math, os, time
import pandas as pd
import torch
import seed_vs_learned as svl

METHODS = os.environ.get("METHODS", "fp,RTN2,RTN3,RTN4,random,kmeans,generator").split(",")
ROTATIONS = [bool(int(x)) for x in os.environ.get("ROTATIONS", "0,1").split(",")]
B = int(os.environ.get("B", 12))
HIDDEN = int(os.environ.get("HIDDEN", 64))
SEED = int(os.environ.get("SEED", 0))
CTX = int(os.environ.get("CTX", 1024))
N_WINDOWS = int(os.environ.get("N_WINDOWS", 40))   # 0 = whole test set (~290 windows)
OUT = os.environ.get("OUT", "results/ppl.csv")
MATRICES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def compress(W, method, rotated):
    """Returns (decoded weight in the original basis, total stored bits)."""
    if rotated:
        U, V = svl.random_rotation(W.shape[0]), svl.random_rotation(W.shape[1])
        W = U @ W @ V
    if method.startswith("RTN"):
        bits = int(method[3:])
        Wh, total_bits = svl.rtn(W, bits), svl.rtn_bpw(bits) * W.numel()
    else:
        X = W.reshape(-1, svl.D)
        Xtr = X[torch.randperm(X.shape[0], device=svl.DEV)[:svl.N_TRAIN]]
        if method == "random":
            C, ov = svl.codebook_random(B, seed=1234 + SEED)
        elif method == "kmeans":
            C, ov = svl.codebook_kmeans(Xtr, B)
        else:
            C, ov = svl.codebook_generator(Xtr, B, HIDDEN)
        C = svl.normalize(C)
        _, gtr = svl.assign(Xtr, C)
        # All blocks are encoded: the codebook is fitted to this matrix and its cost is counted.
        Wh = svl.encode(X, C, svl.GainQuant(gtr)).reshape(W.shape)
        total_bits = (B + svl.GAIN_BITS) / svl.D * W.numel() + ov
    if rotated:
        Wh = U.T @ Wh @ V.T
    return Wh, total_bits


def wikitext_ids(tok):
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    return tok(text, return_tensors="pt").input_ids[0]


@torch.no_grad()
def perplexity(model, ids):
    n = ids.numel() // CTX
    if N_WINDOWS:
        n = min(n, N_WINDOWS)
    nll = 0.0
    for i in range(n):
        x = ids[i * CTX:(i + 1) * CTX][None].to(svl.DEV)
        nll += model(x, labels=x).loss.float().item()   # mean NLL over CTX-1 predictions
    return math.exp(nll / n), n


if __name__ == "__main__":
    from transformers import AutoTokenizer
    ids = wikitext_ids(AutoTokenizer.from_pretrained(svl.MODEL))
    print(f"WikiText-2 test: {ids.numel()} tokens, ctx {CTX}", flush=True)
    for method in METHODS:
        for rotated in ([False] if method == "fp" else ROTATIONS):
            torch.manual_seed(SEED)
            t = time.time()
            model = svl.load_model()   # fresh fp32 copy on CPU each time
            bits = numel = 0
            if method != "fp":
                for layer in model.model.layers:
                    for name in MATRICES:
                        lin = layer
                        for part in name.split("."):
                            lin = getattr(lin, part)
                        W = lin.weight.data.to(svl.DEV)
                        Wh, b = compress(W, method, rotated)
                        lin.weight.data.copy_(Wh.cpu())
                        bits, numel = bits + b, numel + W.numel()
                        del W, Wh
            t_comp = time.time() - t
            inv_freq = model.model.rotary_emb.inv_freq.clone()
            model = model.to(torch.bfloat16)
            model.model.rotary_emb.inv_freq = inv_freq   # RoPE frequencies must stay fp32
            model = model.to(svl.DEV).eval()
            ppl, n = perplexity(model, ids)
            row = dict(method=method, rotated=rotated, B=B if method in ("random", "kmeans", "generator") else None,
                       hidden=HIDDEN if method == "generator" else None, seed=SEED, ctx=CTX, windows=n,
                       bpw=bits / numel if numel else 16.0, ppl=ppl, compress_s=t_comp)
            print(row, flush=True)
            pd.DataFrame([row]).to_csv(OUT, mode="a", index=False, header=not os.path.exists(OUT))
            del model
            torch.cuda.empty_cache()
