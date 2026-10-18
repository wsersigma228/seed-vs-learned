# Systematic grid over layers, matrices, rotation, bit budget, generator size, seeds.
# Results are appended to results/grid.csv; finished configs are skipped on restart.
# Run: python grid.py   (env: LAYERS="2,12,22", BS="8,10,12,14", SEEDS="0,1,2")

import os, time
import pandas as pd
import torch
import seed_vs_learned as svl

LAYERS = [int(x) for x in os.environ.get("LAYERS", "-1,2,12,22").split(",")]   # -1 = Gaussian control
MATRICES = os.environ.get("MATRICES", "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,"
                          "self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj").split(",")
BS = [int(x) for x in os.environ.get("BS", "8,10,12,14").split(",")]
HIDDENS = (32, 64, 128)
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
OUT = os.environ.get("OUT", "results/grid.csv")

if __name__ == "__main__":
    done = set()
    if os.path.exists(OUT):
        old = pd.read_csv(OUT)
        done = set(zip(old.layer, old.matrix, old.rotated, old.B, old.seed))
    model = svl.load_model()
    t0 = time.time()
    for layer in LAYERS:
        for matrix in MATRICES:
            W = svl.get_linear(model, max(layer, 0), matrix).weight.detach().to(svl.DEV).float()
            if layer == -1:
                # control: iid Gaussian of the same shape and scale, i.e. no structure at all
                W = torch.randn(W.shape, device=svl.DEV, generator=torch.Generator(svl.DEV).manual_seed(99 + MATRICES.index(matrix))) * W.std()
            for rotated in (False, True):
                for seed in (SEEDS[:1] if layer == -1 else SEEDS):   # control: one seed
                    for i, B in enumerate(BS):
                        if (layer, matrix, rotated, B, seed) in done:
                            continue
                        t = time.time()
                        rows = svl.run(W, rotated, B=B, hiddens=HIDDENS, seed=seed,
                                       rtn_bits=(2, 3, 4))
                        for r in rows:
                            r.update(layer=layer, matrix=matrix, rotated=rotated, B=B,
                                     seed=seed, numel=W.numel())
                        pd.DataFrame(rows).to_csv(OUT, mode="a", index=False,
                                                  header=not os.path.exists(OUT))
                        print(f"L{layer} {matrix} rot={rotated} seed={seed} B={B}: "
                              f"{time.time() - t:.1f}s (total {time.time() - t0:.0f}s)", flush=True)
            del W
            torch.cuda.empty_cache()
