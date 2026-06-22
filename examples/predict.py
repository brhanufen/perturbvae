"""Minimal example: predict the response to a held-out perturbation.

Loads a GEARS-format AnnData, builds the co-expression gene embeddings, trains a
small conditional VAE on the single-perturbation conditions, and prints the
predicted-vs-observed delta-Pearson for one held-out perturbation (or for an
unseen double, embedded from its two single-gene embeddings).

    python examples/predict.py --h5ad data/norman/perturb_processed.h5ad --cond "FOSB+CEBPB"

This is a usage demo on real data, not a benchmark run; for the full evaluation
use src/scripts/train.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from perturbvae import data as D, splits as S, metrics as M
from perturbvae.model import ConditionalVAE, vae_loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--cond", required=True,
                    help="condition to predict, e.g. 'FOSB+ctrl' or 'FOSB+CEBPB'")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"

    adata = D.load_adata(a.h5ad)
    genes, ctrl_mean, cond_delta, conds = D.pseudobulk(adata)
    n_genes = len(genes)
    gene_emb, dim = D.coexpression_gene_embedding(adata, dim=128)
    cond2emb = {c: D.condition_embedding(c, gene_emb, dim)
                for c in set(adata.obs["condition"].astype(str))}
    if a.cond not in cond_delta:
        raise SystemExit(f"{a.cond} not in dataset; example conditions: "
                         f"{[c for c in conds if not S.is_control(c)][:5]}")

    # train on all singles except the requested condition
    train = [c for c in conds if S.is_single(c) and c != a.cond]
    cond = adata.obs["condition"].astype(str).values
    keep = set(train)
    mask = np.array([(c in keep) or S.is_control(c) for c in cond])
    X = D._dense(adata.X).astype(np.float32)[mask] - ctrl_mean.astype(np.float32)[None, :]
    E = np.stack([cond2emb[c] for c in cond[mask]]).astype(np.float32)
    X, E = torch.tensor(X), torch.tensor(E)

    model = ConditionalVAE(n_genes, dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    from torch.utils.data import TensorDataset, DataLoader
    dl = DataLoader(TensorDataset(X, E), batch_size=512, shuffle=True, drop_last=True)
    model.train()
    for ep in range(a.epochs):
        b = 0.5 * min(1.0, (ep + 1) / max(1, a.epochs // 3))
        for xb, eb in dl:
            xb, eb = xb.to(dev), eb.to(dev)
            xhat, q, p = model(xb, eb)
            loss, _, _ = vae_loss(xhat, xb, q, p, beta=b)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()

    ec = torch.tensor(cond2emb[a.cond], dtype=torch.float32, device=dev)
    pred = model.predict_response(ec, n_samples=30).cpu().numpy()
    true = cond_delta[a.cond]
    print(f"condition           : {a.cond} "
          f"({'unseen double' if S.is_double(a.cond) else 'held-out single'})")
    print(f"delta-Pearson       : {M.delta_pearson(pred, true):.3f}")
    print(f"top-20 DE Jaccard   : {M.deg_jaccard(pred, true, k=20):.3f}")


if __name__ == "__main__":
    main()
