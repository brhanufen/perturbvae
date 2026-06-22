"""Train a conditional VAE per held-out split and evaluate vs the baseline.

For each split (A random / B function-grouped / C combinatorial) we train a
fresh cVAE on that split's TRAIN conditions (plus controls), then predict the
held-out perturbation deltas and score them. We report the cVAE vs the
mean-of-training baseline, and the cVAE's own A-vs-B leakage gap.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perturbvae import data as D, splits as S, baselines as Bl, metrics as M
from perturbvae.model import ConditionalVAE, vae_loss


def cell_tensors(adata, keep_conds, cond2emb, control_mean):
    cond = adata.obs["condition"].astype(str).values
    keep = set(keep_conds)
    mask = np.array([(c in keep) or S.is_control(c) for c in cond])
    # train on the DELTA: cell expression minus the control mean
    X = D._dense(adata.X).astype(np.float32)[mask] - control_mean.astype(np.float32)[None, :]
    E = np.stack([cond2emb[c] for c in cond[mask]]).astype(np.float32)
    return torch.tensor(X), torch.tensor(E)


def train_one(X, E, n_genes, dim, device, epochs, beta):
    dl = DataLoader(TensorDataset(X, E), batch_size=512, shuffle=True, drop_last=True)
    model = ConditionalVAE(n_genes=n_genes, pert_dim=dim, latent=64, hidden=512).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(epochs):
        b = beta * min(1.0, (ep + 1) / max(1, epochs // 3))  # KL anneal
        for xb, eb in dl:
            xb, eb = xb.to(device), eb.to(device)
            xhat, q, p = model(xb, eb)
            loss, _, _ = vae_loss(xhat, xb, q, p, beta=b)
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def predict_delta(model, cond2emb, cond, dim, device, n=30):
    ec = torch.tensor(cond2emb[cond], dtype=torch.float32, device=device)
    return model.predict_response(ec, n).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--out", default="results/cvae.json")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    adata = D.load_adata(a.h5ad)
    genes, ctrl_mean, cond_delta, conds = D.pseudobulk(adata); n_genes = len(genes)
    gene_emb, dim = D.coexpression_gene_embedding(adata, dim=128)
    groups = D.functional_groups_from_coexpr(gene_emb, n_groups=40)
    perturbed = sorted({g for c in conds for g in S.perturbed_genes(c)})
    held = S.choose_holdout_groups(groups, perturbed, 0.2)
    cond2emb = {c: D.condition_embedding(c, gene_emb, dim) for c in set(adata.obs["condition"].astype(str))}
    splits = {"A_random": S.random_split(conds, 0.2),
              "B_function": S.function_split(conds, groups, held),
              "C_combinatorial": S.combinatorial_split(conds)}
    res = {"n_genes": n_genes, "pert_dim": dim, "device": dev}
    for name, (tr, te) in splits.items():
        te_present = [c for c in te if c in cond_delta]
        if not te_present:
            print(f"[{name}] skipped (no test conditions in this dataset)", flush=True)
            continue
        t0 = time.time()
        X, E = cell_tensors(adata, tr, cond2emb, ctrl_mean)
        model = train_one(X, E, n_genes, dim, dev, a.epochs, a.beta)
        te_use = [c for c in te if c in cond_delta]
        per = {c: M.score_condition(predict_delta(model, cond2emb, c, dim, dev), cond_delta[c]) for c in te_use}
        cvae = M.summarize(per)
        tr_use = [c for c in tr if c in cond_delta]
        mot = Bl.mean_of_training_delta([cond_delta[c] for c in tr_use])
        bse = M.summarize({c: M.score_condition(mot, cond_delta[c]) for c in te_use})
        res[name] = {"cvae": cvae, "mean_of_training": bse, "n_train": len(tr_use),
                     "n_test": len(te_use), "n_cells_train": int(X.shape[0]), "sec": round(time.time()-t0, 1)}
        print(f"[{name}] cVAE pearson={cvae['mean_pearson']:.3f} jacc={cvae['mean_jaccard']:.3f} "
              f"| MoT pearson={bse['mean_pearson']:.3f} | n_test={len(te_use)} ({res[name]['sec']}s)", flush=True)
    if "A_random" in res and "B_function" in res:
        res["cvae_leakage_gap"] = res["A_random"]["cvae"]["mean_pearson"] - res["B_function"]["cvae"]["mean_pearson"]
        res["mot_leakage_gap"] = res["A_random"]["mean_of_training"]["mean_pearson"] - res["B_function"]["mean_of_training"]["mean_pearson"]
        print(f"LEAKAGE GAP (A-B):  cVAE={res['cvae_leakage_gap']:.3f}  MoT={res['mot_leakage_gap']:.3f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print("DONE_CVAE")

if __name__ == "__main__":
    main()
