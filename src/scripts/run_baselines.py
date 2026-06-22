"""Non-learned baselines on the three held-out splits (real numbers, eval-first)."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perturbvae import data as D, splits as S, baselines as B, metrics as M

def eval_split(train, test, cond_delta, n_genes):
    train = [c for c in train if c in cond_delta]
    test = [c for c in test if c in cond_delta]
    mot = B.mean_of_training_delta([cond_delta[c] for c in train]) if train else B.control_mean_delta(n_genes)
    preds = {"mean_of_training": mot, "control_mean": B.control_mean_delta(n_genes)}
    out = {}
    for bn, pred in preds.items():
        per = {c: M.score_condition(pred, cond_delta[c]) for c in test}
        out[bn] = M.summarize(per)
    return {"n_train": len(train), "n_test": len(test), "baselines": out}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--out", default="results/baselines.json")
    a = ap.parse_args()
    adata = D.load_adata(a.h5ad)
    genes, ctrl_mean, cond_delta, conds = D.pseudobulk(adata)
    n_genes = len(genes)
    gene_emb, dim = D.coexpression_gene_embedding(adata, dim=128)
    groups = D.functional_groups_from_coexpr(gene_emb, n_groups=40)
    perturbed = sorted({g for c in conds for g in S.perturbed_genes(c)})
    held = S.choose_holdout_groups(groups, perturbed, 0.2)
    trA, teA = S.random_split(conds, 0.2)
    trB, teB = S.function_split(conds, groups, held)
    trC, teC = S.combinatorial_split(conds)
    res = {"dataset": a.h5ad, "n_genes": n_genes, "n_conditions": len(conds),
           "n_holdout_groups": len(held),
           "A_random": eval_split(trA, teA, cond_delta, n_genes),
           "B_function": eval_split(trB, teB, cond_delta, n_genes),
           "C_combinatorial": eval_split(trC, teC, cond_delta, n_genes)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    for sp in ["A_random","B_function","C_combinatorial"]:
        s=res[sp]; print(f"[{sp}] train={s['n_train']} test={s['n_test']}")
        for bn,sm in s["baselines"].items():
            print(f"   {bn:16} pearson={sm['mean_pearson']:.3f} mse={sm['mean_mse']:.4f} jaccard={sm['mean_jaccard']:.3f}")
    print("DONE_BASELINES")

if __name__ == "__main__":
    main()
