"""Data adapter: GEARS-format AnnData -> pseudobulk deltas + gene embeddings.

The processed AnnData (from GEARS ``PertData``) has:

    obs['condition']  : 'ctrl', 'GENE+ctrl' (single), 'GENEA+GENEB' (double)
    obs['control']    : control flag (1 for non-targeting controls)
    var['gene_name']  : gene symbols
    X                 : log-normalized expression (float32)

The evaluation works on the per-condition pseudobulk DELTA: the mean expression
of a perturbation minus the mean of control cells. To predict the response of a
perturbation never seen in training, each perturbed gene is represented by a
generalizable embedding (its co-expression signature on control cells), so the
conditional prior can be queried for any gene, seen or unseen.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .splits import is_control, perturbed_genes


def load_adata(path: str):
    import anndata as ad
    return ad.read_h5ad(path)


def _dense(X) -> np.ndarray:
    import scipy.sparse as sp
    return np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)


def _gene_names(adata) -> List[str]:
    return adata.var["gene_name"].astype(str).tolist()


def pseudobulk(adata) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray], List[str]]:
    """Return (gene_names, control_mean[G], cond_delta{cond: delta[G]}, conditions)."""
    genes = _gene_names(adata)
    cond = adata.obs["condition"].astype(str).values
    X = _dense(adata.X).astype(np.float32)
    ctrl_mask = np.array([is_control(c) for c in cond])
    if ctrl_mask.sum() == 0:
        raise RuntimeError("no control cells found")
    control_mean = X[ctrl_mask].mean(0).astype(np.float64)
    cond_delta: Dict[str, np.ndarray] = {}
    for c in sorted(set(cond)):
        if is_control(c):
            continue
        cond_delta[c] = X[cond == c].mean(0).astype(np.float64) - control_mean
    return genes, control_mean, cond_delta, sorted(set(cond))


def coexpression_gene_embedding(adata, dim: int = 128, seed: int = 0) -> Tuple[Dict[str, np.ndarray], int]:
    """Per-gene embedding = PCA of the gene-gene co-expression signature computed
    on control cells. Defined for every gene, so it generalizes to perturbations
    of genes never seen during training."""
    from sklearn.decomposition import PCA

    genes = _gene_names(adata)
    cond = adata.obs["condition"].astype(str).values
    X = _dense(adata.X).astype(np.float32)
    ctrl = X[np.array([is_control(c) for c in cond])]
    Xc = ctrl - ctrl.mean(0, keepdims=True)
    sd = Xc.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    Xc = Xc / sd
    corr = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)  # [G, G] gene-gene correlation
    d = int(min(dim, corr.shape[0]))
    emb = PCA(n_components=d, random_state=seed).fit_transform(corr.astype(np.float64))
    return {g: emb[i] for i, g in enumerate(genes)}, d


def functional_groups_from_coexpr(
    gene_emb: Dict[str, np.ndarray], n_groups: int = 20, seed: int = 0
) -> Dict[str, str]:
    """Cluster genes by their co-expression embedding into functional groups,
    used to define the leakage-aware split B (whole groups withheld)."""
    from sklearn.cluster import KMeans

    genes = list(gene_emb.keys())
    M = np.stack([gene_emb[g] for g in genes])
    labels = KMeans(n_clusters=n_groups, random_state=seed, n_init=10).fit_predict(M)
    return {g: f"grp{int(l)}" for g, l in zip(genes, labels)}


def condition_embedding(cond: str, gene_emb: Dict[str, np.ndarray], dim: int) -> np.ndarray:
    """Embedding for a condition = mean of its perturbed genes' embeddings.
    Zeros for the control (or a condition whose genes are all unknown)."""
    gs = [g for g in perturbed_genes(cond) if g in gene_emb]
    if not gs:
        return np.zeros(dim, dtype=np.float64)
    return np.mean([gene_emb[g] for g in gs], axis=0)
