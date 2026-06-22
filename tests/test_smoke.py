"""Correctness/shape smoke test on SYNTHETIC data (not a scientific result)."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from perturbvae import splits as S
from perturbvae import metrics as M
from perturbvae import baselines as B
from perturbvae.model import ConditionalVAE, vae_loss

def test_splits():
    assert S.perturbed_genes("GENE+ctrl") == ["GENE"]
    assert S.perturbed_genes("A+B") == ["A", "B"]
    assert S.is_single("X+ctrl") and S.is_double("A+B") and S.is_control("ctrl")
    conds = [f"G{i}+ctrl" for i in range(100)] + ["A+B", "C+D", "ctrl"]
    tr, te = S.random_split(conds, 0.2)
    assert set(tr).isdisjoint(te) and 10 <= len(te) <= 30 and len(tr)+len(te) == 100
    assert S.random_split(conds,0.2) == (tr,te)  # deterministic
    g2grp = {f"G{i}": f"path{i%5}" for i in range(100)}
    perturbed = [f"G{i}" for i in range(100)]
    held = S.choose_holdout_groups(g2grp, perturbed, 0.2)
    trf, tef = S.function_split(conds, g2grp, held)
    # group-atomic: no train gene shares a held-out group
    train_groups = {g2grp[S.perturbed_genes(c)[0]] for c in trf}
    assert train_groups.isdisjoint(set(held)), "leakage: held-out group in train"
    trc, tec = S.combinatorial_split(conds)
    assert all(S.is_single(c) for c in trc) and all(S.is_double(c) for c in tec)
    print("splits OK")

def test_metrics():
    rng = np.random.RandomState(0); x = rng.randn(50)
    assert abs(M.delta_pearson(x, x) - 1.0) < 1e-9
    assert abs(M.delta_pearson(x, -x) + 1.0) < 1e-9
    assert M.delta_mse(x, x) == 0.0
    assert abs(M.deg_jaccard(x, x, 10) - 1.0) < 1e-9
    per = {"c1": M.score_condition(x, x), "c2": M.score_condition(x, -x)}
    s = M.summarize(per); assert s["n_conditions"] == 2
    gap = M.leakage_gap({"mean_pearson":0.5},{"mean_pearson":0.2})
    assert abs(gap["pearson_gap"] - 0.3) < 1e-9
    print("metrics OK")

def test_baselines():
    ds = [np.ones(10)*2, np.zeros(10), np.ones(10)*4]
    m = B.mean_of_training_delta(ds); assert np.allclose(m, 2.0)
    assert np.allclose(B.control_mean_delta(10), 0.0)
    print("baselines OK")

def test_model():
    torch.manual_seed(0)
    Bn, G, P = 8, 50, 16
    model = ConditionalVAE(n_genes=G, pert_dim=P, latent=16, hidden=64)
    x = torch.randn(Bn, G); pe = torch.randn(Bn, P)
    xhat, q, p = model(x, pe)
    assert xhat.shape == (Bn, G)
    loss, rec, kl = vae_loss(xhat, x, q, p, beta=1.0, free_bits=1.0)
    assert torch.isfinite(loss) and rec >= 0 and kl >= 0
    loss.backward()  # gradients flow
    assert any(pm.grad is not None for pm in model.parameters())
    d = model.predict_delta(pe[0], torch.zeros(P), n_samples=5)
    assert d.shape == (G,) and torch.isfinite(d).all()
    r = model.predict_response(pe[0], n_samples=5)
    assert r.shape == (G,) and torch.isfinite(r).all()
    print("model OK")

if __name__ == "__main__":
    test_splits(); test_metrics(); test_baselines(); test_model()
    print("\nALL SMOKE CHECKS PASSED (synthetic data; correctness only, not a result)")
