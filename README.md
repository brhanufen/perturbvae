# PerturbVAE

A leakage-aware benchmark for single-cell perturbation-response prediction. A small conditional VAE predicts the transcriptional response to a genetic perturbation it has never seen, and is evaluated under three leakage-controlled held-out splits on two real Perturb-seq datasets (Norman et al. 2019; Replogle et al. 2022). The focus of the project is the evaluation, not the architecture: how much apparent generalization survives when held-out perturbations are not allowed to be functional neighbors of the training perturbations, and where a learned model actually beats a baseline that ignores the perturbation identity.

**This is a mixed result, reported honestly.** The learned model beats a strong baseline in one regime, compositional prediction of unseen *double* perturbations, and loses to it on every single-perturbation split. A leakage-free split changes the measured generalization a lot on one dataset and almost nothing on another. The full write-up is in [`docs/PerturbVAE_Report.pdf`](docs/PerturbVAE_Report.pdf).

## Summary

| Question | Answer |
|---|---|
| Does the cVAE beat a strong baseline anywhere? | Yes, on **compositional** generalization. Predicting unseen *double* perturbations from singles (Norman), delta-Pearson **0.748 vs 0.551** for mean-of-training (+0.20), top-20 DE-gene overlap 0.34 vs 0.20. |
| Beats the baseline on single-perturbation prediction? | No. On every single-perturbation split of both datasets, mean-of-training is higher. The advantage is specific to composition. |
| Is there a leakage gap (random vs leakage-free split)? | Yes, but **dataset-dependent**: large on Norman (cVAE -0.27, baseline -0.20), nearly absent on Replogle (about -0.02). |
| Honest about scale? | Yes. Two datasets, three split designs, deterministic hashing, a zero-effect control and a strong mean baseline on every split. |

## Key numbers

| Dataset | Split | n test | cVAE Pearson | mean-of-training | leakage gap |
|---|---|---:|---:|---:|---:|
| Norman | A random | 20 | 0.522 | **0.616** | n/a |
| Norman | B function (leakage-free) | 28 | 0.257 | **0.412** | cVAE 0.265 · MoT 0.204 |
| Norman | C combinatorial | 131 | **0.748** | 0.551 | n/a |
| Replogle | A random | 237 | 0.305 | **0.391** | n/a |
| Replogle | B function (leakage-free) | 86 | 0.283 | **0.370** | cVAE 0.023 · MoT 0.021 |

## Repository layout

```
src/perturbvae/      splits, metrics, baselines, data adapter, conditional VAE model
src/scripts/         run_baselines.py, train.py, make_figures.py, sbatch_*.sh
tests/               unit tests (splits, metrics, baselines, model) on synthetic arrays
docs/                technical report (PDF + LaTeX source + docx)
figures/             result figures
results/             evaluation artifacts (per-split JSON for both datasets)
examples/            minimal one-command prediction example
```

## The three splits

- **A, random singles.** Random 80/20 over single-gene conditions. The optimistic number matching common practice.
- **B, function-grouped singles (leakage-free).** Genes are clustered into functional groups from their control-cell co-expression signature, and *entire groups* are withheld, so no gene from a held-out group ever appears in training. The A-to-B drop is the **leakage gap**.
- **C, combinatorial.** Train on all singles, test on all doubles (Norman only). A fixed mean vector structurally cannot predict an interaction, so this is the split where the learned model can outperform the baseline.

All splits are deterministic (BLAKE2b hashing), so they are identical across machines and runs.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Both datasets are the GEARS-format processed AnnData, downloaded once via GEARS `PertData`:

```python
from gears import PertData
PertData("data").load(data_name="norman")                 # -> data/norman/perturb_processed.h5ad
PertData("data").load(data_name="replogle_k562_essential")
```

Expression is log-normalized; `obs['condition']` is `GENE+ctrl` / `GENEA+GENEB` / `ctrl`. No synthetic or simulated data is used in any reported result; the only synthetic arrays in this repo are in `tests/`, as correctness checks.

## Reproduce the results

```bash
# strong + zero-effect baselines on the three splits
python src/scripts/run_baselines.py --h5ad data/norman/perturb_processed.h5ad \
    --out results/baselines_norman.json

# conditional VAE on the three splits (GPU)
python src/scripts/train.py --h5ad data/norman/perturb_processed.h5ad \
    --out results/cvae_norman_delta.json --epochs 40

# repeat both with data/replogle_k562_essential/perturb_processed.h5ad
# figures from the result JSONs
python src/scripts/make_figures.py
```

The saved `results/*.json` in this repo are the exact artifacts the report and figures are built from.

## Tests

```bash
pip install pytest && pytest -q
```

`tests/test_smoke.py` checks the splits, metrics, baselines, and model forward/predict paths on small synthetic arrays. These are correctness tests, explicitly labeled as such, not results.

## Citation

If this is useful, please cite the report:

```
@misc{znabu2026perturbvae,
  title  = {PerturbVAE: A Leakage-Aware Benchmark for Single-Cell Perturbation-Response Prediction},
  author = {Znabu, Brhanu F.},
  year   = {2026},
  note   = {https://github.com/brhanufen/perturbvae}
}
```

## License

MIT. See [`LICENSE`](LICENSE).
