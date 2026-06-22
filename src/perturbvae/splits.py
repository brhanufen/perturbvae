"""Held-out split design for perturbation-response generalization.

A ``condition`` is a string in the GEARS / scPerturb convention:

    'ctrl'            control (no perturbation)
    'GENE+ctrl'       single perturbation of GENE
    'GENEA+GENEB'     double (combinatorial) perturbation

We define three leakage-controlled splits. The contribution of this project is
the comparison between A and B: A is the optimistic number most papers report,
B is the leakage-free number, and the gap quantifies how much apparent
generalization is really interpolation between functionally similar
perturbations (the direct analog of phylogenetic-leakage controls).

    A. random        random 80/20 over single-gene conditions
    B. function      hold out entire gene groups (pathways/complexes)
    C. combinatorial train on singles, test on doubles
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

CTRL = "ctrl"


def perturbed_genes(condition: str) -> List[str]:
    """Genes actually perturbed in a condition (drops the 'ctrl' filler)."""
    return [g for g in str(condition).split("+") if g and g != CTRL]


def is_control(condition: str) -> bool:
    return len(perturbed_genes(condition)) == 0


def is_single(condition: str) -> bool:
    return len(perturbed_genes(condition)) == 1


def is_double(condition: str) -> bool:
    return len(perturbed_genes(condition)) == 2


def _hash_frac(key: str) -> float:
    """Deterministic uniform value in [0, 1) from a string key (BLAKE2b)."""
    h = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2 ** 64


def random_split(
    conditions: Sequence[str], val_fraction: float = 0.2, seed: str = "A",
) -> Tuple[List[str], List[str]]:
    """Split A: random held-out single-gene conditions, deterministic by hash."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")
    singles = sorted({c for c in conditions if is_single(c)})
    train, test = [], []
    for c in singles:
        (test if _hash_frac(f"{seed}|{c}") < val_fraction else train).append(c)
    return train, test


def function_split(
    conditions: Sequence[str],
    gene_to_group: Mapping[str, str],
    holdout_groups: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Split B (leakage-aware): hold out every single whose gene is in a
    held-out functional group. Genes with no group annotation are kept in train
    so the held-out set is purely the chosen groups.
    """
    holdout = set(holdout_groups)
    train, test = [], []
    for c in conditions:
        if not is_single(c):
            continue
        grp = gene_to_group.get(perturbed_genes(c)[0])
        (test if grp in holdout else train).append(c)
    return train, test


def choose_holdout_groups(
    gene_to_group: Mapping[str, str],
    perturbed: Sequence[str],
    target_fraction: float = 0.2,
    seed: str = "B",
) -> List[str]:
    """Pick whole groups to hold out so ~target_fraction of perturbed genes are
    withheld. Groups are selected by deterministic hash order until the quota is
    met, so the choice is reproducible and group-atomic (no gene from a held-out
    group ever appears in train)."""
    genes = [g for g in perturbed if g in gene_to_group]
    if not genes:
        return []
    from collections import Counter
    per_group = Counter(gene_to_group[g] for g in genes)
    total = sum(per_group.values())
    # Smallest groups first (tie-broken by hash) so the accumulated held-out
    # fraction creeps up to ~target with minimal overshoot, instead of a single
    # dominant cluster blowing past it.
    order = sorted(per_group, key=lambda grp: (per_group[grp], _hash_frac(f"{seed}|{grp}")))
    held, acc = [], 0
    for grp in order:
        held.append(grp)
        acc += per_group[grp]
        if acc >= target_fraction * total:
            break
    return held


def combinatorial_split(
    conditions: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Split C: train on all singles, test on all doubles (compositional)."""
    train = sorted({c for c in conditions if is_single(c)})
    test = sorted({c for c in conditions if is_double(c)})
    return train, test
