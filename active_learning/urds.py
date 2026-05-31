#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Uncertainty-Reinforced Diversity Sampling (URDS).

URDS is the core one-shot cold-start query strategy of CSAL-3D. It combines
*diversity* and *uncertainty* in a hierarchical process (paper Sec. 2, Eq. 9-12):

1. Cluster the SSL feature bank into ``M`` clusters with multiple-kernel
   k-Means, where ``M`` is the annotation budget (Eq. 9).
2. **Typicality-gated diverse candidate selection** -- within each cluster,
   keep the top ``N_cand`` samples ranked by typicality, i.e. the inverse
   average cosine distance to the rest of the cluster (Eq. 10-11).
3. **Uncertainty-guided final selection** -- among the candidates of each
   cluster, pick the single most uncertain sample using the SSL-driven
   ensemble uncertainty score ``S(X)`` (Eq. 12).

Two ablation variants from the paper (Table 1) are also provided:

* ``urds``     : the full hierarchical strategy described above.
* ``div-only`` : diversity only -- pick the most typical sample per cluster.
* ``unc-only`` : uncertainty only -- pick the ``M`` globally most uncertain samples.

Example
-------
    python -m active_learning.urds urds \
        --organ BrainTumour \
        --feats  ./BrainTumour/feats/Ours.npz \
        --scores ./BrainTumour/feats/Ours_scores.tsv \
        --num-samples 20 --ncand 3
"""
from __future__ import annotations

import argparse
import os.path as osp

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances

GAMMA_VALUES = (0.1, 1, 10)


def read_tsv(file_path: str) -> np.ndarray:
    """Read a one-score-per-line ``.tsv`` file into a 1-D array."""
    with open(file_path, "r") as f:
        return np.array([float(line.strip()) for line in f])


def load_features(feats_file: str):
    """Load ``feats`` and ``name_list`` arrays from an ``.npz`` feature file."""
    data = dict(np.load(feats_file, allow_pickle=True))
    return np.array(data["feats"]), np.array(data["name_list"])


# --------------------------------------------------------------------------- #
# Multiple-kernel k-Means clustering (Eq. 9)                                  #
# --------------------------------------------------------------------------- #
def construct_kernels(feats: np.ndarray):
    """Build a bank of RBF kernels at multiple bandwidths."""
    kernels = []
    for gamma in GAMMA_VALUES:
        pairwise_dist = pairwise_distances(feats, metric="euclidean")
        kernels.append(np.exp(-gamma * pairwise_dist ** 2 / 2))
    return kernels


def multiple_kernel_kmeans(kernels, k: int, max_iter: int = 100):
    """Multiple-kernel k-Means in the combined kernel space."""
    n_samples = kernels[0].shape[0]
    assignments = np.random.choice(k, n_samples)
    combined_kernel = np.sum(kernels, axis=0)

    for _ in range(max_iter):
        new_assignments = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            distances = np.full(k, np.inf)
            for j in range(k):
                in_cluster = assignments == j
                size = np.sum(in_cluster)
                if size == 0:
                    continue
                idx = np.where(in_cluster)[0]
                distances[j] = (
                    combined_kernel[i, i]
                    - 2 * np.sum(combined_kernel[i, idx]) / size
                    + np.sum(combined_kernel[np.ix_(idx, idx)]) / (size ** 2)
                )
            new_assignments[i] = np.argmin(distances)

        # Re-seed any empty cluster to keep exactly k clusters alive.
        for j in range(k):
            if np.sum(new_assignments == j) == 0:
                new_assignments[np.random.choice(n_samples)] = j

        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments

    return assignments


def typicality(points: np.ndarray) -> np.ndarray:
    """Typicality (Eq. 10): inverse mean cosine distance within the cluster."""
    distances = cosine_distances(points)
    return 1 / (np.mean(distances, axis=1) + 1e-10)


# --------------------------------------------------------------------------- #
# Sample-selection variants (Table 1)                                         #
# --------------------------------------------------------------------------- #
def select_urds(feats, scores, name_list, num_samples, ncand):
    """Full URDS: cluster -> typicality-gated candidates -> most uncertain."""
    labels = multiple_kernel_kmeans(construct_kernels(feats), num_samples)

    selected = []
    for m in range(num_samples):
        mask = labels == m
        if not np.any(mask):
            continue
        cluster_names = name_list[mask]
        cluster_scores = scores[mask]

        # Eq. 11: top-N_cand typical samples form the candidate set.
        cand_idx = np.argsort(typicality(feats[mask]))[-ncand:]
        # Eq. 12: among candidates, take the most uncertain one.
        chosen = cand_idx[np.argmax(cluster_scores[cand_idx])]
        selected.append(np.where(name_list == cluster_names[chosen])[0][0])
    return np.array(selected)


def select_div_only(feats, name_list, num_samples):
    """Div-only ablation: most typical sample per cluster (diversity only)."""
    labels = multiple_kernel_kmeans(construct_kernels(feats), num_samples)

    selected = []
    for m in range(num_samples):
        mask = labels == m
        if not np.any(mask):
            continue
        cluster_names = name_list[mask]
        chosen = np.argmax(typicality(feats[mask]))
        selected.append(np.where(name_list == cluster_names[chosen])[0][0])
    return np.array(selected)


def select_unc_only(scores, num_samples):
    """Unc-only ablation: the M globally most uncertain samples."""
    return np.argsort(scores)[-num_samples:]


# --------------------------------------------------------------------------- #
# Plan generation                                                             #
# --------------------------------------------------------------------------- #
def generate_plan(args):
    np.random.seed(args.seed)
    feats, name_list = load_features(args.feats)

    if args.variant == "urds":
        scores = read_tsv(args.scores)
        idx = select_urds(feats, scores, name_list, args.num_samples, args.ncand)
        tag = f"URDS_{args.num_samples}"
    elif args.variant == "div-only":
        idx = select_div_only(feats, name_list, args.num_samples)
        tag = f"URDS_div_only_{args.num_samples}"
    else:  # unc-only
        scores = read_tsv(args.scores)
        idx = select_unc_only(scores, args.num_samples)
        tag = f"URDS_unc_only_{args.num_samples}"

    paths = [osp.join(f"./{args.organ}/data/{pid}.npz") for pid in name_list[idx]]
    save_path = args.output or f"./{args.organ}/plans/{tag}.npz"
    np.savez(save_path, paths=paths)
    print(f"[URDS:{args.variant}] selected {len(paths)} samples -> {save_path}")
    for p in paths:
        print(f"  {p}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "variant",
        choices=["urds", "div-only", "unc-only"],
        help="full URDS or an ablation variant",
    )
    parser.add_argument("--organ", required=True, help="dataset / organ name")
    parser.add_argument("--feats", required=True, help="path to the .npz feature file")
    parser.add_argument("--scores", help="path to the .tsv uncertainty score file")
    parser.add_argument("--num-samples", type=int, required=True, help="annotation budget M")
    parser.add_argument("--ncand", type=int, default=3, help="candidate number N_cand (urds)")
    parser.add_argument("--output", help="output .npz plan path (auto-generated if omitted)")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    return parser


def main():
    args = build_parser().parse_args()
    if args.variant in ("urds", "unc-only") and not args.scores:
        raise SystemExit(f"--scores is required for the '{args.variant}' variant")
    generate_plan(args)


if __name__ == "__main__":
    main()
