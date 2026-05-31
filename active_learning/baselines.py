#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline cold-start active-learning sample-selection strategies.

This module collects the unsupervised / diversity-based baselines that CSAL-3D
is compared against. Each strategy consumes a pre-computed ``.npz`` feature file
(``feats`` + ``name_list``) and writes a selection plan (a ``.npz`` listing the
chosen volume paths) under ``./{organ}/plans/``.

Supported methods: ``fps``, ``typiclust``, ``calr``, ``alps``, ``probcover``,
``usl``, ``usl-t``, ``coreset``.

Example
-------
    python -m active_learning.baselines typiclust \
        --organ Heart --feats ./Heart/feats/Ours.npz --num-samples 5
"""
from __future__ import annotations

import argparse
import os.path as osp
import random

import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.cluster import Birch, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def load_features(feats_file: str):
    data = dict(np.load(feats_file, allow_pickle=True))
    return np.array(data["feats"]), np.array(data["name_list"])


# --------------------------------------------------------------------------- #
# FPS (cluster + farthest pair)                                               #
# --------------------------------------------------------------------------- #
def select_fps(feats, name_list, num_samples):
    num_clusters = num_samples // 2
    labels = KMeans(n_clusters=num_clusters, random_state=0).fit(feats).labels_

    selected = []
    for i in range(num_clusters):
        idx = np.where(labels == i)[0]
        distances = cdist(feats[idx], feats[idx], metric="euclidean")
        a, b = np.unravel_index(np.argmax(distances), distances.shape)
        selected.append(name_list[idx[a]])
        selected.append(name_list[idx[b]])

    if num_samples % 2 == 1:
        last = np.where(labels == num_clusters - 1)[0]
        selected.append(name_list[last[0]])
    return selected[:num_samples]


# --------------------------------------------------------------------------- #
# TypiClust (cluster + typicality)                                            #
# --------------------------------------------------------------------------- #
def _typicality(cluster_feats):
    return 1 / np.mean(cdist(cluster_feats, cluster_feats, metric="euclidean"), axis=1)


def select_typiclust(feats, name_list, num_samples):
    labels = KMeans(n_clusters=num_samples, random_state=0).fit(feats).labels_
    selected = []
    for i in range(num_samples):
        idx = np.where(labels == i)[0]
        selected.append(name_list[idx[np.argmax(_typicality(feats[idx]))]])
    return selected


# --------------------------------------------------------------------------- #
# CALR (BIRCH + information density)                                          #
# --------------------------------------------------------------------------- #
def _information_density(cluster_feats):
    return cosine_similarity(cluster_feats).mean(axis=1)


def select_calr(feats, name_list, num_samples):
    labels = Birch(n_clusters=num_samples).fit(feats).labels_
    selected = []
    for i in range(num_samples):
        idx = np.where(labels == i)[0]
        selected.append(name_list[idx[np.argmax(_information_density(feats[idx]))]])
    return selected


# --------------------------------------------------------------------------- #
# ALPS (cluster + closest-to-centroid)                                        #
# --------------------------------------------------------------------------- #
def select_alps(feats, name_list, num_samples):
    kmeans = KMeans(n_clusters=num_samples, random_state=0).fit(feats)
    labels, centers = kmeans.labels_, kmeans.cluster_centers_
    selected = []
    for i in range(num_samples):
        idx = np.where(labels == i)[0]
        distances = np.linalg.norm(feats[idx] - centers[i], axis=1)
        selected.append(name_list[idx[np.argmin(distances)]])
    return selected


# --------------------------------------------------------------------------- #
# ProbCover (max-coverage graph greedy)                                       #
# --------------------------------------------------------------------------- #
def _estimate_delta(embedding, num_classes, alpha=0.95):
    cluster_labels = KMeans(n_clusters=num_classes).fit_predict(embedding)
    dist_matrix = distance_matrix(embedding, embedding)
    best_delta = 0
    for delta in np.linspace(0, np.max(dist_matrix), num=100):
        pure_balls = total_balls = 0
        for i in range(len(embedding)):
            neighbors = np.where(dist_matrix[i] <= delta)[0]
            if len(neighbors) > 0:
                if np.all(cluster_labels[neighbors] == cluster_labels[i]):
                    pure_balls += 1
                total_balls += 1
        if pure_balls / total_balls >= alpha:
            best_delta = delta
        else:
            break
    return best_delta


def select_probcover(feats, name_list, num_samples, delta=None, alpha=0.95):
    if delta is None:
        delta = _estimate_delta(feats, num_samples, alpha)
    dist_matrix = distance_matrix(feats, feats)
    adjacency = (dist_matrix <= delta).astype(int)

    selected = []
    for _ in range(num_samples):
        out_degrees = adjacency.sum(axis=1)
        node = np.argmax(out_degrees)
        selected.append(node)
        covered = np.where(adjacency[node] > 0)[0]
        adjacency[:, covered] = 0
        adjacency[covered, :] = 0
    return [name_list[i] for i in selected]


# --------------------------------------------------------------------------- #
# USL / USL-T                                                                 #
# --------------------------------------------------------------------------- #
def select_usl(feats, name_list, num_samples, k=5, alpha=2,
               num_iters=10, lambda_reg=0.5, epsilon=1e-10):
    n = feats.shape[0]
    dist = np.linalg.norm(feats[:, None, :] - feats[None, :, :], axis=2)
    avg_distances = np.array([np.mean(np.sort(dist[i])[1:k + 1]) for i in range(n)])
    density = 1 / (avg_distances + epsilon)

    selected = []
    reg_values = np.zeros(n)
    for _ in range(num_iters):
        for i in range(n):
            reg_values[i] = sum(
                1 / ((np.linalg.norm(feats[i] - feats[j]) ** alpha) + epsilon)
                for j in selected
            )
        utility = density - lambda_reg * reg_values
        selected = np.argsort(utility)[-num_samples:].tolist()
    return [name_list[i] for i in selected]


def select_usl_t(feats, name_list, num_samples, k=5, num_iters=10,
                 lambda_reg=0.5, t=0.25, tau=0.5, epsilon=1e-10):
    n, _ = feats.shape
    centroids = feats[random.sample(range(n), num_samples)]
    dist = np.linalg.norm(feats[:, None, :] - feats[None, :, :], axis=2)
    nearest_neighbors = np.argsort(dist, axis=1)[:, 1:k + 1]

    cluster_confidences = np.zeros(n)
    for _ in range(num_iters):
        soft_assignments = softmax(np.dot(feats, centroids.T) / t, axis=1)

        confident = np.max(soft_assignments, axis=1) >= tau
        conf_assign, conf_feats = soft_assignments[confident], feats[confident]
        for j in range(num_samples):
            w = conf_assign[:, j]
            centroids[j] = np.sum(w[:, None] * conf_feats, axis=0) / (np.sum(w) + epsilon)

        local_reg = np.zeros((n, num_samples))
        for i in range(n):
            neighbor_avg = np.mean(soft_assignments[nearest_neighbors[i]], axis=0)
            local_reg[i] = softmax(neighbor_avg / t)

        combined = lambda_reg * local_reg + soft_assignments
        cluster_confidences = np.max(combined, axis=1)

    selected = np.argsort(-cluster_confidences)[:num_samples]
    return [name_list[i] for i in selected]


# --------------------------------------------------------------------------- #
# Core-Set (k-center greedy)                                                  #
# --------------------------------------------------------------------------- #
def select_coreset(feats, name_list, num_samples):
    n = len(name_list)
    dist_mat = np.matmul(feats, feats.T)
    diag = np.diag(dist_mat).reshape(-1, 1)
    dist_mat = np.sqrt(np.clip(-2 * dist_mat + diag + diag.T, 0, None))

    first = np.random.choice(np.arange(n))
    min_distances = dist_mat[:, first]
    selected = [name_list[first]]
    for _ in tqdm(range(num_samples - 1), ncols=100):
        nxt = np.argmax(min_distances)
        selected.append(name_list[nxt])
        min_distances = np.minimum(min_distances, dist_mat[:, nxt])
    return selected


METHODS = {
    "fps": select_fps,
    "typiclust": select_typiclust,
    "calr": select_calr,
    "alps": select_alps,
    "probcover": select_probcover,
    "usl": select_usl,
    "usl-t": select_usl_t,
    "coreset": select_coreset,
}


def generate_plan(args):
    np.random.seed(args.seed)
    feats, name_list = load_features(args.feats)
    selected = METHODS[args.method](feats, name_list, args.num_samples)

    paths = [osp.join(f"./{args.organ}/data/{pid}.npz") for pid in selected]
    tag = f"{args.method}_{args.num_samples}"
    save_path = args.output or f"./{args.organ}/plans/{tag}.npz"
    np.savez(save_path, paths=paths)
    print(f"[{args.method}] selected {len(paths)} samples -> {save_path}")
    for p in paths:
        print(f"  {p}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=sorted(METHODS), help="baseline strategy")
    parser.add_argument("--organ", required=True, help="dataset / organ name")
    parser.add_argument("--feats", required=True, help="path to the .npz feature file")
    parser.add_argument("--num-samples", type=int, required=True, help="annotation budget")
    parser.add_argument("--output", help="output .npz plan path (auto-generated if omitted)")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    return parser


def main():
    generate_plan(build_parser().parse_args())


if __name__ == "__main__":
    main()
