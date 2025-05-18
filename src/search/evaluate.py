#!/usr/bin/env python3
"""
evaluate.py

Automate baseline metrics for your matching engine:
 - Loads embeddings.npz and profiles.json
 - Builds FAISS index
 - Defines synthetic ground-truth (shared interests + distance)
 - Runs NN search for every profile in test set
 - Computes Hit@K, Precision@K, MRR, MAP@K, NDCG@K
 - Prints summary and optionally exports CSV of per-query stats
"""

import argparse
import json
import math
import numpy as np
import faiss
from sklearn.metrics import ndcg_score
from typing import List, Set, Dict

def haversine(lat1, lon1, lat2, lon2):
    # distance in kilometers
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def load_data(emb_path: str, profiles_path: str):
    # 1) Load embeddings and cast IDs → native Python ints
    data = np.load(emb_path)
    raw_ids = data["ids"]                   # e.g. array([1,2,3,...], dtype=int64)
    ids = [int(i) for i in raw_ids]         # now a list of Python ints
    embs = data["embeddings"].astype("float32")

    # 2) Load the profiles JSON
    with open(profiles_path, "r") as f:
        profiles = json.load(f)

    # 3) Build a lookup { id: profile_dict }
    id2profile = {}
    for p in profiles:
        pid = int(p["id"])
        id2profile[pid] = p

    # 4) Gather lat/lon/interests in the *same order* as `ids`
    lat, lon, interests = [], [], []
    for pid in ids:
        prof = id2profile.get(pid)
        if prof is None:
            raise KeyError(f"Profile with id={pid} not found in JSON")
        # cast to float and default to 0.0 if missing
        lat.append(float(prof.get("latitude", 0.0)))
        lon.append(float(prof.get("longitude", 0.0)))
        interests.append(set(prof.get("interests", [])))

    # Convert back to numpy arrays
    lat = np.array(lat, dtype="float32")
    lon = np.array(lon, dtype="float32")

    return np.array(ids, dtype=int), embs, id2profile, lat, lon, interests

def build_index(embs: np.ndarray):
    faiss.normalize_L2(embs)
    d = embs.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(embs)
    return idx

def make_positives(
    ids: np.ndarray,
    lat: np.ndarray, lon: np.ndarray, interests: List[Set[str]],
    min_shared: int, max_km: float
) -> Dict[int, Set[int]]:
    """
    For each profile i, positives[i] = set of j such that:
      - share >= min_shared interests
      - haversine distance <= max_km
    """
    N = len(ids)
    positives = {int(ids[i]): set() for i in range(N)}
    for i in range(N):
        for j in range(i+1, N):
            shared = len(interests[i] & interests[j])
            dist = haversine(lat[i], lon[i], lat[j], lon[j])
            if shared >= min_shared and dist <= max_km:
                a, b = int(ids[i]), int(ids[j])
                positives[a].add(b)
                positives[b].add(a)
    return positives

def evaluate(
    ids: np.ndarray, embs: np.ndarray, idx: faiss.IndexFlatIP,
    positives: Dict[int, Set[int]], K_list: List[int]
):
    N = len(ids)
    # we'll store per-query metrics
    stats = {K: {"hit": [], "prec": [], "rr": [], "ap": []} for K in K_list}
    # for ndcg, we need full-score arrays
    y_true_full = []
    y_score_full = []

    for qi in range(N):
        qid = int(ids[qi])
        if len(positives[qid]) == 0:
            continue
        
        q_emb = embs[qi:qi+1]
        faiss.normalize_L2(q_emb)
        D, I = idx.search(q_emb, max(K_list)+1)  # get enough results
        neigh = I[0].tolist()
        scores = D[0].tolist()

        # drop self if present
        if neigh[0] == qi:
            neigh = neigh[1: max(K_list)+1+1]
            scores = scores[1: max(K_list)+1+1]

        # build binary relevance vector for full list
        # we'll only use it for NDCG
        rel = [1 if int(ids[n]) in positives[qid] else 0 for n in neigh]
        y_true_full.append(rel)
        y_score_full.append(scores)

        # compute metrics for each K
        for K in K_list:
            topk = neigh[:K]
            topk_rel = rel[:K]
            hits = sum(topk_rel) > 0
            prec = sum(topk_rel) / K
            # MRR
            try:
                first = topk_rel.index(1) + 1
                rr = 1.0 / first
            except ValueError:
                rr = 0.0
            # AP
            num_pos = len(positives[qid]) or 1
            ap = sum((sum(topk_rel[:i+1])/(i+1)) for i, v in enumerate(topk_rel) if v) / num_pos
            stats[K]["hit"].append(hits)
            stats[K]["prec"].append(prec)
            stats[K]["rr"].append(rr)
            stats[K]["ap"].append(ap)

    # aggregate
    results = {}
    for K in K_list:
        results[K] = {
            "Hit@{}".format(K): np.mean(stats[K]["hit"]),
            "Prec@{}".format(K): np.mean(stats[K]["prec"]),
            "MRR@{}".format(K): np.mean(stats[K]["rr"]),
            "MAP@{}".format(K): np.mean(stats[K]["ap"]),
        }
    # NDCG using sklearn
    ndcg = ndcg_score(np.array(y_true_full), np.array(y_score_full), k=max(K_list))
    results["NDCG@{}".format(max(K_list))] = ndcg
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", "-e", required=True)
    parser.add_argument("--profiles", "-p", required=True)
    parser.add_argument("--min-shared", type=int, default=2,
                        help="min shared interests for GT")
    parser.add_argument("--max-km", type=float, default=50,
                        help="max distance (km) for GT")
    parser.add_argument("--Ks", type=int, nargs="+", default=[1,5,10],
                        help="list of K values for Hit@K, etc.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    print("Loading data…")
    ids, embs, id2p, lat, lon, interests = load_data(args.embeddings, args.profiles)
    print(f"{len(ids)} profiles loaded.")

    print("Building FAISS index…")
    idx = build_index(embs)

    print("Building synthetic ground truth…")
    positives = make_positives(ids, lat, lon, interests,
                               min_shared=args.min_shared,
                               max_km=args.max_km)
    
    total_pairs = sum(len(v) for v in positives.values()) // 2
    avg_per_profile = sum(len(v) for v in positives.values()) / len(positives)
    print(f"Total positive pairs: {total_pairs}")
    print(f"Average positives per profile: {avg_per_profile:.2f}")
    
    print("Evaluating…")
    res = evaluate(ids, embs, idx, positives, args.Ks)

    print("\n=== Baseline Metrics ===")
    numeric_Ks = sorted(k for k in res.keys() if isinstance(k, int))
    for K in numeric_Ks:
        metrics = res[K]
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    # Then print any remaining string‐keyed metrics (e.g. NDCG)
    for key in sorted(k for k in res.keys() if isinstance(k, str)):
        print(f"{key}: {res[key]:.4f}")

if __name__ == "__main__":
    main()
