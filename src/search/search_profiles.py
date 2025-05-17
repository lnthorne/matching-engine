#!/usr/bin/env python3
"""
search_profiles.py

Load profile embeddings and metadata, build a FAISS index, and
perform nearest‐neighbor search to retrieve similar profiles.

Usage:
    python search_profiles.py \
      --embeddings ../embedding/embeddings.npz \
      --profiles ../generator/data/test_profiles.json \
      --query-id 42 \
      --top-k 5
"""

import argparse
import json
import numpy as np
import faiss

def load_data(emb_path: str, profiles_path: str):
    # Load embeddings and IDs
    data = np.load(emb_path)
    ids = data["ids"]            # shape (N,)
    embeddings = data["embeddings"].astype("float32")  # shape (N, D)
    # Load profile metadata
    with open(profiles_path, "r") as f:
        profiles = json.load(f)
    # Map profile ID → profile dict
    id2profile = {p["id"]: p for p in profiles}
    return ids, embeddings, id2profile

def build_faiss_index(embeddings: np.ndarray):
    d = embeddings.shape[1]
    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(d)   # inner‐product = cosine on normalized vectors
    index.add(embeddings)
    return index

def search_by_profile_id(query_id: int,
                         ids: np.ndarray,
                         embeddings: np.ndarray,
                         index: faiss.IndexFlatIP,
                         id2profile: dict,
                         top_k: int):
    # Find row in embeddings corresponding to query_id
    idx = np.where(ids == query_id)[0]
    if idx.size == 0:
        raise ValueError(f"No profile found with id={query_id}")
    # Prepare query vector
    q_vec = embeddings[idx]
    faiss.normalize_L2(q_vec)
    # Search (we request top_k+1 and then skip the query itself)
    D, I = index.search(q_vec, top_k + 1)
    neighbors = I[0]
    scores = D[0]
    results = []
    for nb_idx, score in zip(neighbors, scores):
        nb_id = int(ids[nb_idx])
        if nb_id == query_id:
            continue
        results.append({
            "profile": id2profile[nb_id],
            "score": float(score)
        })
        if len(results) >= top_k:
            break
    return results

def main():
    parser = argparse.ArgumentParser(description="Nearest‐neighbor search on profile embeddings")
    parser.add_argument("--embeddings", "-e", required=True,
                        help="Path to embeddings (.npz)")
    parser.add_argument("--profiles", "-p", required=True,
                        help="Path to profiles JSON")
    parser.add_argument("--query-id", "-q", type=int, required=True,
                        help="Profile ID to query")
    parser.add_argument("--top-k", "-k", type=int, default=5,
                        help="Number of nearest neighbors to return")
    args = parser.parse_args()

    print("Loading data...")
    ids, embeddings, id2profile = load_data(args.embeddings, args.profiles)
    print(f"Loaded {len(ids)} embeddings.")

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print(f"Searching for top {args.top_k} neighbors of profile {args.query_id}...")
    neighbors = search_by_profile_id(
        args.query_id, ids, embeddings, index, id2profile, args.top_k
    )

    for rank, hit in enumerate(neighbors, start=1):
        prof = hit["profile"]
        score = hit["score"]
        print(f"\nRank {rank} (score={score:.4f}):")
        print(f"  ID:      {prof['id']}")
        print(f"  Name:    {prof.get('name', 'N/A')}")
        print(f"  Age:     {prof.get('age', 'N/A')}")
        print(f"  City:    {prof.get('city', 'N/A')}")
        print(f"  Bio:     {prof['bio']}")
        print(f"  Interests: {', '.join(prof['interests'])}")

if __name__ == "__main__":
    main()
