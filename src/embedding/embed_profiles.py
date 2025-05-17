#!/usr/bin/env python3
"""
embed_profiles.py

Reads synthetic profiles from a JSON file, computes text embeddings using
SentenceTransformers, and saves embeddings (and profile IDs) to a .npz file.

Usage:
    python embed_profiles.py \
        --input ../generator/data/test_profiles.json \
        --output embeddings.npz \
        --model all-MiniLM-L6-v2
"""

import argparse
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def load_profiles(path: str) -> list[dict]:
    """Load a list of profiles from a JSON file."""
    with open(path, 'r') as f:
        profiles = json.load(f)
    return profiles

def profile_to_text(profile: dict) -> str:
    """
    Build a single text string from a profile for embedding.
    Combines the bio with interests.
    """
    bio = profile.get("bio", "")
    interests = profile.get("interests", [])
    if interests:
        interests_str = ", ".join(interests)
        return f"{bio} Interests: {interests_str}"
    return bio

def embed_profiles(
    profiles: list[dict],
    model_name: str,
    batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute embeddings for each profile.
    Returns:
      - embeddings: array of shape (N, dim)
      - ids: integer array of shape (N,)
    """
    texts = [profile_to_text(p) for p in profiles]
    ids = np.array([p["id"] for p in profiles], dtype=int)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    return ids, embeddings

def main():
    parser = argparse.ArgumentParser(
        description="Embed synthetic dating-app profiles"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to the profiles JSON file"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="embeddings.npz",
        help="Path to save the embeddings (.npz format)"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="all-MiniLM-L6-v2",
        help="Name of the SentenceTransformer model to use"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for embedding computation"
    )
    args = parser.parse_args()

    print(f"Loading profiles from {args.input}...")
    profiles = load_profiles(args.input)
    print(f"Loaded {len(profiles)} profiles.")

    print(f"Encoding with model '{args.model}' (batch size {args.batch_size})...")
    ids, embeddings = embed_profiles(profiles, args.model, args.batch_size)
    print(f"Computed embeddings: shape {embeddings.shape}")

    print(f"Saving embeddings to {args.output}...")
    np.savez_compressed(args.output, ids=ids, embeddings=embeddings)
    print("Done.")

if __name__ == "__main__":
    main()
