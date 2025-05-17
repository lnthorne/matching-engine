## Vector Index & Nearest-Neighbour (NN) Search

### 1. Quick idea

A **vector index** (often called a _vector database_) is the data structure that lets us answer, lightning-fast:

> “Given a new vector **q**, which stored vectors are _closest_ to it?”

That query is a **nearest-neighbour search**. You usually ask for the _k_ nearest vectors (top-k ANN search).

---

### 2. Why we need an index

| Approach                                                                    | Complexity                                | When it breaks down           |
| --------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------- |
| **Brute force** – compare **q** to _every_ vector with a dot-product/cosine | `O(N · d)` (N = # vectors, d = dimension) | Slow once N ≫ 10 k            |
| **Indexed search** – prune most comparisons                                 | ~ `O(log N)` or bucket look-ups           | Requires up-front build + RAM |

For a dating-app engine comparing one user against 1 M+ candidates per swipe, brute force would saturate CPUs; an index keeps latency in the low-ms range.

---

### 3. Exact vs Approximate NN

| Strategy                                     | Guarantees                        | Speed                       | Use case                                      |
| -------------------------------------------- | --------------------------------- | --------------------------- | --------------------------------------------- |
| **Exact** (KD-tree, VP-tree)                 | True closest neighbours           | Slower at high-dim (>100 D) | Small sets, ≤ 30-D embeddings                 |
| **Approximate** (HNSW, IVF-PQ, ScaNN, Annoy) | _Very likely_ top-k (99 % recall) | 10-100× faster (tunable)    | Modern 128-1536 D embeddings, 100 k + vectors |

ANN is the default for text/image embeddings.

---

### 4. Popular libraries / services

| Library / SaaS                                                        | Core algorithm(s)             | Notes                                    |
| --------------------------------------------------------------------- | ----------------------------- | ---------------------------------------- |
| **Faiss**                                                             | IVF, HNSW, PQ (+ GPU)         | C++/Python; de-facto research standard   |
| **ScaNN**                                                             | Tree-AH, asymmetric hash      | Optimised for CPUs/TPUs; backs Vertex AI |
| **Annoy**                                                             | Random-projection trees       | Great read-only index for millions       |
| **HNSWlib**                                                           | HNSW                          | High recall, dynamic inserts             |
| **Vector DBs** (Pinecone, Qdrant, Weaviate, pgvector, OpenSearch-kNN) | Wrap one or more of the above | Give you persistence, filtering, auth    |

---

### 5. Prototype query flow

```mermaid
graph LR
    A[Embed seeker profile → vector **q**] --> B{Metadata filter<br>(age, distance)}
    B --> C[(Vector index)]
    C -->|top-K ANN| D[Candidate list]
    D --> E{Re-rank<br>business rules}
    E --> F[Results to client]
```
