import os
import pickle
import numpy as np
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = "rag_chunks_clean.csv"

INDEX_DIR = "rag_index"

INDEX_FILE = os.path.join(
    INDEX_DIR,
    "hea_mpea_faiss.index"
)

METADATA_FILE = os.path.join(
    INDEX_DIR,
    "hea_mpea_faiss_metadata.pkl"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HEA/MPEA RAG - FAISS INDEX CREATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        print(f"ERROR: {INPUT_FILE} not found.")

        return

    df = pd.read_csv(INPUT_FILE)

    print(f"Chunks loaded: {len(df)}")
    print(f"Papers: {df['paper_file'].nunique()}")

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if df["text"].isna().any():

        print("ERROR: Empty text detected.")

        return

    if df["text"].duplicated().any():

        print("ERROR: Duplicate text detected.")

        return

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print("\nLoading embedding model:")

    print(MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded.")

    # --------------------------------------------------------
    # Prepare text
    # --------------------------------------------------------

    texts = df["text"].astype(str).tolist()

    print(f"\nGenerating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    print(
        f"Embedding matrix shape: {embeddings.shape}"
    )

    # --------------------------------------------------------
    # Create FAISS index
    # --------------------------------------------------------

    dimension = embeddings.shape[1]

    print(f"Embedding dimension: {dimension}")

    # Inner product + normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(f"FAISS vectors stored: {index.ntotal}")

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        INDEX_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save FAISS index
    # --------------------------------------------------------

    faiss.write_index(
        index,
        INDEX_FILE
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = df[
        [
            "paper_file",
            "page",
            "chunk_id",
            "word_count",
            "text"
        ]
    ].copy()

    with open(
        METADATA_FILE,
        "wb"
    ) as f:

        pickle.dump(
            metadata,
            f
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FAISS INDEX CREATION COMPLETE")
    print("=" * 70)

    print(f"Total vectors: {index.ntotal}")
    print(f"Vector dimension: {dimension}")
    print(f"Papers indexed: {metadata['paper_file'].nunique()}")

    print("\nFiles created:")

    print(f"  {INDEX_FILE}")
    print(f"  {METADATA_FILE}")


if __name__ == "__main__":
    main()