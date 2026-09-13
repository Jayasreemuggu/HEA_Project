import pandas as pd


INPUT_FILE = "rag_chunks.csv"
OUTPUT_FILE = "rag_chunks_clean.csv"


def main():

    print("=" * 70)
    print("HEA/MPEA RAG CHUNK CLEANING")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    print(f"Original chunks: {len(df)}")

    # --------------------------------------------------------
    # Remove empty text
    # --------------------------------------------------------
    before = len(df)

    # Remove duplicates only within the same paper.
    # Identical text appearing in different papers is retained
    # because each paper is an independent literature source.

    df = df.drop_duplicates(
        subset=["paper_file", "text"],
        keep="first"
    ).copy()

    print(f"Removed duplicate chunks within papers: {before - len(df)}")

    # --------------------------------------------------------
    # Remove duplicates only within the same paper
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=["paper_file", "text"],
        keep="first"
    ).copy()

    print(
        f"Removed duplicate chunks within papers: "
        f"{before - len(df)}"
    )
    
    # --------------------------------------------------------
    # Remove very short chunks
    # --------------------------------------------------------

    before = len(df)

    df = df[df["word_count"] >= 100].copy()

    print(f"Removed chunks <100 words: {before - len(df)}")

    # --------------------------------------------------------
    # Recreate clean chunk IDs
    # --------------------------------------------------------

    df = df.reset_index(drop=True)

    df["chunk_id"] = range(len(df))

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLEANING COMPLETE")
    print("=" * 70)

    print(f"Final chunks: {len(df)}")
    print(f"Papers: {df['paper_file'].nunique()}")
    print(f"Minimum words: {df['word_count'].min()}")
    print(f"Maximum words: {df['word_count'].max()}")
    print(f"Average words: {df['word_count'].mean():.1f}")

    print("\nChunks by paper:")
    print(
        df.groupby("paper_file")
          .size()
          .to_string()
    )

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()