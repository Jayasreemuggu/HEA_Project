import os
import re
import pandas as pd
import pymupdf


# ============================================================
# PATHS
# ============================================================

PDF_DIR = "papers"
OUTPUT_DIR = "rag_chunks"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CHUNK SETTINGS
# ============================================================

CHUNK_SIZE = 800          # words approximately
CHUNK_OVERLAP = 150       # overlapping words


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean extracted PDF text while preserving scientific content.
    """

    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize repeated newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Fix hyphenated line breaks
    text = re.sub(r"-\n", "", text)

    # Replace single line breaks with spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    return text.strip()


# ============================================================
# CHUNKING
# ============================================================
def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    words = text.split()

    chunks = []

    start = 0
    chunk_id = 0

    while start < len(words):

        end = min(start + chunk_size, len(words))

        chunk_words = words[start:end]

        if len(chunk_words) < 50:
            break

        chunks.append({
            "chunk_id": chunk_id,
            "text": " ".join(chunk_words),
            "word_count": len(chunk_words)
        })

        chunk_id += 1

        # Stop when the final chunk has been reached
        if end >= len(words):
            break

        start = end - overlap

    return chunks

# ============================================================
# PROCESS ONE PDF
# ============================================================

def process_pdf(pdf_path):

    paper_name = os.path.basename(pdf_path)

    print(f"\nProcessing: {paper_name}", flush=True)

    doc = pymupdf.open(pdf_path)

    all_chunks = []
    global_chunk_id = 0
    total_pages = len(doc)

    print(f"Total pages: {total_pages}", flush=True)

    for page_number in range(total_pages):

        print(
            f"  Reading page {page_number + 1}/{total_pages}...",
            flush=True
        )

        try:
            page = doc[page_number]

            # Use text extraction directly
            raw_text = page.get_text("text", sort=True)

            if not raw_text:
                print("    No text found.", flush=True)
                continue

            text = clean_text(raw_text)

            print(
                f"    Extracted {len(text)} characters.",
                flush=True
            )

            page_chunks = create_chunks(text)

            for chunk in page_chunks:

                all_chunks.append({
                    "paper_file": paper_name,
                    "page": page_number + 1,
                    "chunk_id": global_chunk_id,
                    "word_count": chunk["word_count"],
                    "text": chunk["text"]
                })

                global_chunk_id += 1

            print(
                f"    Chunks: {len(page_chunks)}",
                flush=True
            )

        except Exception as e:

            print(
                f"    ERROR on page {page_number + 1}: {e}",
                flush=True
            )

    doc.close()

    print(
        f"Completed {paper_name}: {len(all_chunks)} chunks",
        flush=True
    )

    return all_chunks

# ============================================================
# MAIN
# ============================================================

def main():

    pdf_files = sorted([
        f for f in os.listdir(PDF_DIR)
        if f.lower().endswith(".pdf")
    ])

    print("=" * 70)
    print("HEA/MPEA RAG TEXT CHUNKING")
    print("=" * 70)

    print(f"PDF directory: {PDF_DIR}")
    print(f"PDF files found: {len(pdf_files)}")

    if not pdf_files:
        print("ERROR: No PDF files found.")
        return

    all_records = []

    for pdf_file in pdf_files:

        pdf_path = os.path.join(PDF_DIR, pdf_file)

        chunks = process_pdf(pdf_path)

        all_records.extend(chunks)

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    df = pd.DataFrame(all_records)

    output_csv = "rag_chunks.csv"

    df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Save individual paper chunks
    # --------------------------------------------------------

    for paper_name in df["paper_file"].unique():

        paper_df = df[df["paper_file"] == paper_name]

        safe_name = os.path.splitext(paper_name)[0]

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{safe_name}_chunks.csv"
        )

        paper_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RAG CHUNKING COMPLETE")
    print("=" * 70)

    print(f"Total chunks: {len(df)}")
    print(f"Total papers: {df['paper_file'].nunique()}")

    if len(df) > 0:

        print(
            f"Average words/chunk: "
            f"{df['word_count'].mean():.1f}"
        )

        print(
            f"Minimum words/chunk: "
            f"{df['word_count'].min()}"
        )

        print(
            f"Maximum words/chunk: "
            f"{df['word_count'].max()}"
        )

        print("\nChunks by paper:")

        print(
            df.groupby("paper_file")
              .size()
              .to_string()
        )

    print("\nSaved:")
    print(f"  {output_csv}")
    print(f"  {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()