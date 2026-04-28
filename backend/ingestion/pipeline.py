from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


from backend.core.config import config
from backend.ingestion.chunker import chunk_regulation_pdf, load_chunks, save_chunks
from backend.ingestion.embedder import RegulationEmbedder
from backend.ingestion.fetcher import fetch_regulation_pdfs
from backend.ingestion.obligation_extractor import (
    ObligationExtractor,
    enrich_deadlines,
    save_obligations,
)


def run(force_redownload: bool = False, skip_extraction: bool = False) -> None:
    print(f"\n{'=' * 60}")
    print(f"CompliVibe Ingestion Pipeline")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'=' * 60}\n")

    regulations_processed = 0
    total_chunks = 0
    errors: dict[str, str] = {}

    print("[1/4] Fetching regulation PDFs...")
    try:
        pdf_paths = fetch_regulation_pdfs(force_redownload=force_redownload)
        print(f"✓ fetched {len(pdf_paths)} PDFs\n")
    except Exception as exc:
        print(f"✗ fetching failed: {exc}\n")
        return

    print("[2/4] Chunking and saving regulation PDFs...")
    chunk_files: dict[str, Path] = {}
    for regulation, pdf_path in pdf_paths.items():
        try:
            print(f"\n→ processing {regulation}...")
            chunks = chunk_regulation_pdf(regulation, pdf_path)
            output_path = save_chunks(regulation, chunks)
            chunk_files[regulation] = output_path
            regulations_processed += 1
            total_chunks += len(chunks)
            print(f"✓ {regulation}: {len(chunks)} chunks saved")
        except Exception as exc:
            error_msg = f"chunking failed: {exc}"
            errors[regulation] = error_msg
            print(f"✗ {regulation}: {error_msg}")

    print(f"\n✓ chunked {regulations_processed} regulations, {total_chunks} total chunks\n")

    print("[3/4] Embedding chunks into ChromaDB...")
    embedder = RegulationEmbedder()
    vectors_embedded = 0

    for regulation, chunk_path in chunk_files.items():
        try:
            print(f"\n→ embedding {regulation}...")
            chunks = load_chunks(chunk_path)
            embedder.embed_chunks(chunks)
            vectors_embedded += len(chunks)
            print(f"✓ {regulation}: {len(chunks)} vectors embedded")
        except Exception as exc:
            error_msg = f"embedding failed: {exc}"
            errors[regulation] = error_msg
            print(f"✗ {regulation}: {error_msg}")

    print(f"\n✓ embedded {vectors_embedded} vectors\n")

    print("[4/4] Extracting obligations...")
    obligations_total = 0
    obligations_by_regulation: dict[str, int] = {}
    if skip_extraction:
        print("⚠ extraction skipped (--skip-extraction)")
    else:
        extractor = ObligationExtractor()

        for regulation in pdf_paths.keys():
            try:
                chunk_path = config.processed_data_dir / f"{regulation}_chunks.json"
                chunks = load_chunks(chunk_path)
                obligations = extractor.extract_all(chunks)
                obligations = enrich_deadlines(obligations)
                save_obligations(obligations, regulation)
                obligations_total += len(obligations)
                obligations_by_regulation[regulation] = len(obligations)
                print(f"✓ {regulation}: {len(obligations)} obligations extracted")
            except Exception as exc:
                error_msg = f"obligation extraction failed: {exc}"
                errors[regulation] = error_msg
                print(f"✗ {regulation}: {error_msg}")

    stats = embedder.get_stats()
    print(f"{'=' * 60}")
    print("Pipeline Summary")
    print(f"{'=' * 60}")
    print(f"Regulations processed: {regulations_processed}")
    print(f"Total chunks: {total_chunks}")
    print(f"ChromaDB vector count: {sum(stats.values())}")
    print(f"Vectors per regulation: {stats}")
    print(f"Total obligations: {obligations_total}")
    print(f"Obligations per regulation: {obligations_by_regulation}")
    if errors:
        print(f"\nErrors encountered:")
        for regulation, error in errors.items():
            print(f"  • {regulation}: {error}")
    print(f"Completed: {datetime.now().isoformat()}")
    print(f"{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CompliVibe Ingestion Pipeline: Fetch, chunk, and embed regulations."
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force re-download of regulation PDFs even if cached.",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip obligation extraction if API quota is low.",
    )
    args = parser.parse_args()

    try:
        run(
            force_redownload=args.force_redownload,
            skip_extraction=args.skip_extraction,
        )
    except KeyboardInterrupt:
        print("\n\n✗ pipeline interrupted by user")
        sys.exit(1)
    except Exception as exc:
        print(f"\n\n✗ pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
