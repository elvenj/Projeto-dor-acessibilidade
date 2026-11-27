import sys
import json
import hashlib

from pathlib import Path
from collections import Counter

import chromadb

from Services.orm import ingestion
from Database.chromaDb.chroma import collection as original


def rebuild_index(destination):

    destination = Path(destination)

    client = chromadb.PersistentClient(
        path=str(destination)
    )

    # Preservar a métrica de distância do índice original

    metadata = dict(original.metadata or {})
    configuration = getattr(original, "configuration", {})

    if isinstance(configuration, dict):

        hnsw = configuration.get("hnsw") or {}

        if hnsw.get("space"):
            metadata["hnsw:space"] = hnsw["space"]

    target = client.create_collection(
        name=original.name,
        metadata=metadata or None
    )

    pdfs = sorted(
        Path(path) for path in ingestion.load_pdfs()
    )

    assert pdfs, "Nenhum PDF encontrado."

    # Usar o pipeline original de ingestão no novo índice

    ingestion.collection = target

    seen = set()
    duplicates = 0
    empty = 0
    processed = 0
    sources = set()

    for pdf in pdfs:

        fingerprint = hashlib.sha256(
            pdf.read_bytes()
        ).hexdigest()

        if fingerprint in seen:

            duplicates += 1
            continue

        seen.add(fingerprint)
        sources.add(pdf.name)

        before = target.count()

        ingestion.process_pdf(pdf)

        added = target.count() - before

        if added > 0:
            processed += 1
        else:
            empty += 1

        print(pdf.name, ":", added, "chunks", flush=True)

    # Validar proveniência de todos os chunks

    total = target.count()

    assert total > 0, "O novo índice está vazio."

    valid = 0
    source_counts = Counter()

    for offset in range(0, total, 500):

        batch = target.get(
            limit=500,
            offset=offset,
            include=["metadatas"]
        )

        for metadata in batch["metadatas"]:

            metadata = metadata or {}

            source = metadata.get("source")
            page = metadata.get("page")

            if (
                source
                and Path(str(source)).name in sources
                and str(page).isdigit()
            ):

                valid += 1
                source_counts[str(source)] += 1

    report = {
        "pdfs_found": len(pdfs),
        "pdfs_processed": processed,
        "duplicate_pdfs": duplicates,
        "pdfs_without_new_chunks": empty,
        "total_chunks": total,
        "chunks_with_provenance": valid,
        "chunks_without_provenance": total - valid,
        "sources": dict(source_counts),
        "index_path": str(destination),
        "collection": target.name
    }

    assert valid == total, report

    (destination / "manifest.json").write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print(json.dumps(report, indent=4, ensure_ascii=False))

    return report


if __name__ == "__main__":
    rebuild_index(sys.argv[1])
