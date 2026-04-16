# Copyright 2026 Google LLC
"""Generate sample embeddings for Siemens Teamcenter RAG docs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.append(str(ROOT))

from rag_index import DOCS_PATH  # pylint: disable=wrong-import-position
from rag_index import embed_text  # pylint: disable=wrong-import-position
from rag_index import load_docs  # pylint: disable=wrong-import-position


def main() -> None:
  docs = load_docs(DOCS_PATH)
  rows = []
  for doc in docs:
    rows.append(
        {
            "doc_id": doc["doc_id"],
            "role": doc["role"],
            "source": doc["source"],
            "vector": embed_text(doc["content"]),
        }
    )

  out_path = ROOT / "data" / "sample_embeddings.json"
  out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
  print(f"Wrote {len(rows)} vectors to {out_path}")


if __name__ == "__main__":
  main()
