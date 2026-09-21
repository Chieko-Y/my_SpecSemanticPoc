"""
data/chunks.jsonl と data/embeddings.npy から、ChromaDBの索引(data/chroma/)を作る。
embed.py の実行後(データを変更したとき)に実行し直す。
"""

import json
import shutil
import sys
from pathlib import Path

import chromadb
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings.npy"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION = "spec_chunks"
BATCH = 500


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    chunks = [json.loads(l) for l in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    embeddings = np.load(EMBEDDINGS_PATH)
    if len(chunks) != embeddings.shape[0]:
        raise SystemExit("chunks.jsonl と embeddings.npy の件数が一致しません。embed.py を再実行してください")

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # cosine: 正規化済みベクトルなので 類似度 = 1 - distance で今の方式と同じスコアになる
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    for start in range(0, len(chunks), BATCH):
        end = min(start + BATCH, len(chunks))
        col.add(
            # idは「chunks.jsonl内の行番号」。検索結果を元のチャンクに対応づけるため
            ids=[str(i) for i in range(start, end)],
            embeddings=embeddings[start:end].tolist(),
            metadatas=[{"maker": chunks[i]["maker"]} for i in range(start, end)],
        )
    print(f"ChromaDBに{col.count()}件を登録しました: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
