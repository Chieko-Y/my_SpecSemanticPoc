"""
data/chunks.jsonl の各チャンクをベクトル化し、data/embeddings.npy に保存する。

data/embeddings.npy の行順は chunks.jsonl の行順と一致させる
(indexで対応付け。search.py はこの前提でコサイン類似度を計算する)。
"""

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings.npy"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 64


def load_chunks() -> list[dict]:
    chunks = []
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not CHUNKS_PATH.exists():
        print(f"[ERROR] {CHUNKS_PATH} がありません。先に app/ingest.py を実行してください")
        sys.exit(1)

    chunks = load_chunks()
    print(f"チャンク数: {len(chunks)}")
    print(f"モデル読み込み中: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["clean_text"] for c in chunks]
    print("ベクトル化中...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    np.save(EMBEDDINGS_PATH, embeddings)

    print()
    print(f"件数: {embeddings.shape[0]}")
    print(f"次元数: {embeddings.shape[1]}")
    print(f"出力: {EMBEDDINGS_PATH}")

    if embeddings.shape[0] != len(chunks):
        print("[WARN] 件数がchunks.jsonlと一致していません")


if __name__ == "__main__":
    main()
