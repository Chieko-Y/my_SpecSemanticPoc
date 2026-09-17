"""
チャンクのベクトルに対してクエリをコサイン類似度検索するロジック。
CLIからも FastAPI (main.py) からも同じ SearchIndex を使う。
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


class SearchIndex:
    def __init__(self) -> None:
        if not CHUNKS_PATH.exists() or not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                "data/chunks.jsonl または data/embeddings.npy がありません。"
                "先に app/ingest.py と app/embed.py を実行してください"
            )

        self.chunks: list[dict] = []
        with CHUNKS_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))

        self.embeddings = np.load(EMBEDDINGS_PATH)
        if self.embeddings.shape[0] != len(self.chunks):
            raise ValueError(
                f"chunks.jsonl({len(self.chunks)}件) と "
                f"embeddings.npy({self.embeddings.shape[0]}件) の件数が一致しません。"
                "app/embed.py を再実行してください"
            )

        self.model = SentenceTransformer(MODEL_NAME)

    def search(self, query: str, limit: int = 10, maker: str | None = None) -> list[dict]:
        query_vec = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)[0]

        scores = self.embeddings @ query_vec

        if maker:
            # maker指定時は対象チャンクのみを候補にしてから上位を取る
            # (先に全体の上位N件を取ってから絞り込むと、件数の少ないメーカーが
            #  候補から漏れて0件になってしまうため)
            mask = np.fromiter((c["maker"] == maker for c in self.chunks), dtype=bool, count=len(self.chunks))
            scores = np.where(mask, scores, -np.inf)

        top_indices = np.argsort(-scores)[:limit]

        results = []
        for idx in top_indices:
            if not np.isfinite(scores[idx]):
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "score": float(scores[idx]),
                    "chunk_id": chunk["chunk_id"],
                    "maker": chunk["maker"],
                    "model": chunk["model"],
                    "category": chunk["category"],
                    "doc_path": chunk["doc_path"],
                    "doc_name": chunk["doc_name"],
                    "heading_path": chunk["heading_path"],
                    "page_range": chunk.get("page_range", ""),
                    "excerpt": chunk["clean_text"][:400],
                }
            )
        return results


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) < 2:
        print("使い方: python app/search.py <質問文> [件数]")
        sys.exit(1)

    query = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print("インデックス読み込み中...")
    index = SearchIndex()

    print(f"クエリ: {query!r}")
    print()
    for i, r in enumerate(index.search(query, limit=limit), start=1):
        print(f"[{i}] score={r['score']:.3f}  {r['maker']}/{r['model']}  {r['doc_name']}")
        print(f"    heading_path: {r['heading_path']}")
        print(f"    excerpt: {r['excerpt'][:150]!r}")
        print()


if __name__ == "__main__":
    main()
