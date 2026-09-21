"""
チャンクのベクトルに対してクエリをコサイン類似度検索するロジック。
CLIからも FastAPI (main.py) からも同じ SearchIndex を使う。
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# `python app/search.py` のように直接実行された場合、app パッケージが
# import できるようプロジェクトルートをパスに足しておく
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.keyword_search import KeywordIndex, looks_like_model_code  # noqa: E402
from app.query_expansion import expand_query, load_groups  # noqa: E402
from app.scoring import get_current_policy  # noqa: E402
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings.npy"

CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

# 環境変数 SEARCH_BACKEND=chroma のときだけChromaDBで意味検索する(既定はnumpy総当たり)
BACKEND = os.environ.get("SEARCH_BACKEND", "numpy")

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

        self.collection = None
        if BACKEND == "chroma":
            import chromadb

            self.collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection("spec_chunks")

        self.model = SentenceTransformer(MODEL_NAME)
        self.expansion_groups = load_groups()
        self.keyword_index = KeywordIndex([c["clean_text"] for c in self.chunks])
        self.policy = get_current_policy()

    def _context(self, idx: int, offset: int, chars: int = 200) -> str:
        """idxの前後(offset=-1 or +1)のチャンクから、同じ文書内であれば文脈を切り出す。"""
        n = idx + offset
        if n < 0 or n >= len(self.chunks):
            return ""
        neighbor = self.chunks[n]
        if neighbor["doc_path"] != self.chunks[idx]["doc_path"]:
            return ""
        text = neighbor["clean_text"]
        return text[-chars:] if offset < 0 else text[:chars]

    def _maker_mask(self, maker: str) -> np.ndarray:
        return np.fromiter((c["maker"] == maker for c in self.chunks), dtype=bool, count=len(self.chunks))

    def _chroma_scores(self, query_vec: np.ndarray, limit: int, maker: str | None) -> np.ndarray:
        """ChromaDBの近似探索の上位を、numpy方式と同じ「全チャンク分のスコア配列」に直す。
        上位以外は -inf(候補外)。メーカー指定はChroma側のwhereで絞り込む。"""
        res = self.collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=min(limit, len(self.chunks)),
            where={"maker": maker} if maker else None,
        )
        scores = np.full(len(self.chunks), -np.inf, dtype=np.float32)
        for id_, dist in zip(res["ids"][0], res["distances"][0]):
            scores[int(id_)] = 1.0 - dist  # cosine距離 → 類似度
        return scores

    def search(self, query: str, limit: int = 10, maker: str | None = None) -> dict:
        expanded_query, added_terms = expand_query(query, self.expansion_groups)

        query_vec = self.model.encode(
            [expanded_query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)[0]

        mask = self._maker_mask(maker) if maker else None
        if self.collection is not None:
            scores = self._chroma_scores(query_vec, limit, maker)
        else:
            scores = self.embeddings @ query_vec
            if mask is not None:
                # maker指定時は対象チャンクのみを候補にしてから上位を取る
                # (先に全体の上位N件を取ってから絞り込むと、件数の少ないメーカーが
                #  候補から漏れて0件になってしまうため)
                scores = np.where(mask, scores, -np.inf)

        keyword_scores = self.keyword_index.scores(expanded_query)
        if mask is not None:
            keyword_scores = np.where(mask, keyword_scores, -np.inf)

        # キーワード検索への切替条件:
        #  ・意味検索の最高スコアが「該当なし」の水準、または
        #  ・クエリに型番らしい英数字(例: CVT-2000)が含まれる
        # ただしキーワードが1件も一致しなければ意味検索の結果をそのまま使う
        mode = "semantic"
        best_semantic = float(scores.max()) if scores.size else -np.inf
        weak_semantic = not self.policy.is_hit(best_semantic)
        if (weak_semantic or looks_like_model_code(query)) and float(keyword_scores.max()) > 0:
            mode = "keyword"
            scores = keyword_scores

        top_indices = np.argsort(-scores)[:limit]

        results = []
        for idx in top_indices:
            if not np.isfinite(scores[idx]) or (mode == "keyword" and scores[idx] <= 0):
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "score": float(scores[idx]),
                    "mode": mode,
                    "chunk_id": chunk["chunk_id"],
                    "maker": chunk["maker"],
                    "model": chunk["model"],
                    "category": chunk["category"],
                    "doc_path": chunk["doc_path"],
                    "doc_name": chunk["doc_name"],
                    "heading_path": chunk["heading_path"],
                    "page_range": chunk.get("page_range", ""),
                    "excerpt": chunk["clean_text"][:400],
                    "context_before": self._context(int(idx), -1),
                    "context_after": self._context(int(idx), 1),
                }
            )
        return {"results": results, "expanded_terms": added_terms, "mode": mode}


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

    result = index.search(query, limit=limit)
    print(f"クエリ: {query!r}")
    if result["expanded_terms"]:
        print(f"  拡張で追加された語: {result['expanded_terms']}")
    print()
    for i, r in enumerate(result["results"], start=1):
        print(f"[{i}] ({r['mode']}) score={r['score']:.3f}  {r['maker']}/{r['model']}  {r['doc_name']}")
        print(f"    heading_path: {r['heading_path']}")
        if r["context_before"]:
            print(f"    前の文脈: {r['context_before'][-80:]!r}")
        print(f"    excerpt: {r['excerpt'][:150]!r}")
        if r["context_after"]:
            print(f"    後の文脈: {r['context_after'][:80]!r}")
        print()


if __name__ == "__main__":
    main()
