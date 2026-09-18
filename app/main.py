"""FastAPIエントリポイント。/api/search・/api/stats と検索画面を提供する。"""

from collections import Counter
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.facets import FacetsNotDeclaredError, load_axes
from app.scoring import get_current_policy
from app.search import SearchIndex

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "app" / "static"

POLICY = get_current_policy()

try:
    # data/facets.json で「maker」が絞り込み軸として宣言されている場合のみ、
    # メーカー絞り込み機能(画面のチップ・APIのmakerパラメータ)を有効にする。
    # 宣言が無い/読めないデータに対しては、誤ったメーカー名を表示しないよう無効化する。
    HAS_MAKER_AXIS = "maker" in load_axes()
except FacetsNotDeclaredError:
    HAS_MAKER_AXIS = False

app = FastAPI(title="仕様書 類似検索")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_index: SearchIndex | None = None


def get_index() -> SearchIndex:
    global _index
    if _index is None:
        _index = SearchIndex()
    return _index


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    maker: str | None = None


@app.on_event("startup")
def _warm_up() -> None:
    get_index()


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/stats")
def stats() -> dict:
    index = get_index()
    total_chunks = len(index.chunks)

    makers: list[dict] = []
    if HAS_MAKER_AXIS:
        doc_paths_by_maker: dict[str, set] = {}
        chunk_counts: Counter = Counter()
        for c in index.chunks:
            doc_paths_by_maker.setdefault(c["maker"], set()).add(c["doc_path"])
            chunk_counts[c["maker"]] += 1

        makers = [
            {
                "maker": maker,
                "doc_count": len(doc_paths_by_maker[maker]),
                "chunk_count": chunk_counts[maker],
            }
            for maker in sorted(doc_paths_by_maker)
        ]
        total_docs = sum(m["doc_count"] for m in makers)
    else:
        # maker軸が宣言されていないデータでは、文書数は doc_path のユニーク数で数える
        total_docs = len({c["doc_path"] for c in index.chunks})

    return {
        "makers": makers,
        "has_maker_axis": HAS_MAKER_AXIS,
        "total_docs": total_docs,
        "total_chunks": total_chunks,
        "policy_version": POLICY.version,
    }


@app.post("/api/search")
def search(req: SearchRequest) -> dict:
    index = get_index()
    query = req.query.strip()
    if not query:
        return {"query": req.query, "results": []}

    maker = req.maker if HAS_MAKER_AXIS else None
    search_result = index.search(query, limit=req.limit, maker=maker)

    results = []
    for r in search_result["results"]:
        r["grade"] = POLICY.grade_for_score(r["score"])
        r["hit"] = POLICY.is_hit(r["score"])
        results.append(r)

    return {
        "query": req.query,
        "results": results,
        "expanded_terms": search_result["expanded_terms"],
        "policy_version": POLICY.version,
    }
