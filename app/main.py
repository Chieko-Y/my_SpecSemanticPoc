"""FastAPIエントリポイント。/api/search・/api/stats と検索画面を提供する。"""

from collections import Counter
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.search import SearchIndex

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "app" / "static"

SCORE_THRESHOLD = 0.30

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


def grade_for_score(score: float) -> str:
    if score >= 0.70:
        return "A"
    if score >= 0.50:
        return "B"
    if score >= SCORE_THRESHOLD:
        return "C"
    return "D"


@app.on_event("startup")
def _warm_up() -> None:
    get_index()


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/stats")
def stats() -> dict:
    index = get_index()
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
    total_chunks = len(index.chunks)
    return {"makers": makers, "total_docs": total_docs, "total_chunks": total_chunks}


@app.post("/api/search")
def search(req: SearchRequest) -> dict:
    index = get_index()
    query = req.query.strip()
    if not query:
        return {"query": req.query, "results": []}

    raw_results = index.search(query, limit=req.limit, maker=req.maker)

    results = []
    for r in raw_results:
        r["grade"] = grade_for_score(r["score"])
        r["hit"] = r["score"] >= SCORE_THRESHOLD
        results.append(r)

    return {"query": req.query, "results": results}
