"""
data/ 配下のMarkdown仕様書 -> 見出し単位チャンク化 -> data/chunks.jsonl 保存

対象データ形態:
  data/<maker>/<model>/.../published/<category>/*.md
  (例: data/honda/cr-v-2026/ivi/published/features/1-about-your-audio-system.md)

- maker/model はパスの先頭2階層
- category は published/ の直下のサブフォルダ名(無ければ空文字)
- figures/ フォルダの画像は対象外(*.mdではないので自然に除外される)

チャンク化:
  - Markdownの見出し(# 〜 ######)単位で区切る
  - 見出しパスは「ファイル名 > 見出し1 > 見出し2 ...」の形式
  - 1見出しセクションが長い場合は1000〜2000文字目安で追加分割し、
    「見出しパス > 部分N/M」というheading_pathにする
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# `python app/ingest.py` のように直接実行された場合、app パッケージが
# import できるようプロジェクトルートをパスに足しておく
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.facets import FacetsNotDeclaredError, load_axes, parse_facet_values  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "data"
OUT_PATH = PROJECT_ROOT / "data" / "chunks.jsonl"

TARGET_CHUNK = 1800
HARD_MAX = 2400
MAX_SLICES_PER_PARAGRAPH = 2

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
HTML_COMMENT_RE = re.compile(r"^\s*<!--.*-->\s*$")


def discover_files() -> list[Path]:
    files = []
    for p in DATA_ROOT.rglob("*.md"):
        rel_parts = p.relative_to(DATA_ROOT).parts
        if "published" not in rel_parts:
            continue
        files.append(p)
    return sorted(files)


def parse_category(rel_parts: tuple[str, ...]) -> str:
    pub_idx = rel_parts.index("published")
    # published/ の直下がサブカテゴリのフォルダなら category とする(ファイル名自体なら空)
    if pub_idx + 2 < len(rel_parts):
        return rel_parts[pub_idx + 1]
    return ""


def split_sections(text: str) -> list[tuple[list[str], list[str]]]:
    """見出し行で区切り、(見出しパスのリスト, 本文行リスト) のリストを返す。"""
    heading_stack: list[tuple[int, str]] = []
    sections: list[tuple[list[str], list[str]]] = []
    current_lines: list[str] = []

    def current_path() -> list[str]:
        return [t for _, t in heading_stack]

    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            sections.append((current_path(), current_lines))
            current_lines = []
            level = len(m.group(1))
            title = m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
        else:
            if HTML_COMMENT_RE.match(line):
                continue
            current_lines.append(line)
    sections.append((current_path(), current_lines))
    return sections


def split_into_parts(text: str) -> list[str]:
    """段落境界を優先しつつ、TARGET_CHUNK目安・HARD_MAX上限で分割する。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= TARGET_CHUNK:
        return [text]

    parts: list[str] = []
    buf = ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip("\n")
        if not para.strip():
            continue
        if len(para) > HARD_MAX:
            if buf:
                parts.append(buf.strip())
                buf = ""
            limit = HARD_MAX * MAX_SLICES_PER_PARAGRAPH
            truncated = para[:limit]
            for i in range(0, len(truncated), HARD_MAX):
                parts.append(truncated[i : i + HARD_MAX])
            if len(para) > limit:
                parts.append(f"…(異常に長い段落のため以降省略。全{len(para)}文字)")
            continue
        candidate = f"{buf}\n\n{para}" if buf else para
        if len(candidate) > TARGET_CHUNK and buf:
            parts.append(buf.strip())
            buf = para
        else:
            buf = candidate
    if buf.strip():
        parts.append(buf.strip())
    return parts


def build_chunks_for_file(
    path: Path, facet_values: dict[str, str | None], category: str, doc_path: str
) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    doc_name = path.name
    chunks = []
    for heading_titles, lines in split_sections(text):
        section_text = "\n".join(lines).strip()
        if not section_text:
            continue
        base_path = " > ".join([doc_name] + heading_titles) if heading_titles else doc_name
        parts = split_into_parts(section_text)
        total = len(parts)
        for i, part in enumerate(parts, start=1):
            heading_path = base_path if total == 1 else f"{base_path} > 部分{i}/{total}"
            chunks.append(
                {
                    **facet_values,
                    "category": category,
                    "doc_path": doc_path,
                    "doc_name": doc_name,
                    "heading_path": heading_path,
                    "clean_text": part,
                }
            )
    return chunks


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    files = discover_files()
    if not files:
        print(f"[ERROR] {DATA_ROOT} 配下に published/*.md が見つかりません")
        sys.exit(1)

    try:
        axes = load_axes()
    except FacetsNotDeclaredError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    all_chunks: list[dict] = []
    for f in files:
        rel = f.relative_to(DATA_ROOT)
        facet_values = parse_facet_values(rel.parts, axes)
        category = parse_category(rel.parts)
        all_chunks.extend(build_chunks_for_file(f, facet_values, category, rel.as_posix()))

    for idx, c in enumerate(all_chunks, start=1):
        c["chunk_id"] = f"chunk_{idx:05d}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"対象ファイル数: {len(files)}")
    print(f"チャンク数: {len(all_chunks)}")
    print(f"出力: {OUT_PATH}")
    print(f"宣言された絞り込み軸(data/facets.json): {axes}")
    print()

    seen_docs: set[str] = set()
    for axis in axes:
        chunk_counts = Counter(c[axis] for c in all_chunks)
        doc_counts: Counter = Counter()
        seen_docs.clear()
        for c in all_chunks:
            key = (c[axis], c["doc_path"])
            if key not in seen_docs:
                seen_docs.add(key)
                doc_counts[c[axis]] += 1
        print(f"--- 軸「{axis}」別 文書数 / チャンク数 ---")
        for value in sorted(doc_counts):
            print(f"  {value}: {doc_counts[value]}文書 / {chunk_counts[value]}チャンク")
        print()
    print("--- 先頭チャンクのプレビュー ---")
    for c in all_chunks[:3]:
        print(f"[{c['chunk_id']}] {c['maker']}/{c['model']}/{c['category']} heading_path={c['heading_path']!r}")
        print(f"  clean_text: {c['clean_text'][:120]!r}")
        print()

    long_chunks = [c for c in all_chunks if len(c["clean_text"]) > HARD_MAX]
    if long_chunks:
        print(f"[WARN] HARD_MAX({HARD_MAX})を超えるチャンクが{len(long_chunks)}件あります(想定内の安全分割の端数)")


if __name__ == "__main__":
    main()
