"""
検索前のクエリ拡張。app/expansion_rules.json の同義語グループに含まれる語が
クエリに現れたら、同じグループの他の言い換えをクエリ末尾に追記する。
辞書(JSON)を編集するだけで挙動を調整でき、コードの変更は不要。
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RULES_PATH = PROJECT_ROOT / "app" / "expansion_rules.json"


def load_groups() -> list[list[str]]:
    if not RULES_PATH.exists():
        return []
    with RULES_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("groups", [])


def expand_query(query: str, groups: list[list[str]] | None = None) -> tuple[str, list[str]]:
    """クエリに辞書の語が含まれていたら、同じグループの他の言い換えを追記する。

    戻り値: (拡張後のクエリ文字列, 追加された言い換え語のリスト)
    追加が無い場合は (元のクエリ, []) を返す。
    """
    if groups is None:
        groups = load_groups()

    query_lower = query.lower()
    added: list[str] = []
    for group in groups:
        matched = any(term.lower() in query_lower for term in group)
        if not matched:
            continue
        for term in group:
            if term.lower() not in query_lower and term not in added:
                added.append(term)

    if not added:
        return query, []
    return query + " " + " ".join(added), added
