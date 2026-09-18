"""
data/ 配下のフォルダ階層のうち、どの位置が「メーカー」「車種」のような
絞り込み軸(facet)かを、data/facets.json の宣言から読み込む。

宣言が無いデータに対して「1階層目はきっとメーカーだろう」のように
推測してしまうと、フォルダ構成が変わったときに誤ったラベルや
絞り込みボタンを表示してしまう恐れがある。そのため宣言必須にし、
無ければ ingest.py / main.py の両方がはっきりエラーで止まるようにする。
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FACETS_PATH = PROJECT_ROOT / "data" / "facets.json"


class FacetsNotDeclaredError(RuntimeError):
    pass


def load_axes() -> list[str]:
    if not FACETS_PATH.exists():
        raise FacetsNotDeclaredError(
            f"{FACETS_PATH} が見つかりません。data/ 配下のどのフォルダ階層が"
            '「メーカー」「車種」のような絞り込み軸かを、例えば'
            ' {"axes": ["maker", "model"]} のように明示的に宣言してください。'
        )
    with FACETS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    axes = data.get("axes", [])
    if not axes:
        raise FacetsNotDeclaredError(f"{FACETS_PATH} の axes が空です。最低1つの軸名を指定してください。")
    return axes


def parse_facet_values(rel_parts: tuple[str, ...], axes: list[str]) -> dict[str, str | None]:
    """パスの先頭セグメントを、宣言済みの軸名に割り当てる。
    宣言に含まれない軸(例: maker)は None にしておき、キー自体は必ず存在させる
    ことで、後続コードが未宣言データに対して誤った値を拾わないようにする。
    """
    if len(rel_parts) < len(axes):
        raise ValueError(f"パス {rel_parts} が axes({axes})の階層数より浅いです")
    values: dict[str, str | None] = {name: rel_parts[i] for i, name in enumerate(axes)}
    values.setdefault("maker", None)
    values.setdefault("model", None)
    return values
