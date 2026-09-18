"""
検索結果のスコア(コサイン類似度)から、グレード(A/B/C/D)や「該当あり/なし」を
判定するルールをまとめたもの。

チューニングするときは、この SEARCH_POLICIES に新しいバージョンを追加して
CURRENT_POLICY_VERSION を切り替えるだけでよい。main.py や search.py は
変更不要。過去のバージョンは消さずに残しておくことで、いつでも前の設定に戻せる。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchPolicy:
    version: str
    hit_threshold: float  # これ以上のスコアを「該当あり」とみなす
    grade_a: float
    grade_b: float
    grade_c: float

    def grade_for_score(self, score: float) -> str:
        if score >= self.grade_a:
            return "A"
        if score >= self.grade_b:
            return "B"
        if score >= self.grade_c:
            return "C"
        return "D"

    def is_hit(self, score: float) -> bool:
        return score >= self.hit_threshold


SEARCH_POLICIES: dict[str, SearchPolicy] = {
    "v1": SearchPolicy(version="v1", hit_threshold=0.30, grade_a=0.70, grade_b=0.50, grade_c=0.30),
}

CURRENT_POLICY_VERSION = "v1"


def get_current_policy() -> SearchPolicy:
    return SEARCH_POLICIES[CURRENT_POLICY_VERSION]
