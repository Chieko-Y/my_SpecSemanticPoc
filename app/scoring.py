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
    # キーワード一致(TF-IDF)検索のスコアは意味検索とは分布が違うため、別の閾値を持つ
    keyword_hit_threshold: float = 0.15
    keyword_grade_a: float = 0.50
    keyword_grade_b: float = 0.30
    keyword_grade_c: float = 0.15

    def grade_for_score(self, score: float, mode: str = "semantic") -> str:
        a, b, c = (
            (self.keyword_grade_a, self.keyword_grade_b, self.keyword_grade_c)
            if mode == "keyword"
            else (self.grade_a, self.grade_b, self.grade_c)
        )
        if score >= a:
            return "A"
        if score >= b:
            return "B"
        if score >= c:
            return "C"
        return "D"

    def is_hit(self, score: float, mode: str = "semantic") -> bool:
        threshold = self.keyword_hit_threshold if mode == "keyword" else self.hit_threshold
        return score >= threshold


SEARCH_POLICIES: dict[str, SearchPolicy] = {
    "v1": SearchPolicy(version="v1", hit_threshold=0.30, grade_a=0.70, grade_b=0.50, grade_c=0.30),
    # v2: キーワード検索フォールバック用の閾値を追加(意味検索側の値はv1と同じ)
    "v2": SearchPolicy(version="v2", hit_threshold=0.30, grade_a=0.70, grade_b=0.50, grade_c=0.30),
    # v3: 実測に基づき意味検索の閾値を引き上げ。関連する質問の最高スコアは0.55〜0.78、
    #     無関係な質問は最大0.37だったため、0.45を「該当あり」の境界にした
    "v3": SearchPolicy(version="v3", hit_threshold=0.45, grade_a=0.65, grade_b=0.55, grade_c=0.45),
}

CURRENT_POLICY_VERSION = "v3"


def get_current_policy() -> SearchPolicy:
    return SEARCH_POLICIES[CURRENT_POLICY_VERSION]
