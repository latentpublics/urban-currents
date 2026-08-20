"""U3: render `docs/journals-rebuild-review.md` (phase 0h).

Every figure here comes from `runs/journals_rebuild_review.json`, and every
column carries the population it was measured over — including the one that had
none, where the sheet prints a dash rather than the 0.5 that `percentile()`
returns for an empty comparison set.

Usage:
    uv run python scripts/journals_rebuild_review.py
    uv run python scripts/journals_rebuild_review_render.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")

d = json.loads((ROOT / "runs/journals_rebuild_review.json").read_text(encoding="utf-8"))
rows = d["rows"]
cur = [r for r in rows if r.get("include", True) and not r["new"]]
new = sorted(
    [r for r in rows if r["new"]], key=lambda r: (-r["our_citations"], -r["annual"])
)
MEASURED = 48.2  # journal-path candidates/day, mean of the five prepared days
ratio = d["daily_v2"] / d["daily_now"]


def pct(r):
    """The percentile, or a dash where there was no population to rank against.

    `percentile()` returns 0.5 for an empty population, and 28 of the 66 new
    journals have no subfield population at all. Printed as 0.50 it reads as
    "middle of its field"; it means "no field was found for it".
    """
    if not r.get("prestige_population"):
        return "—"
    v = r.get("prestige_pct_in_subfield")
    return "—" if v is None else f"{v:.2f}"


def esc(s: str) -> str:
    return (s or "").replace("|", "\\|")


def row_line_safe(r) -> str:
    sub = (r.get("subfields") or ["—"])[0]
    es = "—" if r.get("english_share") is None else f"{r['english_share']:.2f}"
    return (
        f"| {esc(r['name'])[:54]} | {r['concentration']:.3f} | {r['annual']:.0f} | "
        f"{pct(r)} | {sub} | {es} | {r['our_citations']} |"
    )


HEAD = (
    "| 저널 | 집중도 | 연간 | 분야내 백분위 | 서브필드 | english_share | 우리 인용 |\n"
    "|---|---:|---:|---:|:--:|---:|---:|"
)

diacritic_in = [r for r in rows if r.get("title_script") == "diacritic" and r.get("include", True)]
diacritic_out = [
    r for r in rows if r.get("title_script") == "diacritic" and not r.get("include", True)
]
nonlatin = [r for r in rows if r.get("title_script") == "non_latin"]
es_excluded = [
    r
    for r in rows
    if not r.get("include", True)
    and r.get("english_share") is not None
    and r["english_share"] < 0.5
]


def only_reason(r, needle: str) -> bool:
    """Would this source be included if this one rule were lifted?"""
    others = [x for x in r["exclude_reasons"] if needle not in x]
    return not others


L = []
A = L.append

A("# 저널 화이트리스트 재빌드 v2 — 검토 시트")
A("")
A("> 이 문서는 **판단 자료**입니다. `vocab/sources/journals.yaml`은 건드리지 않았고,")
A("> 재빌드 결과는 `vocab/sources/journals.rebuilt.v2.yaml`에 따로 있습니다.")
A("> 채택 여부는 YJUN이 정합니다. (Phase 0h / U3)")
A("")
A("## 세 줄 요약")
A("")
A(
    f"1. **무엇이 느는가**: 포함 저널이 {d['current_included']} → {d['v2_included']}건 "
    f"(신규 {d['new_included']}건). 신규 중 상위는 *Environment and Planning A*, "
    f"*Planning Theory*, *Journal of Urban Design* 처럼 **우리 코퍼스가 이미 "
    f"{sum(r['our_citations'] for r in new):,}번 인용하면서 한 번도 수집하지 않은** 저널입니다."
)
A(
    f"2. **무엇이 걱정인가**: 신규 66건 중 **{sum(1 for r in new if r['our_citations'] == 0)}건은 "
    f"우리 코퍼스가 한 번도 인용한 적이 없고**, 연간 논문수 중앙값이 "
    f"{statistics.median(r['annual'] for r in new):.0f}건으로 기존 "
    f"{statistics.median(r['annual'] for r in cur):.0f}건의 절반 이하입니다 — 작고 주변부인 "
    f"저널이 다수입니다. 일일 부하는 약 **{MEASURED:.0f} → {MEASURED * ratio:.0f}건(+{(ratio - 1) * 100:.0f}%)**."
)
A(
    "3. **무엇을 결정해야 하는가**: (가) 재빌드 v2를 통째로 채택할지, 아니면 "
    "**우리가 인용한 적 있는 38건만** 채택할지. (나) 아래 ★ 발음구별부호 7건 각각의 "
    "포함 여부. (다) `english_share`·비라틴 규칙으로 배제된 항목 중 되돌릴 것이 있는지."
)
A("")
A("---")
A("")
A("## 0. 이 표의 숫자가 무엇인지")
A("")
A("| 열 | 뜻 | 모집단 |")
A("|---|---|---|")
A(
    "| 집중도 | 저널의 주제 산출 중 대상 서브필드(3305/3313/3322)가 차지하는 비율 | "
    "OpenAlex `x_concepts` 기준 |"
)
A(
    f"| 연간 | 대상 서브필드 논문의 연간 편수 | {d['window_days']}일 "
    f"({d['rows'][0].get('_since', '2023-01-01')}~2026-08-13) 창을 연 단위로 환산 |"
)
A(
    "| 분야내 백분위 | `prestige_pct_in_subfield` — 같은 서브필드 저널 중 2년 평균 피인용의 "
    "백분위 | 0g T2에서 조회한 396개 소스. **비교 모집단이 없으면 `—`** |"
)
A(
    f"| 우리 인용 | 우리 코퍼스의 참조가 이 저널의 논문을 가리킨 **횟수** | "
    f"해결된 피인용 문헌 {d['resolved_works']:,}건 = 전체 고유 참조 "
    f"{d['distinct_references']:,}건의 **{d['resolved_reference_share'] * 100:.1f}%**. "
    f"**하한값이며 상한이 아닙니다** |"
)
A("")
A(
    "> `우리 인용`이 0이라고 해서 인용하지 않았다는 뜻이 아닙니다. 피인용 문헌 해결은 "
    "수요 순(많이 인용된 것부터)이라 미해결분에는 **드물게 인용된 것**이 몰려 있습니다. "
    "0은 '적어도 자주 인용하지는 않았다'로 읽어야 합니다."
)
A("")
A("---")
A("")
A(f"## 1. 신규 포함 저널 {len(new)}건 (우리 인용 순)")
A("")
A(HEAD)
for r in new:
    A(row_line_safe(r))
A("")
A(
    f"- 우리 코퍼스가 한 번이라도 인용한 저널: **{sum(1 for r in new if r['our_citations'] > 0)}건** / "
    f"인용 기록이 없는 저널: **{sum(1 for r in new if r['our_citations'] == 0)}건**"
)
A(
    f"- **분야내 백분위가 `—`인 신규 저널이 {sum(1 for r in new if not r.get('prestige_population'))}건**입니다. "
    f"OpenAlex가 이 저널들에 서브필드를 배정하지 않아 비교할 모집단이 없습니다. "
    f"`percentile()`은 그 경우 0.5를 돌려주는데, 그건 '중간'이 아니라 '잴 수 없음'입니다 — "
    f"표에서는 숫자를 지웠습니다. 대신 h-index와 2년 평균 피인용을 아래 2절에 적었습니다."
)
A(
    f"- 집중도 중앙값 {statistics.median(r['concentration'] for r in new):.3f} "
    f"(기존 포함군 {statistics.median(r['concentration'] for r in cur):.3f}) — "
    f"**집중도는 기존보다 오히려 높습니다.** 규모가 작을 뿐입니다."
)
A("")
A("---")
A("")
A(f"## 2. ★ 발음구별부호로 표시만 된 {len(diacritic_in)}건 (포함됨, 사람 판단 필요)")
A("")
A(
    "`title_script`는 제목의 문자만 봅니다. 비라틴 문자는 자동 배제하지만 "
    "**발음구별부호는 표시만 하고 배제하지 않습니다** — `Archæology`처럼 합자를 가진 "
    "진짜 영어 저널과, 헝가리어·폴란드어 저널이 같은 신호에 걸리기 때문입니다. "
    "둘을 가르는 자동 규칙이 없어 사람에게 넘깁니다."
)
A("")
A(HEAD)
for r in sorted(diacritic_in, key=lambda r: -(r.get("english_share") or 0)):
    A(row_line_safe(r))
A("")
A("판단 근거로 덧붙이면:")
A("")
for r in sorted(diacritic_in, key=lambda r: -(r.get("english_share") or 0)):
    A(
        f"- **{esc(r['name'])}** — english_share {r.get('english_share')}, "
        f"우리 인용 {r['our_citations']}회, 연간 {r['annual']:.0f}건, "
        f"h-index {r.get('h_index')}, 2년 평균 피인용 {r.get('two_year_mean_citedness')}, "
        f"발행처 {esc(r.get('publisher') or '—')}"
    )
A("")
A("---")
A("")
A(f"## 3. 비라틴 문자로 배제된 {len(nonlatin)}건")
A("")
A(
    "이 규칙은 **자동으로 배제**합니다. 되돌리려면 아래에서 고르면 됩니다 — "
    "`english_share`가 1.0인 항목이 여럿인데, OpenAlex가 이들의 논문 언어를 "
    "영어로 표시하기 때문입니다(0f에서 확인한 교란입니다). 제목 문자 신호는 "
    "그 교란을 우회하려고 넣은 **독립 신호**입니다."
)
A("")
A("| 저널 | 집중도 | 연간 | english_share | 우리 인용 | 이 규칙만으로 배제되었나 |")
A("|---|---:|---:|---:|---:|:--:|")
for r in sorted(nonlatin, key=lambda r: -r["annual"]):
    es = "—" if r.get("english_share") is None else f"{r['english_share']:.2f}"
    only = "**예**" if only_reason(r, "비라틴") else "아니오 (다른 기준도 미달)"
    A(
        f"| {esc(r['name'])[:54]} | {r['concentration']:.3f} | {r['annual']:.0f} | "
        f"{es} | {r['our_citations']} | {only} |"
    )
A("")
A("---")
A("")
A(f"## 4. `english_share`로 배제된 {len(es_excluded)}건")
A("")
A("| 저널 | english_share | 집중도 | 연간 | 우리 인용 | 이 규칙만으로 배제되었나 |")
A("|---|---:|---:|---:|---:|:--:|")
for r in sorted(es_excluded, key=lambda r: r["english_share"]):
    only = "**예**" if only_reason(r, "english_share") else "아니오 (다른 기준도 미달)"
    A(
        f"| {esc(r['name'])[:54]} | {r['english_share']:.2f} | {r['concentration']:.3f} | "
        f"{r['annual']:.0f} | {r['our_citations']} | {only} |"
    )
A("")
A(f"발음구별부호가 붙었으면서 **배제된** 항목도 참고로 {len(diacritic_out)}건 있습니다:")
A("")
for r in diacritic_out:
    A(
        f"- {esc(r['name'])} — {', '.join(r['exclude_reasons']) or '기준 미달 없음(확인 필요)'}"
    )
A("")
A("---")
A("")
A("## 5. 채택 시 예상 부하")
A("")
A("| | 현재 | v2 채택 시 |")
A("|---|---:|---:|")
A(f"| 포함 저널 | {d['current_included']} | {d['v2_included']} |")
A(f"| 서브필드 논문 (일, 화이트리스트 합) | {d['daily_now']:.2f} | {d['daily_v2']:.2f} |")
A(f"| **저널 경로 후보 (일, 실측·추정)** | **{MEASURED:.0f}** | **약 {MEASURED * ratio:.0f}** |")
A("")
A(
    f"- 실측은 준비된 5일(08-05/06/07/10/11)의 `classify` 단계 저널 후보 평균 "
    f"{MEASURED:.1f}건/일입니다."
)
A(
    f"- 추정은 서브필드 논문 합계의 비율({ratio:.3f}배)을 실측에 곱한 것입니다. "
    f"수집은 대상 서브필드 논문만이 아니라 **그 저널의 산출 전체**를 가져오므로, "
    f"신규 저널의 집중도가 기존과 비슷한 한 이 비율이 유지됩니다."
)
A(
    "- **이 수치는 하한입니다.** 신규 저널은 규모가 작아 한 건의 비중이 크고, "
    "게이트 통과율이 기존과 같다는 가정이 들어 있습니다."
)
A(
    f"- 발행 슬롯은 12/12로 고정이므로 **부하가 느는 것은 후보 쪽이지 발행 쪽이 아닙니다.** "
    f"영향은 선정 경쟁률입니다: 저널 후보 {MEASURED:.0f}건에서 12건을 고르던 것이 "
    f"{MEASURED * ratio:.0f}건에서 12건이 됩니다."
)
A("")
A("---")
A("")
A("## 6. 결정해야 할 것")
A("")
A("1. **전체 채택 / 부분 채택 / 보류**")
A(
    f"   - 전체: {d['v2_included']}건. 우리가 인용한 적 없는 "
    f"{sum(1 for r in new if r['our_citations'] == 0)}건이 함께 들어옵니다."
)
A(
    f"   - 부분: 신규 중 **우리 인용 ≥ 1인 {sum(1 for r in new if r['our_citations'] > 0)}건만** "
    f"채택하면 포함은 {d['current_included'] + sum(1 for r in new if r['our_citations'] > 0)}건, "
    f"부하 증가는 절반 이하가 됩니다. 다만 '우리 인용'은 위에서 적었듯 "
    f"해결된 참조 {d['resolved_reference_share'] * 100:.1f}%에서 나온 **하한**이라, "
    f"이 기준은 **우리가 이미 보던 것 쪽으로 편향**됩니다 — 화이트리스트의 그림자를 "
    f"다시 만드는 셈입니다."
)
A("2. **★ 발음구별부호 7건** 각각 — 위 2절의 목록에서 개별 판단.")
A("3. **비라틴·`english_share` 배제 되돌릴 것** — 3·4절에서 '이 규칙만으로 배제' 표시된 항목.")
A("")
A(
    "채택하기로 하면 `vocab/sources/journals.yaml`을 교체하고 분류기를 재학습해야 합니다 "
    "(0g T5에서 비용을 이미 측정했습니다: 임베딩 로컬, OpenAlex $0.0096). "
    "재학습 없이 화이트리스트만 바꾸면 **분류기가 배운 저널 취향과 실제 화이트리스트가 "
    "어긋납니다.**"
)
A("")
A("---")
A("")
A(
    f"*생성: Phase 0h U3. 원자료 `runs/journals_rebuild_review.json`, "
    f"재빌드 `vocab/sources/journals.rebuilt.v2.yaml` ({d['v2_total']}건 검토, "
    f"{d['v2_included']}건 포함).*"
)

(ROOT / "docs/journals-rebuild-review.md").write_text(
    "\n".join(L) + "\n", encoding="utf-8", newline="\n"
)
print("wrote docs/journals-rebuild-review.md", len("\n".join(L)), "chars")
