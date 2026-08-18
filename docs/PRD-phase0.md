# Urban Currents — Phase 0 PRD (v1.1)

| | |
|---|---|
| 문서 | Product Requirements Document — Phase 0 (PoC) |
| 날짜 | 2026-08-12 (v1.1 개정) |
| 상태 | 확정 (개발 착수용) |
| 상위 문서 | `2026-08-12_urban-currents-service-plan.md` (서비스 정의 v1.0) |
| 저장소 | `github.com/youngjour/urban-currents` (private) |
| 기간 | 2주 |

> **v1.1 변경 요약** — Places 축 우선순위 하향 / OpenAlex를 Papers 트랙의 스파인으로
> 채택 / 관련도 필터를 seed centroid에서 저널 기반 분류기로 교체 / 요약을 What+Why로
> 축소 / **OpenAlex API가 2026-02부터 키·예산제로 전환된 사실 반영**.
> 개정 근거는 §14.

> 스키마·필드명·설정키는 영어, 서술은 한국어. 저장소에 커밋되는 산출물
> (README, OPERATIONS.md, 코드 주석, 발행 콘텐츠)은 전부 영어다.

---

## 1. Phase 0가 답해야 하는 질문

Phase 0는 제품이 아니라 **네 개 질문에 대한 측정 장치**다.

| # | 질문 | 측정 방법 | 판단 기준 |
|---|---|---|---|
| Q1 | 필터가 쓸 만한가 | 분류기 홀드아웃 AUC + 실제 후보 상위 30건 × 5일을 YJUN이 keep/drop 라벨링 | AUC ≥ 0.9, precision@10 ≥ 0.7 |
| Q2 | 분야에 데일리를 지탱할 신호량이 있는가 | 90일 백필의 일별 통과 아이템 수 분포 | 중앙값 ≥ 5건/일 |
| Q3 | quiet day 임계값을 어디에 둘 것인가 | 90일 백필 헤드라인 점수 분포의 분위수 | 헤드라인 발생률 30–50%로 캘리브레이션 |
| Q4 | **사람 없이 돌 수 있는가** | 보류 큐 · 사후 표본 검수 · 발행 후 철회 건수 | **7일 이상 무인 운영에서 편집자가 철회할 항목 0건** |

> **[Q4 재정의 — Phase 0L, M2]** 원래 Q4는 *"하루치 검수 중앙값 ≤ 15분"*이었습니다.
> **그 정의로 측정된 값은 0일입니다** — `review_s`를 가진 실행이 여덟 배치 동안 한 건도
> 없었습니다. 정의를 바꾼 이유는 답이 나오지 않아서가 아니라 **운영 조건이 바뀌었기**
> 때문입니다: YJUN이 3일~일주일씩 자리를 비우는 것이 정상이고, 매일 15분을 요구하는
> 설계는 그 조건에서 작동하지 않습니다.
>
> **정의를 바꿨다고 답이 생긴 것은 아닙니다.** 옛 정의로는 미측정이고, 새 정의로는
> 아직 무인 운영 일수가 0일입니다. 둘 다 그렇게 기록합니다.
>
> 따라오는 변경: **발행 승인이 없어집니다.** v1.0 §8의 오래된 긴장(사람의 발행 승인 ↔
> ILP "예외 없음")은 이것으로 해소됩니다 — 승인 없이 자동 발행하고, 품질은 사람이 매일
> 막는 대신 **선정 정책이 막습니다**(보류 큐, `pipeline/held.py`).

부가 측정: 아이템당 LLM·임베딩 비용, OpenAlex 일일 예산 소진량, 요약 품질
(useful/vacuous/wrong 3분류, 50건 — 기준 미달이어도 Go/No-Go를 막지는 않는다).

**Phase 0의 산출물은 코드 + `docs/phase0-report.md` (위 숫자와 근거 데이터)이다.**

---

## 2. 범위

### 하는 것
- **Papers 트랙만** — arXiv + OpenAlex
- 90일 백필 + 일일 증분 수집
- **저널 기반 관련도 분류기** (§5.4)
- preprint↔journal 중복 병합
- 엔티티: OpenAlex 네이티브(Topics/Authors/Institutions/citations) 그대로 + 자체 오버레이(Methods/Data/Tools)
- 2층 요약 (What / Why it matters) — 영어만
- 헤드라인 점수 + quiet day 판정
- 콘텐츠 JSON 발행 (`/content`)
- 로컬 단일 HTML 프리뷰
- 계측 로그

### 안 하는 것 (Phase 1+)
Conferences·Code·Data 트랙 / 한국어 번역 / Astro 사이트 / 배포 / 이메일 / RSS /
giscus / Pagefind / GitHub Actions cron / 봇 PR / Deep Dive / Ask Currents / Monthly Map 렌더링

### 우선순위 하향 — Places 축
v1.0은 Places를 서비스의 시그니처로 잡았으나 **Phase 0에서는 시그니처가 아니다.**

- 커버리지 리포트, "연구되지 않는 도시들" 서사, Places 기반 마케팅 문안: **전부 보류**
- 다만 **수집은 계속한다.** LLM 엔티티 추출 패스에서 Places 후보는 이미 부산물로
  나오므로 한계비용이 0에 가깝다. 추출해서 저장하되 confidence가 낮으면 그냥 비워둔다.
- Wikidata 조회는 **best-effort**로 강등한다. 실패해도 로그만 남기고 진행한다.
- `places_status` 3-state(`resolved`/`unspecified`/`not_applicable`)는 스키마에 남긴다.
  나중에 이 축을 살릴 때 아카이브를 소급 재처리하지 않기 위해서다 — 지금 필드를
  비워두는 비용은 0이고, 나중에 추가하는 비용은 전체 재처리다.
- Q1–Q4 어디에도 Places 지표는 없다. 수용 기준에서도 뺀다.

> 그래프 스키마는 Phase 0 첫날부터 콘텐츠에 내장한다. 스키마는 이 프로젝트에서
> 가장 비싼 락인 지점이다.

---

## 3. 데이터 모델

### 3.1 세 종류의 객체

| 객체 | 정체 | 수명 | 위치 |
|---|---|---|---|
| **Item** | 하나의 artifact (논문 1편) | 영구, **가변** | `content/items/{work_key}.json` |
| **Issue** | 하루치 에디션 | 발행 후 **불변** | `content/issues/YYYY-MM-DD.json` |
| **Entity** | 태그 노드 | 영구, 가변 | `content/entities/{facet}/{id}.json` |

**왜 분리하는가.** preprint가 4개월 뒤 저널에 실릴 때 같은 논문이 두 번 헤드라인이
되어서는 안 되지만 `[preprint]` 배지는 갱신되어야 한다. Item이 가변·영구이고
Issue가 불변·일일이면 이게 자연스럽게 처리된다.

### 3.2 Item 스키마

**OpenAlex의 Work 객체 관례를 따른다** — 특히 `ids` 블록(식별자 다중화)과
엔티티 참조를 ID 문자열로 두는 방식. 우리 스키마는 OpenAlex Work의
**서브셋 + 오버레이**이지 별개 설계가 아니다 (§4.1).

```jsonc
{
  "schema_version": "0.2.0",
  "work_key": "arxiv:2608.01234",     // 안정 식별자. 한번 정해지면 바뀌지 않는다.
  "track": "papers",
  "first_published": "2026-08-14",
  "updated": "2026-08-14",

  "ids": {                            // OpenAlex 관례. 새 식별자는 여기에 추가된다.
    "openalex": "W4392…",             // 수집 시점에 없을 수 있음 → 나중에 채워짐
    "doi": "10.48550/arXiv.2608.01234",
    "arxiv": "2608.01234",
    "pmid": null
  },

  "bibliography": {                   // 전부 수집기가 메타데이터에서 기입. LLM 접근 금지.
    "title": "…",
    "authors": [
      {"name": "…", "orcid": "…", "openalex": "A5023…",
       "institutions": [{"ror": "https://ror.org/…", "name": "…"}]}
    ],
    "publication_date": "2026-08-11",
    "primary_location": {"source_id": "S4306400194", "source_name": "arXiv",
                         "type": "repository", "version": "submittedVersion",
                         "landing_page_url": "https://arxiv.org/abs/2608.01234",
                         "pdf_url": "https://arxiv.org/pdf/2608.01234"},
    "abstract": "…"                   // 요약의 유일한 근거. 원문 보존.
  },

  "publication_status": {
    "state": "preprint",              // preprint | published
    "journal": null, "source_id": null, "doi": null, "detected_at": null
  },

  "graph": {                          // OpenAlex에서 그대로 받는다. 인용 그래프는 공짜다.
    "referenced_works": ["W2145…", "W3011…"],
    "related_works": ["W4123…"],
    "cited_by_count": 0
  },

  "summary": {
    "en": {
      "what": "…",                    // 2–3문장. 측정값·도시명·해상도를 그대로 노출.
      "why": "…"                      // 1–2문장. 분야 맥락에서의 의미.
    }
    // caveats는 선택 필드 (§5.5). "ko"는 Phase 1에서 필드 추가.
  },

  "signals": {                        // 구 caveat_flags. 정형 판정 — 배지·필터·그래프 속성.
    "geographic_scope": {"value": "single_city", "confidence": "high", "basis": "llm"},
    "sample_size_reported": {"value": true, "detail": "3.4M images", "confidence": "high", "basis": "rule"},
    "temporal_coverage_reported": {"value": false, "confidence": "high", "basis": "rule"},
    "code_available": {"value": true, "url": "https://github.com/…", "confidence": "high", "basis": "rule"},
    "data_available": {"value": false, "confidence": "medium", "basis": "llm"},
    "is_retracted": {"value": false, "confidence": "high", "basis": "rule"}
  },

  "badges": ["code", "preprint"],     // code | data | preprint | published

  "entities": {
    // — OpenAlex 네이티브. 그대로 받는다. LLM 개입 없음.
    "topics": [{"id": "openalex:T10746", "label": "Urban Transport Systems",
                "subfield": "3322", "score": 0.82, "is_primary": true}],
    "people": [{"id": "orcid:0000-0002-…", "label": "…"}],
    "orgs":   [{"id": "ror:https://ror.org/…", "label": "…"}],
    // — 우리 오버레이. 여기가 부가가치다.
    "methods": [{"id": "method:gnn", "label": "graph neural network", "confidence": 0.91}],
    "data":    [{"id": "data:street-view", "label": "street view imagery", "confidence": 0.97}],
    "tools":   [{"id": "github:gboeing/osmnx", "label": "OSMnx", "confidence": 0.88}],
    // — 우선순위 하향. best-effort. 비어 있어도 정상.
    "places": [{"id": "wikidata:Q8684", "label": "Seoul", "role": "study_area", "confidence": 0.94}],
    "places_status": "resolved"       // resolved | unspecified | not_applicable | not_attempted
  },

  "lens": "behavior",                 // behavior | system | null — 에디토리얼 속성

  "scores": {
    "relevance": 0.87,                // 분류기 예측 확률 (§5.4). 0–1, 캘리브레이션됨.
    "headline": 0.63,
    "components": {"relevance": 0.87, "source_multiplicity": 0.0,
                   "artifact_completeness": 0.5, "novelty": 0.4}
  },

  "cluster": {
    "cluster_id": "clu_2026-08-14_003",
    "members": ["arxiv:2608.01234", "openalex:W4392…"],
    "merge_basis": "doi_match"        // doi_match | arxiv_location | title_author_fuzzy
  },

  "provenance": {
    "collected_at": "2026-08-14T21:00:00Z",
    "collectors": ["arxiv", "openalex"],
    "pipeline_version": "0.2.0",
    "llm": {"model": "…", "prompt_version": "summarize/papers@0.2.0"},
    "classifier_version": "clf-2026-08-13",
    "cost_usd": 0.0031, "tokens": {"in": 1420, "out": 300}
  },

  "review": {
    "status": "approved",             // pending | approved | rejected | edited
    "reviewer_notes": null,
    "edits": []                       // YJUN이 손댄 필드 경로 → 프롬프트 개선의 재료
  }
}
```

`review.edits`가 조용한 핵심이다. YJUN이 어느 필드를 얼마나 자주 고치는지가
Phase 1에서 프롬프트를 어디부터 손볼지 알려준다. 검수를 데이터로 바꾸는 장치.

### 3.3 Issue 스키마

```jsonc
{
  "schema_version": "0.2.0",
  "date": "2026-08-14",
  "headline": {"present": true, "work_key": "arxiv:2608.01234",
               "line": "A 12-city street-view model puts a number on…"},
  "quiet_day": false,
  "scan_meta": {"arxiv_categories": 7, "journals": 42, "candidates_scanned": 912,
                "candidates_after_gate": 118, "items_published": 6,
                "minutes_saved_estimate": 42},
  "items": ["arxiv:2608.01234"],
  "status_changes": [{"work_key": "arxiv:2604.09876", "from": "preprint",
                      "to": "published", "journal": "Cities"}],
  "run_id": "run_2026-08-14T21-00-00Z"
}
```

### 3.4 Entity 스키마

```jsonc
{
  "id": "method:gnn",
  "facet": "methods",
  "label": "graph neural network",
  "aliases": ["GNN", "graph neural networks", "graph convolutional network"],
  "parent": "method:deep-learning",
  "canonical": {"openalex": null, "wikidata": "Q30297837"},
  "item_count": 14,
  "first_seen": "2026-08-14",
  "last_seen": "2026-09-02"
}
```

### 3.5 Edge — 파생 생성

엣지는 별도 소스가 아니라 Item에서 **파생**된다. `pipeline/graph/build.py`가
전체 Item을 읽어 `content/graph/edges.jsonl`을 생성한다 (빌드 산출물, 손편집 금지).

```
{"src":"arxiv:2608.01234","dst":"method:gnn",       "type":"uses_method",  "date":"2026-08-14"}
{"src":"arxiv:2608.01234","dst":"openalex:T10746",  "type":"has_topic",    "date":"2026-08-14"}
{"src":"arxiv:2608.01234","dst":"openalex:W2145…",  "type":"cites",        "date":"2026-08-14"}
```

`cites` 엣지가 OpenAlex `referenced_works`에서 공짜로 나온다. Phase 2의
"related papers"와 Monthly Map은 이 위의 쿼리다 — 새 파이프라인이 아니다.

### 3.6 통제 어휘 (`/vocab`)

| 파일 | 내용 | Phase 0 초기화 |
|---|---|---|
| `methods.yaml` | family → method 2계층 | 학습셋 초록에서 후보 추출 → YJUN 큐레이션 |
| `data.yaml` | 데이터 종류 | 동일 |
| `tools.yaml` | GitHub repo 화이트리스트 | v1.0 §12의 30–50 repo |
| `sources/arxiv.yaml` | 카테고리 + 게이트 키워드 | 손으로 작성 |
| `sources/journals.yaml` | OpenAlex source ID 화이트리스트 | **자동 생성 후 YJUN 1회 검토** (§5.4) |
| `places_aliases.yaml` | 지오코딩 예외 매핑 | 하향 — 비워둔 채 시작 |

---

## 4. OpenAlex 채택 결정

### 4.1 왜 OpenAlex 구조를 쓰는가 — 저작권 문제는 없다

**OpenAlex 데이터는 CC0(퍼블릭 도메인)다.** 재사용·재배포·파생 DB 구축 전부 허용되고
출처 표기도 의무가 아니다(권장일 뿐). MAG 포맷 스냅샷만 ODC-BY로 표기 의무가 있는데,
우리는 그걸 쓰지 않는다. **구조를 차용하지 않을 이유는 라이선스가 아니었다.**

그래서 v1.1에서 다음을 OpenAlex에서 **그대로 받는다**:

| 받는 것 | 필드 | 우리가 만들었다면 |
|---|---|---|
| 주제 분류 | `topics`, `primary_topic` (domain→field→subfield→topic 4계층, ~4,500 topic) | 수개월 + 지속 유지 |
| 저자 정규화 | `authorships[].author.orcid`, `.openalex` | 동명이인 처리 지옥 |
| 기관 정규화 | `authorships[].institutions[].ror` | 동일 |
| **인용 그래프** | `referenced_works`, `related_works`, `cited_by_count` | 불가능에 가까움 |
| 저널 메타 | Source 객체 (`issn_l`, `host_organization`, `topics`, `type`) | 불가능 |
| 오픈액세스 상태 | `best_oa_location`, `locations[].version` | 불가능 |

특히 **`referenced_works`가 v1.0 설계에서 빠져 있던 큰 자산이다.** 우리 그래프는
item→tag 엣지만 있었는데, OpenAlex는 item→item 엣지를 준다. alphaXiv의
"related papers"가 정확히 이 위에 서 있다. 이걸 안 쓰는 건 순전한 손실이었다.

### 4.2 그렇다면 왜 OpenAlex를 통째로 쓰지 않는가

세 가지 이유이고, 전부 **커버리지**의 문제지 라이선스나 품질의 문제가 아니다.

1. **트랙 2–4가 OpenAlex 모델에 없다.** GitHub 릴리스, 데이터셋, 학회 CFP는
   Work/Author/Institution/Source/Topic 어디에도 해당하지 않는다. Papers만 OpenAlex
   모양이고 나머지가 다른 모양이면 그게 더 나쁘다 — 그래서 **Item이 상위 추상이고
   OpenAlex Work는 Papers 트랙에서의 구현체**다.
2. **오버레이 4축이 OpenAlex에 없다.** Methods / Data / Tools / Places — "무슨 방법으로,
   무슨 데이터로, 무슨 도구로, 어디를" 은 어떤 논문 DB도 체계적으로 붙이지 않는다.
   여기가 우리 부가가치이고, OpenAlex 스키마를 그대로 쓰면 이걸 넣을 자리가 없다.
3. **arXiv preprint 처리가 불완전하다** (§5.2). OpenAlex는 preprint를 별도 Work로
   두기도 하고 published Work의 location으로 합치기도 한다 — 둘 다 관측된다.
   우리 dedup 레이어는 여전히 필요하다.

**결론: OpenAlex를 Papers 트랙의 스파인으로 삼고, 그 위에 오버레이를 얹는다.**
스키마 설계 원칙 — OpenAlex가 주는 필드는 **이름을 바꾸지 않고 그대로 쓴다**
(`referenced_works`를 `citations`로 개명하지 않는다). 우리 것만 새 이름을 갖는다.

### 4.3 API 접근 조건 — 2026년 2월에 바뀌었다

**v1.0/PRD v1.0의 "polite pool + `mailto=`" 지시는 이제 유효하지 않다.**

- **2026-02-13부터 모든 요청에 API 키가 필요하다.** `openalex.org/settings/api`에서
  무료 발급. polite pool은 폐지됐다.
- 과금이 요청 수가 아니라 **USD 일일 예산**이다: 키 없음 $0.10/일, 무료 키 $1/일.
  단가 — singleton 무료, list+filter $0.10/1,000, full-text search $1/1,000,
  semantic search $1/1,000, PDF 다운로드 $10/1,000.
- 하드 캡 100 req/s. 초과 시 429. 응답 헤더에 `X-RateLimit-Remaining`,
  `X-RateLimit-Credits-Used`, `X-RateLimit-Reset`. 응답 `meta`에 `cost_usd`.

**우리 예산 계산**: 무료 키 $1/일 = list 호출 10,000회/일. 페이지당 200건이므로
이론상 200만 레코드/일. 일일 증분(저널 42개 × 커서 페이징)은 하루 수십 회,
90일 백필과 학습셋 구축(1만 건)도 수백 회 수준. **무료 키로 충분하다.**
다만 `meta.cost_usd`를 metrics에 누적 기록해 실제 소진량을 확인한다.

> ⚠️ 이 항목은 문서가 빠르게 바뀌고 있다. 착수 시 `developers.openalex.org`의
> authentication 페이지를 다시 확인할 것. GitHub의 `ourresearch/openalex-docs`는
> **오래된 내용**(키 불필요, 폴라이트 풀)을 담고 있으니 신뢰하지 말 것.

### 4.4 alphaXiv에서 차용하는 것

새로 만들지 않는다. 구조를 빌린다.

| 차용 | 내용 | 어디에 |
|---|---|---|
| **ID 기반 라우팅** | 논문 페이지 URL이 canonical 논문 ID를 그대로 씀 (`/abs/2608.01234` 대응) | Phase 1 사이트 라우팅. Phase 0에서는 `work_key`를 URL-safe하게 잡아두는 것으로 대비 |
| **related papers** | 인용·공통인용 기반 이웃 추천 | `graph.related_works` + `referenced_works` (Phase 0에서 수집만, 렌더는 Phase 2) |
| **trending 랭킹** | 시간감쇠 인기 점수 | `scores.headline`의 `novelty` 성분 설계 참고 |
| **paper blog** | 논문당 1회 고정비의 정적 심화 해설 | Phase 2 Deep Dive의 선례 |

**Ask Currents(Phase 3)의 참고 구현**: `AsyncFuncAI/alphaxiv-open` (MIT) — FastAPI +
Markitdown(PDF→MD) + MiniRAG/LightRAG + LLM. Phase 0와는 무관하지만 Phase 3에서
스크래치로 만들 필요가 없다는 근거다.
https://github.com/AsyncFuncAI/alphaxiv-open

---

## 5. 파이프라인

```
collect → dedup/merge → gate → classify → select → link → summarize → headline → issue → preview
```

각 단계는 독립 실행 가능해야 한다 (`uc <stage> --date …`). 요약을 다시 돌리려고
수집부터 다시 하면 2주가 순식간에 없어진다.

### 5.1 수집

**arXiv** — `cs.CY, cs.SI, cs.LG, cs.CV, cs.AI, stat.AP, physics.soc-ph`
- 요청 간격 3초 이상, `User-Agent`에 연락처, 백오프 3회
- 백필: `submittedDate:[YYYYMMDD TO YYYYMMDD]` 범위 쿼리
- OpenAlex Work가 아직 없을 수 있으므로 arXiv 메타데이터만으로 Item이 성립해야 한다

**OpenAlex** — `pyalex` (v0.21+, 커서 페이징·재시도 내장) 사용. 직접 HTTP를 짜지 않는다.
- `pyalex.config.api_key`에 `.env`의 **`OPENALEX_KEY`** (변수명 주의 — `OPENALEX_API_KEY` 아님)
- 저널 증분: `filter=primary_location.source.id:S…|S…, from_publication_date:…`
- **OpenAlex 보강 패스**: 이미 수집된 arXiv Item에 대해 `doi:10.48550/arXiv.{id}` 또는
  제목 검색으로 Work를 찾아 `ids.openalex`, `graph.*`, `entities.topics/people/orgs`를
  채운다. 찾지 못하면 그냥 비워둔다 (arXiv 등재가 늦는 경우가 있다). 며칠 뒤
  재시도하도록 `openalex_enrich_pending` 큐에 남긴다.

**원본 응답 보존은 협상 대상이 아니다.** `runs/{run_id}/raw/`에 그대로 저장한다.
파서를 고칠 때마다 API를 다시 때리면 예산이 소진되고, 무엇보다 어제와 같은 입력으로
재현할 수 없게 된다.

### 5.2 중복 병합

같은 논문이 arXiv preprint와 OpenAlex 저널판으로 두 번 들어온다. OpenAlex 자체도
일관되지 않다 — preprint를 별도 Work(`type: "preprint"`, DOI `10.48550/arxiv.*`)로
두기도 하고, published Work의 `locations[]`에 합치기도 한다. **둘 다 처리해야 한다.**

병합 키 우선순위:

1. **DOI 일치** — `10.48550/arxiv.*` 형태를 정규화해 arXiv ID로 환원한 뒤 비교
2. **arXiv location 매칭** — Work의 `locations[]` 중 `source.id == "S4306400194"`
   (arXiv)인 항목의 `landing_page_url`에서 arXiv ID 추출
3. **제목 정규화 + 제1저자 성** — 소문자·구두점 제거·공백 정규화 후
   `token_sort_ratio ≥ 95` AND 제1저자 성 일치

**병합 결과가 이미 발행된 Item과 일치하면 신규 발행이 아니다.** 기존 Item의
`publication_status`를 갱신하고 Issue의 `status_changes`에 한 줄 남긴다.
이 분기를 빼먹으면 4개월 뒤부터 같은 논문이 두 번 헤드라인이 된다.

`work_key`는 **한 번 정해지면 절대 바뀌지 않는다.** OpenAlex ID를 나중에 알게 되어도
`ids.openalex`에 추가할 뿐 `work_key`는 그대로다. 우선순위: arXiv ID → DOI → OpenAlex ID.

### 5.3 게이트 (volume gate)

arXiv 7개 카테고리 일일 유입은 대략 900–1,100건이고, cs.LG/cs.CV/cs.AI가 90%를
차지하면서 수율은 가장 낮다.

- **cs.CY, cs.SI, stat.AP, physics.soc-ph** (소량·고수율): 게이트 없이 전부 분류기로
- **cs.LG, cs.CV, cs.AI** (대량·저수율): 관대한 OR 키워드 게이트 통과분만
  (urban, city/cities, metropolitan, neighborhood, mobility, transport, traffic,
  land use, street, POI, built environment, housing, spatial, geospatial, GIS,
  satellite, remote sensing, census, pedestrian, transit, commut*, land cover …
  — `vocab/sources/arxiv.yaml`)

**게이트의 재현율은 반드시 측정한다.** Phase 0 중 1회, 탈락 집합에서 무작위 200건을
뽑아 분류기를 돌린다. 임계값 초과가 3건을 넘으면 게이트가 너무 좁은 것이므로
키워드를 넓힌다. 측정 없는 게이트는 그냥 조용한 손실이다.

### 5.4 관련도 분류기 — v1.0의 seed centroid를 대체한다

**핵심 발상 (YJUN):** "무엇이 도시 연구인가"를 손으로 정의하지 말고, **분야가 이미
합의한 정의 — SSCI/Scopus의 Urban Studies 카테고리 저널이 싣는 것 — 을 학습셋으로
쓴다."** 이게 seed 논문 30편을 손으로 고르는 것보다 정확하고, 무엇보다 재현 가능하다.

**행운의 일치:** Scopus ASJC 서브필드 코드 **3322 = Urban Studies**이고,
Scimago 카테고리 번호와 **OpenAlex 서브필드 ID가 같은 체계**를 쓴다. 즉 YJUN이
말한 "SCI-index urban studies edition 저널들"은 OpenAlex에서 한 줄로 뽑힌다.

**저널 화이트리스트 자동 생성** (`scripts/build_journal_whitelist.py`):

```
GET /sources?filter=topics.subfield.id:3322,type:journal&sort=works_count:desc&per_page=200
```

서브필드 후보 — YJUN이 최종 선택:

| ASJC | 이름 | 채택 |
|---|---|---|
| 3322 | Urban Studies | **코어** |
| 3305 | Geography, Planning and Development | **코어** |
| 3313 | Transportation | **코어** |
| 2215 | Building and Construction / 관련 | 검토 |
| 3312 | Sociology and Political Science | 부분 (너무 넓음) |
| 1904 | Earth-Surface Processes / GIScience 계열 | 검토 |

생성 결과는 `vocab/sources/journals.yaml`에 **ID·이름·연간 논문수·서브필드**와 함께
기록하고, **YJUN이 1회 yes/no 통과**한다 (약 100개 저널명 훑기, 20분).
v1.0 §12의 손으로 적은 목록은 이 자동 목록과 **대조용**으로 쓰고, 자동 목록에 없는
항목(예: npj Urban Sustainability, Urban Informatics — 신생 저널)은 수동 추가한다.

**학습셋**

- **Positive (~4,000)**: 화이트리스트 저널의 2024–2026 논문에서 균등 샘플링.
  단 **70%만 저널에서 뽑는다.** 나머지 30%는 arXiv에 올라온 도시 컴퓨팅 논문
  (urban 관련 OpenAlex topic을 가지면서 `primary_location.source.id == S4306400194`).
  **이유:** 저널 초록은 계획·사회과학 문체이고 arXiv 초록은 ML 문체다. 저널만으로
  학습하면 **정확히 우리가 잡고 싶은 arXiv 도시 ML 논문을 낮게 점수 매긴다.**
- **Negative (~4,000)**: 같은 기간 arXiv cs.LG/cs.CV/cs.AI에서 무작위 추출,
  positive와 중복 제거, urban 서브필드 topic을 가진 것 제외.

**임베딩은 로컬 모델을 쓴다.** `sentence-transformers`의 `BAAI/bge-base-en-v1.5`
(CPU에서 동작). 이유 셋 — (1) API 키가 하나 줄어든다, (2) 2만 건 임베딩 비용이 0이고
백필·재학습을 마음대로 돌릴 수 있다, (3) **동일 입력에 동일 벡터**라 분류기 실험이
재현된다 (호스팅 임베딩은 모델이 조용히 바뀐다). `config/pipeline.yaml`의
`embedding.provider`로 나중에 호스팅 API로 교체 가능하게 두되, Phase 0 기본값은 로컬.

**모델**: title+abstract 임베딩 → **로지스틱 회귀** (scikit-learn). 20% 홀드아웃으로
AUC·precision·recall 보고. 출력은 **캘리브레이션된 확률**이므로 임계값을 해석할 수
있다 (코사인 유사도는 그렇지 않다 — 이게 centroid보다 나은 실질적 이유다).
모델과 학습 메타데이터를 `models/clf-{date}.joblib` + `.json`으로 버전 저장하고
`provenance.classifier_version`에 기록한다.

**비용**: 1만 건 × ~300토큰 = 3M 임베딩 토큰, 일회성. 무시할 수준.
**YJUN 부담**: seed 논문 큐레이션 1–2시간 → **저널 목록 검토 20분**으로 줄어든다.

**측정할 실패 모드**: 저널 편향으로 arXiv 쪽 재현율이 떨어지는지. Q1 라벨링에서
keep 라벨이 붙었는데 점수가 낮은 건을 arXiv/저널 출처별로 집계해 확인한다.

### 5.5 요약 — 2층

입력은 **초록 + 서지 메타데이터뿐.** 전문·PDF는 Phase 0에서 읽지 않는다.
서지·링크·출판상태·저자는 LLM 출력에서 받지 않고 수집기 메타데이터에서 직접 기입한다
(LLM은 저자명과 연도를 태연하게 지어낸다).

**What (2–3문장)** — 이 논문이 무엇을 했는가. 측정값을 그대로 노출한다. 표본 규모,
정확도·효과 크기, 도시 수, 데이터셋 크기, 모델명, 공간 해상도, 기간.
"3.4M street-view images across 12 cities, 15 m resolution, 2019–2023".
**초록에 숫자가 없으면 없다고 쓰지, 만들지 않는다.**

**Why it matters (1–2문장)** — 왜 중요한가. "중요하다"는 말 대신 무엇이 달라지는지.
초록이 근거를 주지 않으면 이 줄은 짧아진다.

**Caveats — 선택 필드로 강등** (v1.1 변경)

`signals`(구 `caveat_flags`)는 계속 채운다 — 정형 판정은 값싸고, 배지·필터·그래프
속성으로 여러 곳에서 재사용되기 때문이다. 다만 **서술형 caveats 줄은 Phase 0의
필수 산출물이 아니다.** 채울 재료(confidence high/medium 플래그)가 충분하면 한 줄을
생성하고, 아니면 필드를 생략한다. 수용 기준에서 제외한다.

Phase 1에서 이 줄을 서비스의 서명으로 되살릴지는 Phase 0에서 생성된 표본을 보고
결정한다 — 사실이지만 공허한 문장("this is a preprint")만 나온다면 되살릴 이유가 없다.

**출력 계약:** LLM은 JSON으로만 응답한다. 스키마 위반 시 1회 재시도, 재차 실패하면
`review.status: pending`으로 두고 파이프라인은 계속 간다. **한 아이템의 실패가 그날
발행 전체를 멈추면 안 된다.**

### 5.6 헤드라인 점수와 quiet day

```
headline = 0.40·relevance                # 분류기 확률
         + 0.20·source_multiplicity      # 클러스터 멤버 수 (정규화)
         + 0.20·artifact_completeness    # code + data + published 배지 가중
         + 0.20·novelty                  # 신규 method/tool 엔티티 등장 비율
```

가중치는 `config/scoring.yaml`. Phase 0에서 만지는 것이 정상이다.

**임계값은 백필로 정한다.** 2주 실운영 표본(70–100건)으로 분위수를 잡는 것은 통계적으로
무의미하다. 90일 백필로 수천 건의 점수 분포를 만들고, **헤드라인 발생률 30–50%**가
되는 분위수를 채택한다. 채택값과 히스토그램을 `docs/phase0-report.md`에 기록한다.

quiet day라도 **그날의 아이템 카드는 전부 발행한다.** quiet day는 빈 날이 아니라
편집 판단이 필요 없는 날이다.

### 5.7 프리뷰 렌더

`pipeline/render/preview.py` — Issue JSON + 참조 Item → 단일 HTML
(`runs/{run_id}/preview.html`), 인라인 CSS, 외부 의존 0.

**Phase 1 이식성 제약:** 마크업·카피를 `render/templates/*.html.j2`로 분리하고 Python은
데이터 정형화만 한다. Phase 1의 Astro 컴포넌트가 이 템플릿의 DOM 구조와 클래스명을
그대로 가져간다. 프리뷰를 버리는 코드로 쓰면 카드 레이아웃 결정을 두 번 하게 된다.

렌더 항목: 상단 scan_meta 라인 · 헤드라인(또는 "a quiet day in urban data science") ·
아이템 카드(2층 요약 + 파사드별 태그 + 배지 + 서지 링크) · status_changes.

**외부 이미지·figure는 어떤 경로로도 삽입하지 않는다.**

---

## 6. 저장소 구조

```
urban-currents/
├── pipeline/
│   ├── collectors/     arxiv.py  openalex.py  base.py
│   ├── dedup/          merge.py
│   ├── filters/        gate.py  embed.py  classifier.py
│   ├── linking/        openalex_passthrough.py  vocab_match.py  places.py
│   ├── summarize/      run.py  prompts/papers.md
│   ├── score/          headline.py
│   ├── graph/          build.py
│   ├── render/         preview.py  templates/
│   ├── schemas/        item.schema.json  issue.schema.json  entity.schema.json
│   ├── models.py       pydantic 모델 (스키마의 단일 소스)
│   ├── metrics.py
│   └── cli.py          uc collect|dedup|gate|classify|link|summarize|score|issue|preview|review|report
├── scripts/            build_journal_whitelist.py  build_trainset.py  train_classifier.py
├── models/             clf-{date}.joblib  clf-{date}.json
├── content/
│   ├── items/          {work_key}.json      (':' → '_')
│   ├── issues/         YYYY-MM-DD.json
│   ├── entities/       topics/ methods/ data/ tools/ people/ orgs/ places/
│   └── graph/          edges.jsonl          (빌드 산출물)
├── vocab/              methods.yaml  data.yaml  tools.yaml  places_aliases.yaml
│                       sources/arxiv.yaml  sources/journals.yaml
├── config/             scoring.yaml  pipeline.yaml
├── runs/               {run_id}/ raw/ metrics.json unmatched.jsonl preview.html
├── docs/               PRD-phase0.md  phase0-report.md  OPERATIONS.md
├── prompts/            (directing-prompt 산출물, gitignored)
├── tests/
└── README.md
```

스택: Python 3.11+ / `pyalex` / `httpx` / `pydantic` / `sentence-transformers` /
`scikit-learn` / `numpy` / `rapidfuzz` / `jinja2` / `pyyaml` / `typer` / `pytest`.
패키지 관리 `uv`. 외부 API는 OpenAlex와 Claude 둘뿐 — 임베딩은 로컬.
**DB 없음** — 콘텐츠는 git 안의 JSON이다.

---

## 7. 검수 인터페이스

Phase 0는 GitHub PR을 쓰지 않는다 (봇·Actions는 Phase 1). 대신 CLI:

```
uc review --date 2026-08-14
```

- 프리뷰 HTML을 브라우저로 열고 **시작 시각을 기록**
- 아이템별 `[a]pprove / [r]eject / [e]dit / [s]kip`
- edit 시 `$EDITOR`로 Item JSON을 열고 변경된 필드 경로를 `review.edits`에 기록
- 종료 시 소요 시간을 `runs/{run_id}/metrics.json`에 기록

**이 시간 기록이 Q4의 유일한 근거다.** 손으로 기억하면 반드시 과소보고된다.

라벨링 모드 `uc review --label relevance --date …` — 분류기 상위 30건에 keep/drop만
찍는 빠른 모드. `runs/labels/relevance.jsonl`에 append.

---

## 8. 계측

`runs/{run_id}/metrics.json`:

```jsonc
{
  "run_id": "run_2026-08-14T21-00-00Z",
  "date": "2026-08-14",
  "counts": {"arxiv_fetched": 1043, "openalex_fetched": 22, "after_dedup": 1051,
             "after_gate": 118, "classified": 118, "selected": 24,
             "summarized": 6, "published": 6},
  "cost": {"embedding_usd": 0.004, "llm_usd": 0.018,
           "openalex_usd": 0.002, "total_usd": 0.024},
  "tokens": {"in": 18420, "out": 3400},
  "timing": {"collect_s": 214, "classify_s": 31, "summarize_s": 88, "review_s": 640},
  "linking": {"topics_from_openalex": 6, "unmatched_methods": 3, "unmatched_data": 1,
              "openalex_enrich_pending": 4},
  "errors": []
}
```

`uc report`가 전체 run을 집계해 §1의 지표를 계산하고 `docs/phase0-report.md`를 생성한다.
**Phase 0의 진짜 산출물이 이 명령이다.**

---

## 9. 수용 기준 (Definition of Done)

기능:
- [ ] `uc collect --backfill 90` 이 arXiv 90일치를 레이트리밋 위반 없이 수집, raw 보존
- [ ] `uc collect --date YYYY-MM-DD` 증분 수집이 **멱등**하다
- [ ] preprint→published 전환이 신규 Item을 만들지 않고 기존 Item을 갱신한다 (테스트 포함)
- [ ] OpenAlex 보강 패스가 arXiv Item에 `ids.openalex`·`graph.*`·`topics`를 채운다.
      실패해도 파이프라인이 진행된다
- [ ] `scripts/build_journal_whitelist.py`가 저널 목록을 생성하고 사람 검토 표시를 남긴다
- [ ] `train_classifier.py`가 홀드아웃 AUC·PR 곡선을 리포트하고 모델을 버전 저장한다
- [ ] 모든 Item·Issue가 pydantic 검증을 통과한다
- [ ] `entities`의 모든 태그가 정규 ID를 갖는다. 자유 문자열 0건 (테스트로 강제)
- [ ] LLM 출력이 스키마를 위반해도 파이프라인이 계속 간다
- [ ] `uc preview`가 외부 의존 없는 단일 HTML을 낸다
- [ ] `uc review`가 검수 시간과 편집 이력을 자동 기록한다
- [ ] `uc report`가 §1의 지표를 계산한다

측정:
- [ ] Q1 AUC ≥ 0.9 (홀드아웃) + precision@10 ≥ 0.7 (150건 라벨)
- [ ] Q2 일별 통과 아이템 중앙값 ≥ 5 (90일 백필)
- [ ] Q3 quiet day 임계값 확정 + 분포 히스토그램
- [ ] Q4 7일 이상 무인 운영, 철회 항목 0건 (옛 정의 "검수 ≤ 15분"은 측정치 0일)
- [ ] 아이템당 LLM·임베딩·OpenAlex 비용 실측 → 월 환산

문서:
- [ ] `docs/phase0-report.md` — 지표, 근거, Phase 1 권고
- [ ] `docs/OPERATIONS.md` — 사람 개입 지점과 상한 (v1.0 §9)
- [ ] `README.md` (영어)

---

## 10. 리스크

| 리스크 | 완화 |
|---|---|
| OpenAlex 예산 초과 / API 정책 재변경 | `meta.cost_usd` 매 호출 누적, 일일 상한 도달 시 중단·보고. 착수 시 문서 재확인 |
| 저널 학습셋 편향 → arXiv 논문 저평가 | positive의 30%를 arXiv에서 (§5.4). Q1에서 출처별 재현율 확인 |
| cs.LG/CV/AI 게이트가 좋은 논문을 버린다 | §5.3 재현율 측정 1회 의무 |
| OpenAlex에 arXiv preprint 등재가 늦다 | 보강은 best-effort, `openalex_enrich_pending`로 재시도 |
| 90일 백필 LLM 비용 폭주 | **백필은 요약하지 않는다.** 분류기 점수만 |
| 2주 안에 다 못 끝낸다 | 우선순위: 수집·dedup·분류기·점수 > 요약 > 링킹 > 프리뷰 > 검수 CLI |

---

## 11. v1.0 기획서와 달라진 점 (누적)

| # | v1.0 | 이 PRD | 이유 |
|---|---|---|---|
| 1 | 아이템 카드 단일 객체 | Item(가변)/Issue(불변) 분리 | preprint→published 전환 |
| 2 | "규칙 → 임베딩" 순서 | 소량 카테고리는 게이트 없이 전부 분류기 | 키워드 게이트의 재현율 손실 |
| 3 | seed 논문 centroid | **저널 기반 로지스틱 회귀 분류기** | 분야의 기존 합의를 학습셋으로. 재현 가능·캘리브레이션됨 |
| 4 | Places가 시그니처 축 | **우선순위 하향, best-effort 수집만** | 추출 신뢰도 리스크. 스키마는 남겨 소급 재처리 회피 |
| 5 | Caveats가 서명 | `signals` 정형 판정 유지, 서술은 선택 | 초록만으로 공허하지 않은 비평이 나오는지 미검증 |
| 6 | Phase 0 표본으로 임계값 | 90일 백필 분포로 캘리브레이션 | 2주 표본으로는 분위수를 못 잡는다 |
| 7 | edges를 스키마에 내장 | Item에서 파생 생성 | 이중 소스 동기화 부채 |
| 8 | 검수 시간 "10–15분" 가정 | CLI 자동 실측 | Q4는 가정이 아니라 측정 대상 |
| 9 | (없음) | **OpenAlex를 Papers 스파인으로 채택** — 인용 그래프·Topics·ORCID/ROR 그대로 수용 | CC0, 재구축 불가능한 자산 |
| 10 | OpenAlex "polite pool + mailto" | **API 키 + USD 일일 예산** | 2026-02-13 정책 변경 |

---

## 12. Phase 1로 넘기는 인터페이스 약속

- `item.schema.json` / `issue.schema.json` / `entity.schema.json` — 필드 추가 가능,
  기존 필드 의미 변경·삭제는 마이그레이션 스크립트 동반
- `summary.en` 구조 (`what` / `why` / 선택 `caveats`) — `ko`는 필드 추가로만
- `content/` 디렉토리 레이아웃, `work_key` 규칙
- OpenAlex 유래 필드는 **이름을 바꾸지 않는다** (`referenced_works`, `topics`,
  `primary_location`, `cited_by_count`)
- 정규 ID 접두사: `openalex:` `orcid:` `ror:` `wikidata:` `method:` `data:` `github:`
- 렌더 템플릿의 DOM 구조·클래스명

---

## 13. Phase 0 종료 시 결정할 것

1. Phase 1 Go/No-Go (§1 지표 근거)
2. quiet day 임계값 확정치
3. 발행 빈도 (데일리 유지 여부 — Q2 결과)
4. 서술형 Caveats를 서비스의 서명으로 되살릴지 (생성 표본 검토)
5. Places 축 부활 시점과 조건
6. v1.0 §8 발행 승인 긴장 — 초기 3개월 전수 검수안 확정

---

## 14. v1.1 개정 근거

**Places 하향 (YJUN, 2026-08-12).** 특색 있는 태그가 될 것으로 봤으나 초록 기반
추출 신뢰도가 검증되지 않았고, 검증 자체가 Phase 0의 2주를 잠식한다. 시그니처
지위를 내려놓되 필드는 남긴다 — 나중에 부활시킬 때 아카이브 전체 재처리를 피하는
비용이 지금은 0이기 때문이다.

**저널 기반 분류기 (YJUN, 2026-08-12).** "명확한 필터링 기준이 떠오르지 않는다"는
문제에 대해 "SSCI Urban Studies 저널이 싣는 것이 곧 이 분야다"라는 답. 개인의
취향이 아니라 분야의 기존 합의를 학습셋으로 삼으므로 방어 가능하고 재현 가능하다.
ASJC 3322가 Scimago 카테고리이자 OpenAlex 서브필드이므로 구현도 한 줄이다.

**OpenAlex 스파인 채택 (YJUN 질의, 2026-08-12).** "왜 OpenAlex 구조를 활용하지 않나,
저작권 문제인가?" — 저작권 문제는 없다(CC0). v1.0이 OpenAlex를 Topics·저자 공급원으로만
쓰고 인용 그래프(`referenced_works`)를 놓치고 있었던 것이 설계 누락이었다. 다만
트랙 2–4가 OpenAlex 모델에 없고 오버레이 4축이 OpenAlex에 없으므로 통째 채택은
불가 — Item이 상위 추상, OpenAlex Work가 Papers 트랙 구현체 (§4.2).

**요약 2층화 (YJUN, 2026-08-12).** "무엇을 의미하며 왜 중요한지 설명하면 된다."
Caveats를 버리지는 않되 필수 산출물에서 뺀다. `signals` 정형 판정은 값이 싸고
배지·필터로 재사용되므로 유지.

**참고 출처** — [OpenAlex 라이선스 (CC0)](https://github.com/ourresearch/openalex-docs/blob/main/license.md) ·
[OpenAlex 인증·과금 (2026-02 변경)](https://developers.openalex.org/api-reference/authentication) ·
[OpenAlex Changelog](https://help.openalex.org/hc/en-us/articles/38868153578263-Changelog) ·
[OpenAlex Topics 계층](https://help.openalex.org/hc/en-us/articles/24736129405719-Topics) ·
[Work 객체 필드](https://github.com/ourresearch/openalex-docs/blob/main/api-entities/works/work-object/README.md) ·
[Sources 필터](https://developers.openalex.org/api-reference/sources) ·
[pyalex](https://github.com/J535D165/pyalex) ·
[alphaxiv-open (MIT)](https://github.com/AsyncFuncAI/alphaxiv-open)
