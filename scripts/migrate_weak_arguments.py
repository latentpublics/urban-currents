"""Rename `drop_weak_results` to `drop_weak_arguments` (phase 0Q, R1).

## Why this is a migration and not a re-judgement

YJUN, who made all fifteen of these judgements, corrected what the label meant:

> "제가 `drop_weak_results`라고 라벨링 하였던 것의 의미를 엄밀히 따져서
> 정정해보자면, `drop_weak_arguments`에 가까울 것 같습니다. 논문에서 주장하고자
> 하는 바(결과라고 예상되는 내용)가 너무 좁거나 약하다는 뜻이었습니다.
> 초록으로 판단하기엔 부족하지만, 그래도 판단해볼 수 있는 내용입니다."

**The judgements do not change. The category's name does.** Every row keeps the
verdict it was given; only the string naming that verdict is corrected. That is
why this is not written as a `corrected_from` append — an append would claim the
labeller changed their mind about a paper, and they did not. They changed what
the box is called.

The same reasoning says the standing rule against re-judging label files is not
broken here: nothing is re-judged.

## What it changes downstream, which is larger than the name

0P §5 removed these rows from a gate evaluation on the grounds that weak results
"cannot be predicted from an abstract", and reported that removing them lifts
journal precision@10 from 0.660 to 0.700. **The labeller has withdrawn that
premise**: a weak argument is visible in an abstract, imperfectly but visibly.
So the rows stay in, and Q1b is 0.600 / 0.660. See D204.

## Safety

`--check` reports without writing. `--revert` undoes it. Any row carrying a
label this migration does not recognise is **left alone and reported**, and the
exit code is non-zero — the same shape as `migrate_drop_lens.py` refusing to
discard a non-null `lens`.

Usage:
    uv run python scripts/migrate_weak_arguments.py --check
    uv run python scripts/migrate_weak_arguments.py
    uv run python scripts/migrate_weak_arguments.py --revert
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import paths  # noqa: E402

OLD = "drop_weak_results"
NEW = "drop_weak_arguments"

# Every label string that may legitimately appear in a label file. Anything else
# means this migration is looking at data it does not understand, and it stops
# rather than guessing.
KNOWN = {
    "keep",
    "drop_not_urban",
    "drop_not_our_kind",
    "drop_weak_method",
    "drop_weak",  # the pre-M1 merged label, still read
    OLD,
    NEW,
    "skip",
}

LABEL_FILES = (
    "relevance.jsonl",
    "affinity_probe.jsonl",
    "code_probe.jsonl",
    "subfield_check.jsonl",
    "held_review.jsonl",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Report without writing")
    ap.add_argument("--revert", action="store_true", help=f"{NEW} back to {OLD}")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    frm, to = (NEW, OLD) if a.revert else (OLD, NEW)
    report: dict[str, object] = {}
    unknown: list = []
    total = 0

    for name in LABEL_FILES:
        path = paths.LABELS / name
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        out: list[str] = []
        changed = 0
        for line in lines:
            if not line.strip():
                out.append(line)
                continue
            row = json.loads(line)
            label = row.get("label")
            if label is not None and label not in KNOWN:
                unknown.append({"file": name, "label": label,
                                "work_key": row.get("work_key")})
                out.append(line)
                continue
            # `corrected_from` records what a row used to be. It has to move
            # with the label, or the audit trail starts naming a category that
            # no longer exists.
            touched = False
            if label == frm:
                row["label"] = to
                touched = True
            if row.get("corrected_from") == frm:
                row["corrected_from"] = to
                touched = True
            if touched:
                changed += 1
                out.append(json.dumps(row, sort_keys=True, ensure_ascii=False))
            else:
                out.append(line)

        report[name] = {"rows": len(lines), "renamed": changed}
        total += changed
        if changed and not a.check and not unknown:
            path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")

    verb = "would rename" if a.check else "renamed"
    print(json.dumps({
        "direction": f"{frm} -> {to}",
        verb: total,
        "by_file": report,
        "unknown_labels_left_alone": unknown,
    }, indent=2, ensure_ascii=False))

    if unknown:
        print(
            f"\n{len(unknown)} row(s) carry a label this migration does not know. "
            f"Nothing was written. Add the label to KNOWN if it is legitimate."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
