"""Build src/cxrlandmarks/vocab/{landmarks,groups,references}.csv from the final vocabulary + curation records.

One row per landmark DEFINITION (a provenance `source`: a rated candidate K###, the resident
addition, or an assistive angle-limb point K###+name). A definition can cover several classes
and both views; each (view, class, landmark) is one class-landmark pair (channel).

Inputs (defaults):
  src/cxrlandmarks/vocab/final_vocab_v1.json   final vocabulary + provenance
  curation/kp_final_v1.csv                     per-candidate review outcome
  curation/kp_decisions_round1.csv             round-1 ratings, group names
  curation/kp_rule_changes_v1.csv              rules revised after review
  curation/sources.json                        literature per clinical group

Usage: python scripts/make_landmark_table.py [--curation-dir DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cxrlandmarks.rules import schema as KS  # noqa: E402

LIMB_TAGS = ("+limbs", "+SD", "+slope", "+MS", "+axis", "(angles)")


def _rule_change_ids(rid: str) -> set[str]:
    ids = set(re.findall(r"K\d{3}", rid))
    for a, b in re.findall(r"K(\d{3})-K(\d{3})", rid):
        ids |= {f"K{i:03d}" for i in range(int(a), int(b) + 1)}
    return ids


def build(curation: Path, vocab_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    vocab = json.load(open(vocab_path))
    prov = vocab["provenance"]
    final = {r["ID"]: r for r in csv.DictReader(open(curation / "kp_final_v1.csv"))}
    decisions = list(csv.DictReader(open(curation / "kp_decisions_round1.csv")))
    group_names = collections.OrderedDict((r["group"], r["group_name"]) for r in decisions)
    changes = list(csv.DictReader(open(curation / "kp_rule_changes_v1.csv")))
    src = json.load(open(curation / "sources.json"))
    lit = collections.defaultdict(list)
    for s in src["sources"]:
        for g in s["groups"]:
            lit[g].append(s["id"])
    schemas = {"frontal": KS.SCHEMA_FRONTAL, "lateral": KS.SCHEMA_LATERAL}

    defs: dict[str, list[dict]] = collections.OrderedDict()
    for p in prov:
        defs.setdefault(p["source"], []).append(p)

    rows = []
    for sid, ps in defs.items():
        base, limb = sid.split("+")[0], "+" in sid
        names = list(dict.fromkeys(p["kp"] for p in ps))
        views = [v for v in ("frontal", "lateral") if any(p["view"] == v for p in ps)]
        cls = {v: [p["class"] for p in ps if p["view"] == v] for v in ("frontal", "lateral")}
        methods, vias = [], []
        for p in ps:
            k = next(k for k in schemas[p["view"]][p["class"]].keypoints if k.name == p["kp"])
            methods.append(k.method)
            vias.extend(k.via)
        group = next((p["group"] for p in ps if p["group"]), "")
        if not group and sid == "added":  # resident addition proposed in a comment on K120 -> K120's group
            group = next(r["group"] for r in decisions if r["ID"] == "K120")
        if sid == "added":
            revs = [c for c in changes if c["ID"].startswith("K120")]
        else:
            revs = [c for c in changes
                    if base in _rule_change_ids(c["ID"]) and any(t in c["ID"] for t in LIMB_TAGS) == limb]
        f = final.get(sid, {})
        rows.append({
            "id": sid,
            "landmark": " / ".join(names),
            "views": " + ".join(views),
            "n_pairs": len(ps),
            "classes_frontal": "; ".join(cls["frontal"]),
            "classes_lateral": "; ".join(cls["lateral"]),
            "group": group,
            "group_name": group_names.get(group, ""),
            "rule_method": "; ".join(dict.fromkeys(methods)),
            "rule_partners": "; ".join(dict.fromkeys(v.strip() for v in vias)),
            "rule_revision_v1": " | ".join(f"{c['v1 rule']} [{c['status']}]" for c in revs),
            "review_outcome": "; ".join(dict.fromkeys(p["decision"] for p in ps)),
            "resident_relevance": f.get("resident_relevance", ""),
            "attending_group_relevance": f.get("attending_group", ""),
            "reviewer_comment": f.get("resident_comment", ""),
            "literature": "; ".join(lit.get(group, [])),
        })

    groups = [{"group": g, "group_name": n,
               "n_definitions_final": sum(r["group"] == g for r in rows),
               "relevance": src.get("relevance", {}).get(g, ""),
               "literature": "; ".join(lit.get(g, []))} for g, n in group_names.items()]
    refs = [{"id": s["id"], "groups": "; ".join(s["groups"]), "citation": s.get("citation", ""), "year": s.get("year", ""),
             "url": s.get("url", ""), "type": s.get("type", ""), "supports": s.get("supports", ""),
             "verified": s.get("verified", "")}
            for s in src["sources"]]
    return rows, groups, refs


def _write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--curation-dir", type=Path, default=REPO / "curation")
    ap.add_argument("--out-dir", type=Path, default=REPO / "src" / "cxrlandmarks" / "vocab")
    args = ap.parse_args(argv)
    rows, groups, refs = build(args.curation_dir, Path(KS.FINAL_VOCAB_PATH))
    _write(args.out_dir / "landmarks.csv", rows)
    _write(args.out_dir / "groups.csv", groups)
    _write(args.out_dir / "references.csv", refs)
    print(f"{len(rows)} definitions, {sum(r['n_pairs'] for r in rows)} class-landmark pairs, "
          f"{len(groups)} groups -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
