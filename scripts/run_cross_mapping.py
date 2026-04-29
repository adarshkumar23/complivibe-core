from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from backend.ingestion.obligation_extractor import load_obligations
from backend.mapper.cross_mapper import CrossMappingEngine, load_mappings, save_mappings


def _pct(part: int, total: int) -> float:
    if total == 0:
        return 0.0
    return (part / total) * 100


def _build_report(mappings) -> dict:
    total = len(mappings)
    counts = Counter(mapping.relationship for mapping in mappings)
    avg_overlap = sum(mapping.overlap_score for mapping in mappings) / total if total else 0.0
    eu_covered = len({mapping.eu_obligation_id for mapping in mappings})
    dpdp_covered = len({mapping.dpdp_obligation_id for mapping in mappings})

    return {
        "total_mappings": total,
        "full_overlap": counts.get("full_overlap", 0),
        "partial_overlap": counts.get("partial_overlap", 0),
        "related": counts.get("related", 0),
        "no_overlap": counts.get("no_overlap", 0),
        "avg_overlap_score": avg_overlap,
        "eu_obligations_covered": eu_covered,
        "dpdp_obligations_covered": dpdp_covered,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CompliVibe cross-mapping pipeline")
    parser.add_argument("--force", action="store_true", help="Re-run mapping even if cached file exists")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    mappings_path = root_dir / "data" / "mappings" / "cross_mappings.json"

    if mappings_path.exists() and not args.force:
        mappings = load_mappings()
        print(f"✓ Loaded {len(mappings)} existing mappings from cross_mappings.json")
        report = _build_report(mappings)
    else:
        engine = CrossMappingEngine()
        mappings = engine.run_mapping(
            semantic_threshold=0.40,
            llm_validate_above=0.30,
            max_dpdp_matches_per_eu=3,
        )
        save_mappings(mappings)
        report = engine.generate_report(mappings)

    total_mappings = int(report.get("total_mappings", 0))
    full_overlap = int(report.get("full_overlap", 0))
    partial_overlap = int(report.get("partial_overlap", 0))
    related = int(report.get("related", 0))
    no_overlap = int(report.get("no_overlap", 0))

    eu_obligations = load_obligations("eu_ai_act")
    dpdp_obligations = load_obligations("dpdp")
    eu_total = len(eu_obligations)
    dpdp_total = len(dpdp_obligations)
    eu_covered = int(report.get("eu_obligations_covered", 0))
    dpdp_covered = int(report.get("dpdp_obligations_covered", 0))

    eu_by_id = {ob.obligation_id: ob for ob in eu_obligations}
    dpdp_by_id = {ob.obligation_id: ob for ob in dpdp_obligations}

    top_overlaps = sorted(mappings, key=lambda item: item.overlap_score, reverse=True)[:5]

    print("============================================")
    print("CompliVibe Cross-Mapping Report")
    print("============================================")
    print(f"Total mappings found: {total_mappings}")
    print("")
    print("Relationship breakdown:")
    print(f"  Full overlap:     {full_overlap} ({_pct(full_overlap, total_mappings):.1f}%)")
    print(f"  Partial overlap:  {partial_overlap} ({_pct(partial_overlap, total_mappings):.1f}%)")
    print(f"  Related:          {related} ({_pct(related, total_mappings):.1f}%)")
    print(f"  No overlap:       {no_overlap} ({_pct(no_overlap, total_mappings):.1f}%)")
    print("")
    print("Coverage:")
    print(
        "  EU obligations mapped:   "
        f"{eu_covered}/{eu_total} ({_pct(eu_covered, eu_total):.1f}%)"
    )
    print(
        "  DPDP obligations mapped: "
        f"{dpdp_covered}/{dpdp_total} ({_pct(dpdp_covered, dpdp_total):.1f}%)"
    )
    print("")
    print("Top 5 Unified Actions (highest overlap):")
    if not top_overlaps:
        print("  (none)")
    else:
        for idx, mapping in enumerate(top_overlaps, start=1):
            eu_ob = eu_by_id.get(mapping.eu_obligation_id)
            dpdp_ob = dpdp_by_id.get(mapping.dpdp_obligation_id)
            eu_label = eu_ob.article if eu_ob and eu_ob.article else "?"
            dpdp_label = dpdp_ob.article if dpdp_ob and dpdp_ob.article else "?"
            eu_text = eu_ob.what_must_be_done if eu_ob else ""
            dpdp_text = dpdp_ob.what_must_be_done if dpdp_ob else ""
            print(f"  {idx}. [{eu_label} ↔ DPDP {dpdp_label}]")
            print(f"     EU: {eu_text[:80]}")
            print(f"     DPDP: {dpdp_text[:80]}")
            print(f"     Score: {mapping.overlap_score:.2f}")
    print("")
    print("Saved to data/mappings/cross_mappings.json")


if __name__ == "__main__":
    main()
