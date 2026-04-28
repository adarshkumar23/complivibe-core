from __future__ import annotations

from collections import Counter
from statistics import mean

from backend.ingestion.embedder import RegulationEmbedder
from backend.ingestion.obligation_extractor import load_obligations


def _summarize_obligations(label: str, obligations: list) -> dict:
    if not obligations:
        print(f"\n{label} - no obligations found")
        return {}

    type_counts = Counter(item.obligation_type for item in obligations)
    who_counts = Counter(item.who_must_comply for item in obligations)
    tier_counts = Counter(
        item.applies_to_risk_tier or "unspecified" for item in obligations
    )

    deadline_count = sum(1 for item in obligations if item.deadline)
    penalty_count = sum(
        1
        for item in obligations
        if item.penalty_max_eur is not None or item.penalty_max_inr is not None
    )
    avg_confidence = mean(item.confidence for item in obligations)

    print(f"\n{label} Quality Report")
    print("-" * 60)
    print(f"Total obligations: {len(obligations)}")
    print(f"Breakdown by obligation_type: {dict(type_counts)}")
    print(f"Breakdown by who_must_comply: {dict(who_counts)}")
    print(f"Breakdown by applies_to_risk_tier: {dict(tier_counts)}")
    print(f"Count with deadlines set: {deadline_count}")
    print(f"Count with penalty amounts set: {penalty_count}")
    print(f"Average confidence score: {avg_confidence:.3f}")

    top_samples = sorted(obligations, key=lambda item: item.confidence, reverse=True)[:3]
    print("\nTop 3 sample obligations (highest confidence):")
    for item in top_samples:
        print("-" * 60)
        print(item)

    return {
        "total": len(obligations),
        "type_counts": dict(type_counts),
        "who_counts": dict(who_counts),
        "tier_counts": dict(tier_counts),
    }


def _print_search_results(embedder: RegulationEmbedder) -> None:
    query = "high risk AI system transparency requirements"
    results = embedder.search(query, top_k=3)

    print("\nSemantic Search Test")
    print("-" * 60)
    print(f"Query: {query}")
    if not results:
        print("No results returned.")
        return

    for idx, item in enumerate(results, start=1):
        metadata = item.get("metadata") or {}
        regulation = metadata.get("regulation", "unknown")
        article = metadata.get("article") or "unknown"
        text = item.get("text", "")
        preview = " ".join(text.split())[:240]
        print(f"\nResult {idx}:")
        print(f"Regulation: {regulation}")
        print(f"Article: {article}")
        print(f"Text preview: {preview}...")


def main() -> None:
    eu_obligations = load_obligations("eu_ai_act")
    dpdp_obligations = load_obligations("dpdp")

    _summarize_obligations("EU AI Act", eu_obligations)
    _summarize_obligations("DPDP", dpdp_obligations)

    embedder = RegulationEmbedder()
    _print_search_results(embedder)


if __name__ == "__main__":
    main()
