from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Optional

from tqdm import tqdm

from backend.core.config import config
from backend.ingestion.chunker import RegulationChunk


@dataclass(frozen=True)
class Obligation:
    obligation_id: str
    regulation: str
    regulation_display: str
    article: Optional[str]
    chapter: Optional[str]
    obligation_type: str
    who_must_comply: str
    what_must_be_done: str
    legal_basis: str
    applies_to_risk_tier: Optional[str]
    deadline: Optional[str]
    penalty_max_eur: Optional[int]
    penalty_max_inr: Optional[int]
    source_chunk_id: str
    confidence: float
    raw_text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(payload: dict) -> "Obligation":
        return Obligation(**payload)


def save_obligations(obligations: Iterable[Obligation], regulation_key: str) -> Path:
    config.processed_data_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.processed_data_dir / f"{regulation_key}_obligations.json"

    payload = [item.to_dict() for item in obligations]
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)

    print(f"✓ {regulation_key}: saved {len(payload)} obligations to {output_path}")
    return output_path


def load_obligations(regulation_key: str) -> list[Obligation]:
    input_path = config.processed_data_dir / f"{regulation_key}_obligations.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Obligations file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    obligations = [Obligation.from_dict(item) for item in payload]
    print(f"✓ {regulation_key}: loaded {len(obligations)} obligations from {input_path}")
    return obligations


_SYSTEM_PROMPT = (
    "You are a legal AI specialist extracting compliance obligations from \n"
    "regulatory texts. Extract ALL distinct obligations from the provided \n"
    "regulatory text chunk.\n\n"
    "For each obligation return a JSON object with these exact fields:\n"
    "- obligation_type: one of [transparency, documentation, risk_management, \n"
    "  data_governance, human_oversight, accuracy_robustness, prohibited, \n"
    "  registration, notification, consent, rights, security, other]\n"
    "- who_must_comply: one of [provider, deployer, importer, distributor, \n"
    "  data_fiduciary, data_processor, all]\n"
    "- what_must_be_done: plain English summary under 200 characters\n"
    "- applies_to_risk_tier: one of [unacceptable, high, limited, minimal, \n"
    "  all, gpai] or null\n"
    "- penalty_max_eur: integer in EUR or null (for EU AI Act only)\n"
    "- penalty_max_inr: integer in INR or null (for DPDP only)\n"
    "- confidence: float 0.0 to 1.0\n\n"
    "Return ONLY a valid JSON array. No explanation. No markdown. \n"
    "If no obligations exist in this chunk return []."
)


class ObligationExtractor:
    def __init__(self):
        from google import genai
        from google.genai import types
        import os
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = os.getenv("LLM_MODEL_EXTRACTION", "gemini-2.5-flash")

    def _call_llm(self, prompt: str) -> str:
        import time
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                return response.text
            except Exception as e:
                if attempt < 2:
                    time.sleep(10)
                else:
                    raise e

    def extract_by_article(self, regulation: str, regulation_display: str, article: str, chunks: list[RegulationChunk]) -> list[Obligation]:
        batch_text = "\n\n".join(c.text for c in chunks)
        prompt = (
            "You are a legal AI extracting compliance obligations from regulatory text.\n"
            f"Regulation: {regulation_display}\n"
            f"Article/Section: {article}\n\n"
            "Text:\n"
            f"{batch_text}\n\n"
            "Extract ALL distinct compliance obligations. Return ONLY a valid JSON array. Each object must have:\n"
            "- obligation_type (transparency/documentation/risk_management/\n"
            "  data_governance/human_oversight/accuracy_robustness/prohibited/\n"
            "  registration/notification/consent/rights/security/other)\n"
            "- who_must_comply (provider/deployer/importer/distributor/\n"
            "  data_fiduciary/data_processor/all)\n"
            "- what_must_be_done (plain English, max 200 chars)\n"
            "- applies_to_risk_tier (unacceptable/high/limited/minimal/all/gpai \n"
            "  or null)\n"
            "- penalty_max_eur (integer or null)\n"
            "- penalty_max_inr (integer or null)\n"
            "- confidence (float 0.0-1.0)\n\n"
            "Return [] if no obligations found. No markdown. No explanation."
        )

        try:
            text = self._call_llm(prompt)
            payload = json.loads(text)
            if not isinstance(payload, list):
                raise ValueError("LLM did not return a JSON array")
        except Exception as exc:
            print(f"✗ obligation extraction failed for {regulation} Article {article}: {exc}")
            return []

        obligations: list[Obligation] = []
        for idx, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                continue

            obligation_type = str(item.get("obligation_type", "other")).strip()
            who_must_comply = str(item.get("who_must_comply", "all")).strip()
            what_must_be_done = str(item.get("what_must_be_done", "")).strip()[:200]
            applies_to_risk_tier = _safe_risk_tier(item.get("applies_to_risk_tier"))
            penalty_max_eur = _safe_int(item.get("penalty_max_eur"))
            penalty_max_inr = _safe_int(item.get("penalty_max_inr"))
            confidence = _safe_float(item.get("confidence"))

            if regulation == "dpdp":
                penalty_max_eur = None
            else:
                penalty_max_inr = None

            legal_basis = _legal_basis(regulation, article)
            source_chunk_id = chunks[0].chunk_id if chunks else ""

            obligation_id = _build_obligation_id(regulation, article, chunks[0].chunk_index if chunks else 0, idx)

            obligations.append(
                Obligation(
                    obligation_id=obligation_id,
                    regulation=regulation,
                    regulation_display=regulation_display,
                    article=article,
                    chapter=chunks[0].chapter if chunks else None,
                    obligation_type=obligation_type,
                    who_must_comply=who_must_comply,
                    what_must_be_done=what_must_be_done,
                    legal_basis=legal_basis,
                    applies_to_risk_tier=applies_to_risk_tier,
                    deadline=None,
                    penalty_max_eur=penalty_max_eur,
                    penalty_max_inr=penalty_max_inr,
                    source_chunk_id=source_chunk_id,
                    confidence=confidence,
                    raw_text=batch_text,
                )
            )

        return obligations

    def extract_all(self, chunks: list[RegulationChunk], skip_existing: bool = True, batch_by_article: bool = True) -> list[Obligation]:
        obligations_by_id: dict[str, Obligation] = {}

        if batch_by_article:
            # group chunks by article
            groups: dict[str, list[RegulationChunk]] = {}
            for c in chunks:
                if not c.article:
                    continue
                groups.setdefault(str(c.article), []).append(c)

            for group_idx, (article, group_chunks) in enumerate(tqdm(list(groups.items()), desc="Extracting obligations", unit="article"), start=1):
                regulation = group_chunks[0].regulation
                regulation_display = group_chunks[0].regulation_display
                extracted = self.extract_by_article(regulation, regulation_display, article, group_chunks)
                for ob in extracted:
                    if skip_existing and ob.obligation_id in obligations_by_id:
                        continue
                    obligations_by_id[ob.obligation_id] = ob
        else:
            for c in tqdm(chunks, desc="Extracting obligations", unit="chunk"):
                if not c.article:
                    continue
                extracted = self.extract_by_article(c.regulation, c.regulation_display, c.article, [c])
                for ob in extracted:
                    if skip_existing and ob.obligation_id in obligations_by_id:
                        continue
                    obligations_by_id[ob.obligation_id] = ob

        obligations = list(obligations_by_id.values())
        type_counts = Counter(item.obligation_type for item in obligations)
        who_counts = Counter(item.who_must_comply for item in obligations)

        print(f"✓ extracted {len(obligations)} obligations")
        print(f"✓ by type: {dict(type_counts)}")
        print(f"✓ by who_must_comply: {dict(who_counts)}")

        return obligations


def _safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_risk_tier(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    allowed = {"unacceptable", "high", "limited", "minimal", "all", "gpai"}
    return text if text in allowed else None


def _legal_basis(regulation: str, article: Optional[str]) -> str:
    if not article:
        return ""
    if regulation == "dpdp":
        return f"Section {article}"
    return f"Article {article}"


def _build_obligation_id(
    regulation: str, article: Optional[str], chunk_index: int, index: int
) -> str:
    prefix = "sec" if regulation == "dpdp" else "art"
    label = (article or "0").replace(" ", "")
    return f"{regulation}_{prefix}{label}_ob_{chunk_index:04d}_{index:03d}"


def enrich_deadlines(obligations: list[Obligation]) -> list[Obligation]:
    enriched: list[Obligation] = []

    for item in obligations:
        if item.deadline is not None:
            enriched.append(item)
            continue

        deadline = _infer_deadline(item)
        if deadline is None:
            enriched.append(item)
        else:
            enriched.append(replace(item, deadline=deadline))

    return enriched


def _infer_deadline(obligation: Obligation) -> Optional[str]:
    if obligation.regulation == "dpdp":
        if obligation.obligation_type in {"consent", "rights"}:
            return "2026-01-01"
        return "2025-11-13"

    if obligation.regulation == "eu_ai_act":
        if obligation.obligation_type == "prohibited":
            return "2025-02-02"
        if obligation.applies_to_risk_tier == "gpai":
            return "2025-08-02"
        if obligation.applies_to_risk_tier == "high":
            return "2026-08-02"
        if obligation.obligation_type == "transparency" or obligation.applies_to_risk_tier == "limited":
            return "2026-08-02"
        return "2027-08-02"

    return None
