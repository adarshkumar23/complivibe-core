from __future__ import annotations

import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import util
from tqdm import tqdm

from backend.ingestion.embedder import RegulationEmbedder
from backend.ingestion.obligation_extractor import Obligation, load_obligations


@dataclass(frozen=True)
class ObligationMap:
    map_id: str
    eu_obligation_id: str
    dpdp_obligation_id: str
    eu_obligation_type: str
    dpdp_obligation_type: str
    relationship: str
    overlap_score: float
    eu_satisfied_by_dpdp: bool
    dpdp_satisfied_by_eu: bool
    unified_action: str
    eu_additional_requirements: Optional[str]
    dpdp_additional_requirements: Optional[str]
    confidence: float
    mapping_method: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ObligationMap":
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass(frozen=True)
class MappingStats:
    total_mappings: int
    full_overlap: int
    partial_overlap: int
    related: int
    no_overlap: int
    avg_overlap_score: float
    eu_obligations_covered: int
    dpdp_obligations_covered: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MappingStats":
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


class SemanticMapper:
    SIMILARITY_THRESHOLDS = {
        "full_overlap": 0.80,
        "partial_overlap": 0.60,
        "related": 0.40,
        "no_overlap": 0.0,
    }

    def __init__(self, embedder: RegulationEmbedder) -> None:
        self.embedder = embedder
        self.eu_obligations = load_obligations("eu_ai_act")
        self.dpdp_obligations = load_obligations("dpdp")

        model = getattr(embedder, "model", None)
        if model is None:
            model = getattr(embedder, "_model", None)
        if model is None:
            raise AttributeError("RegulationEmbedder missing model")

        eu_texts = [ob.what_must_be_done for ob in self.eu_obligations]
        dpdp_texts = [ob.what_must_be_done for ob in self.dpdp_obligations]

        self.eu_embeddings = (
            np.asarray(model.encode(eu_texts, show_progress_bar=False))
            if eu_texts
            else np.empty((0, 0))
        )
        self.dpdp_embeddings = (
            np.asarray(model.encode(dpdp_texts, show_progress_bar=False))
            if dpdp_texts
            else np.empty((0, 0))
        )
        self._eu_index_by_id = {
            ob.obligation_id: idx for idx, ob in enumerate(self.eu_obligations)
        }

    def compute_similarity_matrix(self) -> np.ndarray:
        if self.eu_embeddings.size == 0 or self.dpdp_embeddings.size == 0:
            return np.zeros((len(self.eu_obligations), len(self.dpdp_obligations)))

        scores = util.cos_sim(self.eu_embeddings, self.dpdp_embeddings)
        if hasattr(scores, "cpu"):
            scores = scores.cpu().numpy()
        return np.asarray(scores)

    def get_top_matches(self, eu_obligation: Obligation, top_k: int = 5) -> list[tuple[Obligation, float]]:
        if not self.dpdp_obligations:
            return []

        idx = self._eu_index_by_id.get(eu_obligation.obligation_id)
        if idx is None:
            raise ValueError("EU obligation not found in mapper")

        eu_vec = self.eu_embeddings[idx : idx + 1]
        scores = util.cos_sim(eu_vec, self.dpdp_embeddings)
        if hasattr(scores, "cpu"):
            scores = scores.cpu().numpy()
        scores = np.asarray(scores).flatten()

        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.dpdp_obligations[i], float(scores[i])) for i in top_indices]


class LLMMapper:
    def __init__(self) -> None:
        from google import genai
        from google.genai import types

        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = os.getenv("LLM_MODEL_EXTRACTION", "gemini-2.5-flash")

    def _call_llm(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                return response.text
            except Exception as exc:
                error_text = str(exc)
                status_code = getattr(exc, "status_code", None)
                if status_code in (429, 503) or "429" in error_text or "503" in error_text:
                    wait_seconds = None
                    if "retrydelay" in error_text.lower() or "retry in" in error_text.lower():
                        match = re.search(r"retry in (\d+\.?\d*)s", error_text, re.IGNORECASE)
                        if match:
                            wait_seconds = float(match.group(1)) + 2

                    if wait_seconds is None:
                        wait_seconds = [15, 30, 60][attempt]

                    print(f"⏳ Rate limited — waiting {wait_seconds}s before retry {attempt + 1}/3")
                    if attempt < 2:
                        time.sleep(wait_seconds)
                        continue

                raise exc

        raise RuntimeError("LLM failed after retries")

    def validate_and_enrich_mapping(
        self,
        eu_ob: Obligation,
        dpdp_ob: Obligation,
        semantic_score: float,
    ) -> ObligationMap:
        prompt = (
            "You are a legal AI expert specializing in EU AI Act and India "
            "DPDP Act compliance cross-mapping.\n\n"
            "Analyze these two regulatory obligations and determine their "
            "relationship:\n\n"
            "EU AI Act Obligation:\n"
            f"- Article: {eu_ob.article}\n"
            f"- Type: {eu_ob.obligation_type}\n"
            f"- Who must comply: {eu_ob.who_must_comply}\n"
            f"- What must be done: {eu_ob.what_must_be_done}\n\n"
            "DPDP Act Obligation:\n"
            f"- Section: {dpdp_ob.article}\n"
            f"- Type: {dpdp_ob.obligation_type}\n"
            f"- Who must comply: {dpdp_ob.who_must_comply}\n"
            f"- What must be done: {dpdp_ob.what_must_be_done}\n\n"
            f"Semantic similarity score: {semantic_score:.2f}\n\n"
            "Return ONLY a JSON object with:\n"
            "- relationship: full_overlap/partial_overlap/related/no_overlap\n"
            "- overlap_score: float 0.0-1.0\n"
            "- eu_satisfied_by_dpdp: boolean\n"
            "- dpdp_satisfied_by_eu: boolean\n"
            "- unified_action: single action satisfying both obligations "
            "(max 300 chars, null if no_overlap)\n"
            "- eu_additional_requirements: what EU AI Act requires beyond "
            "DPDP (null if none)\n"
            "- dpdp_additional_requirements: what DPDP requires beyond EU "
            "AI Act (null if none)\n"
            "- confidence: float 0.0-1.0\n\n"
            "No markdown. No explanation. Valid JSON only."
        )

        try:
            response_text = self._call_llm(prompt)
            payload = json.loads(response_text)
            if not isinstance(payload, dict):
                raise ValueError("LLM did not return a JSON object")

            relationship = str(payload.get("relationship", "no_overlap")).strip()
            overlap_score = float(payload.get("overlap_score", semantic_score))
            eu_satisfied_by_dpdp = bool(payload.get("eu_satisfied_by_dpdp", False))
            dpdp_satisfied_by_eu = bool(payload.get("dpdp_satisfied_by_eu", False))
            unified_action = payload.get("unified_action")
            if unified_action is None:
                unified_action = ""
            eu_additional = payload.get("eu_additional_requirements")
            dpdp_additional = payload.get("dpdp_additional_requirements")
            confidence = float(payload.get("confidence", 0.0))

            if relationship == "related" and not unified_action:
                unified_action = (
                    "Address both: "
                    f"{eu_ob.what_must_be_done[:100]} AND "
                    f"{dpdp_ob.what_must_be_done[:100]}"
                )[:300]

            return ObligationMap(
                map_id=f"map_{eu_ob.obligation_id}_{dpdp_ob.obligation_id}",
                eu_obligation_id=eu_ob.obligation_id,
                dpdp_obligation_id=dpdp_ob.obligation_id,
                eu_obligation_type=eu_ob.obligation_type,
                dpdp_obligation_type=dpdp_ob.obligation_type,
                relationship=relationship,
                overlap_score=overlap_score,
                eu_satisfied_by_dpdp=eu_satisfied_by_dpdp,
                dpdp_satisfied_by_eu=dpdp_satisfied_by_eu,
                unified_action=unified_action[:300],
                eu_additional_requirements=eu_additional,
                dpdp_additional_requirements=dpdp_additional,
                confidence=confidence,
                mapping_method="llm",
            )
        except Exception:
            relationship = _relationship_from_score(semantic_score)
            unified_action = ""
            if relationship == "related":
                unified_action = (
                    "Address both: "
                    f"{eu_ob.what_must_be_done[:100]} AND "
                    f"{dpdp_ob.what_must_be_done[:100]}"
                )[:300]
            return ObligationMap(
                map_id=f"map_{eu_ob.obligation_id}_{dpdp_ob.obligation_id}",
                eu_obligation_id=eu_ob.obligation_id,
                dpdp_obligation_id=dpdp_ob.obligation_id,
                eu_obligation_type=eu_ob.obligation_type,
                dpdp_obligation_type=dpdp_ob.obligation_type,
                relationship=relationship,
                overlap_score=float(semantic_score),
                eu_satisfied_by_dpdp=relationship == "full_overlap",
                dpdp_satisfied_by_eu=relationship == "full_overlap",
                unified_action=unified_action,
                eu_additional_requirements=None,
                dpdp_additional_requirements=None,
                confidence=0.0,
                mapping_method="semantic",
            )


def _relationship_from_score(score: float) -> str:
    if score >= SemanticMapper.SIMILARITY_THRESHOLDS["full_overlap"]:
        return "full_overlap"
    if score >= SemanticMapper.SIMILARITY_THRESHOLDS["partial_overlap"]:
        return "partial_overlap"
    if score >= SemanticMapper.SIMILARITY_THRESHOLDS["related"]:
        return "related"
    return "no_overlap"


class CrossMappingEngine:
    def __init__(self) -> None:
        self.semantic_mapper = SemanticMapper(RegulationEmbedder())
        self.llm_mapper = LLMMapper()
        self.eu_obligations = self.semantic_mapper.eu_obligations
        self.dpdp_obligations = self.semantic_mapper.dpdp_obligations

    def run_mapping(
        self,
        semantic_threshold: float = 0.40,
        llm_validate_above: float = 0.30,
        max_dpdp_matches_per_eu: int = 3,
    ) -> list[ObligationMap]:
        similarity = self.semantic_mapper.compute_similarity_matrix()
        mappings_by_pair: dict[tuple[str, str], ObligationMap] = {}
        min_semantic = 0.40

        for eu_idx, eu_ob in enumerate(tqdm(self.eu_obligations, desc="Cross-mapping", unit="obligation")):
            if similarity.size == 0:
                continue
            scores = similarity[eu_idx]
            candidate_indices = np.where(scores >= semantic_threshold)[0]
            if candidate_indices.size == 0:
                continue

            ordered = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
            ordered = ordered[: max_dpdp_matches_per_eu]

            for dpdp_idx in ordered:
                score = float(scores[dpdp_idx])
                if score < min_semantic:
                    continue

                dpdp_ob = self.dpdp_obligations[dpdp_idx]
                if score >= llm_validate_above:
                    mapping = self.llm_mapper.validate_and_enrich_mapping(
                        eu_ob,
                        dpdp_ob,
                        score,
                    )
                else:
                    relationship = _relationship_from_score(score)
                    mapping = ObligationMap(
                        map_id=f"map_{eu_ob.obligation_id}_{dpdp_ob.obligation_id}",
                        eu_obligation_id=eu_ob.obligation_id,
                        dpdp_obligation_id=dpdp_ob.obligation_id,
                        eu_obligation_type=eu_ob.obligation_type,
                        dpdp_obligation_type=dpdp_ob.obligation_type,
                        relationship=relationship,
                        overlap_score=score,
                        eu_satisfied_by_dpdp=relationship == "full_overlap",
                        dpdp_satisfied_by_eu=relationship == "full_overlap",
                        unified_action="",
                        eu_additional_requirements=None,
                        dpdp_additional_requirements=None,
                        confidence=0.0,
                        mapping_method="semantic",
                    )

                mappings_by_pair[(mapping.eu_obligation_id, mapping.dpdp_obligation_id)] = mapping

        mappings = sorted(mappings_by_pair.values(), key=lambda item: item.overlap_score, reverse=True)
        save_mappings(mappings)
        return mappings

    def generate_report(self, mappings: Optional[list[ObligationMap]] = None) -> dict:
        if mappings is None:
            mappings = load_mappings()

        total = len(mappings)
        counts = Counter(map_item.relationship for map_item in mappings)
        avg_overlap = float(np.mean([m.overlap_score for m in mappings])) if mappings else 0.0
        eu_covered = len({m.eu_obligation_id for m in mappings})
        dpdp_covered = len({m.dpdp_obligation_id for m in mappings})

        stats = MappingStats(
            total_mappings=total,
            full_overlap=counts.get("full_overlap", 0),
            partial_overlap=counts.get("partial_overlap", 0),
            related=counts.get("related", 0),
            no_overlap=counts.get("no_overlap", 0),
            avg_overlap_score=avg_overlap,
            eu_obligations_covered=eu_covered,
            dpdp_obligations_covered=dpdp_covered,
        )

        top_10 = sorted(mappings, key=lambda item: item.overlap_score, reverse=True)[:10]
        types_covered: dict[str, set[str]] = defaultdict(set)
        for mapping in mappings:
            types_covered[mapping.eu_obligation_type].add(mapping.dpdp_obligation_type)

        unified_actions_count = sum(1 for mapping in mappings if mapping.unified_action)

        report = stats.to_dict()
        report.update(
            {
                "top_10_overlaps": [item.to_dict() for item in top_10],
                "obligation_types_covered": {
                    key: sorted(values) for key, values in types_covered.items()
                },
                "unified_actions_count": unified_actions_count,
            }
        )
        return report


def _mappings_path() -> Path:
    root_dir = Path(__file__).resolve().parents[2]
    return root_dir / "data" / "mappings" / "cross_mappings.json"


def save_mappings(mappings: list[ObligationMap]) -> Path:
    output_path = _mappings_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [mapping.to_dict() for mapping in mappings]
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)

    print(f"✓ saved {len(payload)} mappings to {output_path}")
    return output_path


def load_mappings() -> list[ObligationMap]:
    input_path = _mappings_path()
    if not input_path.exists():
        raise FileNotFoundError(f"Cross-mapping file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    mappings = [ObligationMap.from_dict(item) for item in payload]
    print(f"✓ loaded {len(mappings)} mappings from {input_path}")
    return mappings
