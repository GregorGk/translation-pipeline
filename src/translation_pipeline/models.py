from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

LanguageCode = Literal["EN", "PT-BR", "PL", "FR", "DE", "RU", "UK", "EL"]
"""The eight DeepL-supported languages this pipeline targets in any direction."""

SUPPORTED_LANGUAGES: tuple[LanguageCode, ...] = (
    "EN", "PT-BR", "PL", "FR", "DE", "RU", "UK", "EL",
)

CritiqueSeverity = Literal["low", "medium", "high"]
CritiqueCategory = Literal[
    "accuracy", "fluency", "terminology", "register", "idiom", "cultural"
]
DraftSource = Literal["deepl", "claude"]
StageStatus = Literal["ok", "skipped", "failed"]
StageCriticality = Literal["critical", "non_critical"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _Mutable(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LanguagePair(_Frozen):
    source: LanguageCode
    target: LanguageCode


class SourceDocument(_Frozen):
    path: Path
    text: str
    source_language: LanguageCode
    # Block boundaries when input is structured (DOCX/PDF). Used by ConsistencyStage
    # to split the translated output back into per-block strings via the [[BLK]]
    # sentinel woven into ``text``. Empty for plain-text inputs.
    blocks: tuple[str, ...] = ()


class GlossaryEntry(_Frozen):
    source_term: str
    target_term: str
    note: str | None = None


class TranslationBrief(_Frozen):
    document_type: str
    # Renamed from `register` (PLAN.md) to avoid shadowing ABCMeta.register inherited
    # via BaseModel's metaclass; semantically the same field.
    register_level: str
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    cultural_notes: list[str] = Field(default_factory=list)
    target_audience: str
    special_instructions: list[str] = Field(default_factory=list)


class Chunk(_Frozen):
    index: int
    text: str
    prev_context: str = ""
    next_context: str = ""


class Draft(_Frozen):
    source: DraftSource
    chunks: list[str]


class SynthesizedTranslation(_Frozen):
    text: str
    chunk_alignments: list[str]


class CritiqueIssue(_Frozen):
    category: CritiqueCategory
    severity: CritiqueSeverity
    location: str
    description: str
    suggested_fix: str


class Critique(_Frozen):
    issues: list[CritiqueIssue]
    overall_assessment: str


class IssueDecision(_Frozen):
    issue: CritiqueIssue
    accepted: bool
    reasoning: str


class RevisedTranslation(_Frozen):
    text: str
    issues_addressed: list[IssueDecision] = Field(default_factory=list)
    issues_rejected_with_reason: list[IssueDecision] = Field(default_factory=list)


class BackTranslation(_Frozen):
    text: str


class Divergence(_Frozen):
    segment: str
    source_text: str
    back_translated_text: str
    severity: CritiqueSeverity
    description: str


class FinalOutput(_Frozen):
    text: str
    language_pair: LanguagePair
    brief: TranslationBrief
    glossary_used: list[GlossaryEntry] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # Per-block translated text aligned 1:1 with ``SourceDocument.blocks``. Empty
    # when the input was unstructured or sentinel alignment failed (in which case
    # the format-preserving writers fall back to fresh-document writers).
    blocks: list[str] = Field(default_factory=list)


class StageRecord(_Mutable):
    name: str
    model: str | None = None
    started_at: datetime
    completed_at: datetime
    duration_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    status: StageStatus
    error: str | None = None
    attempts: int = 1


class RunMetadata(_Mutable):
    run_id: str
    source_path: Path
    language_pair: LanguagePair
    pipeline_version: str
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    stages: list[StageRecord] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    total_duration_s: float = 0.0
    warnings: list[str] = Field(default_factory=list)

    def add_stage(self, record: StageRecord) -> None:
        self.stages.append(record)
        self.total_cost_usd += record.cost_usd
        self.total_duration_s += record.duration_s


class PipelineState(_Mutable):
    """Accumulates outputs as the pipeline progresses.

    Each stage reads what it needs and writes its own field. Required fields are set at
    construction; optional fields are populated by stages in order.
    """

    source: SourceDocument
    language_pair: LanguagePair
    metadata: RunMetadata

    brief: TranslationBrief | None = None
    chunks: list[Chunk] = Field(default_factory=list)
    draft_a: Draft | None = None
    draft_b: Draft | None = None
    synthesis: SynthesizedTranslation | None = None
    critique: Critique | None = None
    revised: RevisedTranslation | None = None
    back_translation: BackTranslation | None = None
    divergences: list[Divergence] = Field(default_factory=list)
    final_output: FinalOutput | None = None
