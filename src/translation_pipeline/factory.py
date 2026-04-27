"""Factory functions that build the full real pipeline from Settings."""

from __future__ import annotations

from translation_pipeline.clients import (
    anthropic_client,
    deepl_client,
    openai_client,
)
from translation_pipeline.config import Settings
from translation_pipeline.pipeline import Pipeline
from translation_pipeline.stages.back_translation import BackTranslationStage
from translation_pipeline.stages.base import PipelineStage
from translation_pipeline.stages.brief_extraction import BriefExtractionStage
from translation_pipeline.stages.chunking import ChunkingStage
from translation_pipeline.stages.consistency import ConsistencyStage
from translation_pipeline.stages.critique import CritiqueStage
from translation_pipeline.stages.divergence_detection import DivergenceDetectionStage
from translation_pipeline.stages.draft_a_deepl import DraftAStage
from translation_pipeline.stages.draft_b_claude import DraftBStage
from translation_pipeline.stages.improvement import ImprovementStage
from translation_pipeline.stages.synthesis import SynthesisStage


def build_default_pipeline(settings: Settings) -> Pipeline:
    """Construct the production stage list with real API clients."""
    anthropic = anthropic_client(settings)
    openai = openai_client(settings)
    deepl = deepl_client(settings)

    stages: list[PipelineStage] = [
        BriefExtractionStage(anthropic, settings),
        ChunkingStage(),
        DraftAStage(deepl, settings),
        DraftBStage(anthropic, settings),
        SynthesisStage(anthropic, settings),
        CritiqueStage(openai, settings),
        ImprovementStage(anthropic, settings),
        BackTranslationStage(openai, settings),
        DivergenceDetectionStage(anthropic, settings),
        ConsistencyStage(anthropic, settings),
    ]
    return Pipeline(stages)
