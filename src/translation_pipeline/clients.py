"""Factories for the three external API clients.

Stages take their client through dependency injection so tests can inject fakes
and so client configuration is centralized here.
"""

from __future__ import annotations

import anthropic
import deepl
from openai import OpenAI

from translation_pipeline.config import Settings


def anthropic_client(settings: Settings) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY.get_secret_value())


def openai_client(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.OPENAI_API_KEY.get_secret_value())


def deepl_client(settings: Settings) -> deepl.Translator:
    return deepl.Translator(settings.DEEPL_API_KEY.get_secret_value())
