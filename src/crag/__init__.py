"""Corrective RAG (CRAG) — independent reimplementation of Yan et al., 2024 (arXiv:2401.15884)."""

from crag.pipeline import CRAGPipeline, PlainRAGPipeline, build_pipelines

__all__ = ["CRAGPipeline", "PlainRAGPipeline", "build_pipelines"]
