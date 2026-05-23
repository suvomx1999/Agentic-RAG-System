"""Tests for grader tool — uses mocked LLM."""

import json
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.documents import Document

from agentic_rag.tools.grader import grade_documents, _parse_grade


def _make_docs(n: int) -> list:
    return [
        Document(
            page_content=f"Document {i} about machine learning.",
            metadata={"source": f"test_{i}.txt", "chunk_index": i}
        )
        for i in range(n)
    ]


class TestParseGrade:
    def test_valid_json(self):
        result = _parse_grade('{"relevant": true, "reason": "good match"}')
        assert result["relevant"] is True
        assert result["reason"] == "good match"

    def test_valid_json_false(self):
        result = _parse_grade('{"relevant": false, "reason": "off topic"}')
        assert result["relevant"] is False

    def test_malformed_json_fallback(self):
        result = _parse_grade("This is not JSON at all")
        assert result["relevant"] is True  # Fallback
        assert "fallback" in result["reason"]

    def test_markdown_code_block(self):
        result = _parse_grade('```json\n{"relevant": true, "reason": "test"}\n```')
        assert result["relevant"] is True

    def test_empty_string_fallback(self):
        result = _parse_grade("")
        assert result["relevant"] is True


class TestGradeDocuments:
    @patch("agentic_rag.tools.grader._get_llm")
    def test_partitions_correctly(self, mock_get_llm):
        # Mock LLM: first doc relevant, second irrelevant
        mock_llm = MagicMock()
        responses = [
            MagicMock(content='{"relevant": true, "reason": "good"}'),
            MagicMock(content='{"relevant": false, "reason": "bad"}'),
            MagicMock(content='{"relevant": true, "reason": "ok"}'),
        ]
        mock_llm.invoke = MagicMock(side_effect=responses)
        mock_get_llm.return_value = mock_llm

        docs = _make_docs(3)
        relevant, irrelevant = grade_documents("test query", docs)

        assert len(relevant) == 2
        assert len(irrelevant) == 1

    @patch("agentic_rag.tools.grader._get_llm")
    def test_metadata_present(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(
            return_value=MagicMock(content='{"relevant": true, "reason": "match"}')
        )
        mock_get_llm.return_value = mock_llm

        docs = _make_docs(1)
        relevant, _ = grade_documents("test", docs)

        assert "grade_relevant" in relevant[0].metadata
        assert "grade_reason" in relevant[0].metadata

    @patch("agentic_rag.tools.grader._get_llm")
    def test_handles_malformed_llm_output(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(
            return_value=MagicMock(content="Sorry, I cannot help")
        )
        mock_get_llm.return_value = mock_llm

        docs = _make_docs(1)
        relevant, irrelevant = grade_documents("test", docs)

        # Should fallback to relevant=True
        assert len(relevant) == 1
        assert len(irrelevant) == 0
