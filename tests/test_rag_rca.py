"""Knowledge base, retrieval RCA and the Claude agent (with a fake client, no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from aiops_copilot.core.config import Settings
from aiops_copilot.detection.grouping import DetectedIncident
from aiops_copilot.llm.evidence import FrameEvidence
from aiops_copilot.llm.rca import ClaudeRCA, RCAResult, RetrievalRCA, make_rca_engine
from aiops_copilot.rag.knowledge_base import KnowledgeBase, log_template
from conftest import Trained


def _evidence(trained: Trained) -> FrameEvidence:
    logs = [{k: r[k] for k in ("timestamp", "host", "source", "level", "message")} for r in trained.test_logs]
    return FrameEvidence(trained.test, logs, trained.detector.scaler)


def _matched(trained: Trained) -> list[tuple[DetectedIncident, dict[str, Any]]]:
    """Detected incidents paired with the ground-truth incident they overlap."""
    result = trained.detector.detect(trained.test, trained.test_logs)
    pairs = []
    for inc in result.incidents:
        for truth in trained.test_incidents:
            if truth["host"] == inc.host and pd.Timestamp(truth["start"]) <= inc.end \
                    and inc.start <= pd.Timestamp(truth["end"]):
                pairs.append((inc, truth))
                break
    return pairs


def test_log_template_masks_variable_parts() -> None:
    a = log_template("Failed password for root from 203.0.113.7 port 51234 ssh2")
    b = log_template("Failed password for root from 198.51.100.9 port 40000 ssh2")
    assert a == b
    assert "203.0.113.7" not in a


def test_search_returns_relevant_reports(knowledge_base: KnowledgeBase) -> None:
    hits = knowledge_base.search("failed SSH logins far above normal. Failed password for invalid user", 5,
                                 doc_type="incident_report")
    assert hits
    assert hits[0].metadata["incident_type"] == "ssh_brute_force"


def test_retrieval_rca_identifies_incident_types(trained: Trained, knowledge_base: KnowledgeBase) -> None:
    engine = RetrievalRCA(knowledge_base)
    evidence = _evidence(trained)
    pairs = _matched(trained)
    assert len(pairs) >= 5
    correct = 0
    for inc, truth in pairs:
        out = engine.analyze(inc, evidence)
        assert isinstance(out, RCAResult) and out.summary and out.suggested_fix
        assert 0.0 <= out.confidence <= 1.0
        correct += out.root_cause_id.split("/")[0] == truth["incident_type"]
    assert correct / len(pairs) >= 0.7


def test_no_key_means_offline_engine(kb_settings: Settings, knowledge_base: KnowledgeBase) -> None:
    assert not kb_settings.llm_enabled
    assert isinstance(make_rca_engine(kb_settings, knowledge_base), RetrievalRCA)


# ------------------------------------------------------------------ Claude agent with a fake client
def _block(kind: str, **kw: Any) -> SimpleNamespace:
    return SimpleNamespace(type=kind, **kw)


class FakeClient:
    """Mimics ``client.beta.messages.create`` with scripted responses."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(json.loads(json.dumps(kwargs, default=str)))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


GOOD_ANSWER = {
    "summary": "SSH brute force from one external IP.", "root_cause": "Password auth exposed to the internet.",
    "root_cause_id": "ssh_brute_force/exposed_ssh_password_auth", "suggested_fix": "Disable password auth.",
    "confidence": 0.9, "severity": "high", "evidence": ["BRUTE-001 fired"], "similar_incident_ids": ["x"],
}


@pytest.fixture()
def incident_and_evidence(trained: Trained) -> tuple[DetectedIncident, FrameEvidence]:
    inc, _ = _matched(trained)[0]
    return inc, _evidence(trained)


def _llm_settings(kb_settings: Settings) -> Settings:
    return kb_settings.model_copy(update={"llm_provider": "anthropic", "llm_max_tool_rounds": 3})


def test_agent_uses_tools_then_returns_structured_answer(
        kb_settings: Settings, knowledge_base: KnowledgeBase,
        incident_and_evidence: tuple[DetectedIncident, FrameEvidence]) -> None:
    inc, evidence = incident_and_evidence
    client = FakeClient([
        SimpleNamespace(stop_reason="tool_use", content=[
            _block("tool_use", id="t1", name="search_knowledge_base", input={"query": "ssh brute force"}),
            _block("tool_use", id="t2", name="get_logs", input={
                "host": inc.host, "start": inc.start.isoformat(), "end": inc.end.isoformat()}),
        ]),
        SimpleNamespace(stop_reason="end_turn", content=[_block("text", text=json.dumps(GOOD_ANSWER))]),
    ])
    out = ClaudeRCA(_llm_settings(kb_settings), knowledge_base, client=client).analyze(inc, evidence)
    assert out.engine.startswith("llm:")
    assert out.root_cause_id == GOOD_ANSWER["root_cause_id"]
    # Second call carries both tool results back to the model.
    results = client.calls[1]["messages"][-1]["content"]
    assert {r["tool_use_id"] for r in results} == {"t1", "t2"}
    assert all(not r["is_error"] for r in results)
    assert client.calls[0]["output_config"]["format"]["type"] == "json_schema"


def test_bad_tool_arguments_are_reported_not_raised(
        kb_settings: Settings, knowledge_base: KnowledgeBase,
        incident_and_evidence: tuple[DetectedIncident, FrameEvidence]) -> None:
    inc, evidence = incident_and_evidence
    client = FakeClient([
        SimpleNamespace(stop_reason="tool_use", content=[
            _block("tool_use", id="t1", name="get_logs", input={"host": inc.host, "start": "not-a-date",
                                                                "end": "x"})]),
        SimpleNamespace(stop_reason="end_turn", content=[_block("text", text=json.dumps(GOOD_ANSWER))]),
    ])
    out = ClaudeRCA(_llm_settings(kb_settings), knowledge_base, client=client).analyze(inc, evidence)
    assert out.engine.startswith("llm:")
    assert client.calls[1]["messages"][-1]["content"][0]["is_error"] is True


@pytest.mark.parametrize("responses", [
    [RuntimeError("API down")],
    [SimpleNamespace(stop_reason="refusal", content=[])],
    [SimpleNamespace(stop_reason="end_turn", content=[_block("text", text="not json")])],
    [SimpleNamespace(stop_reason="end_turn", content=[_block("text", text=json.dumps({"summary": "x"}))])],
], ids=["api-error", "refusal", "invalid-json", "schema-violation"])
def test_failures_fall_back_to_retrieval(
        responses: list[Any], kb_settings: Settings, knowledge_base: KnowledgeBase,
        incident_and_evidence: tuple[DetectedIncident, FrameEvidence]) -> None:
    inc, evidence = incident_and_evidence
    out = ClaudeRCA(_llm_settings(kb_settings), knowledge_base, client=FakeClient(responses)).analyze(inc, evidence)
    assert out.engine == "retrieval (llm fallback)"
    assert out.summary


def test_endless_tool_loop_is_cut_off(
        kb_settings: Settings, knowledge_base: KnowledgeBase,
        incident_and_evidence: tuple[DetectedIncident, FrameEvidence]) -> None:
    inc, evidence = incident_and_evidence
    loop = [SimpleNamespace(stop_reason="tool_use", content=[
        _block("tool_use", id=f"t{i}", name="search_knowledge_base", input={"query": "x"})]) for i in range(10)]
    client = FakeClient(loop)
    out = ClaudeRCA(_llm_settings(kb_settings), knowledge_base, client=client).analyze(inc, evidence)
    assert out.engine == "retrieval (llm fallback)"
    assert len(client.calls) == 4  # max_tool_rounds (3) + 1
