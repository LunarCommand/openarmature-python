"""Run every spec retrieval-provider conformance fixture against the reference providers.

The fixtures (``spec/retrieval-provider/conformance/``) describe an
``EmbeddingProvider`` / ``RerankProvider``'s behavior as mock responses +
expected ``embed()`` / ``rerank()`` outcomes. Each fixture wraps the call in a
single-node graph, but that node is just one provider call, so the harness
extracts the ``calls_embed`` / ``calls_rerank`` directive + ``mock_embedding``
/ ``mock_rerank`` and drives the real :class:`OpenAIEmbeddingProvider` /
:class:`CohereRerankProvider` through ``httpx.MockTransport`` directly,
mirroring ``test_llm_provider`` (no graph engine; the observed-event fixtures
cover the observer path).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import yaml

from openarmature.llm import LlmProviderError
from openarmature.retrieval import (
    CohereRerankProvider,
    EmbeddingRuntimeConfig,
    OpenAIEmbeddingProvider,
    RerankRuntimeConfig,
)

from ._deferral import skip_if_deferred

CONFORMANCE_DIR = (
    Path(__file__).resolve().parents[2] / "openarmature-spec" / "spec" / "retrieval-provider" / "conformance"
)

# Default bound model when a fixture's calls_embed omits one; matches the
# fixtures' mock-body model.
_DEFAULT_MODEL = "text-embedding-test"

# Default bound rerank model when a fixture's calls_rerank omits one; matches
# the rerank fixtures' mock-body model.
_DEFAULT_RERANK_MODEL = "rerank-test"

# The rerank protocol fixtures (006-012) run against the reference
# CohereRerankProvider (proposal 0060) as of v0.16.0.
#
# The v0.84.0 pin bump also introduced the retrieval-provider WIRE-MAPPING
# fixtures (013-027) for proposals 0077 (TEI) / 0078 (Jina) / 0079
# (OpenAI-compatible), still present at the current pin, none of which python
# has shipped. This runner drives only the bundled
# OpenAIEmbeddingProvider's embed() and asserts the response, so it cannot
# exercise their request-side mappings (TEI server-side prompt_name, Jina's
# task, the client-side query/document prefix) -- the contracts those
# fixtures exist to verify. Defer the whole batch until each wire-mapping
# impl PR (v0.16.0+).
#
# The v0.88.0 pin pulls in the Cohere wire mappings + the general embed
# batch-chunking rule: 028-031 (0090 Cohere rerank), 032-037 (0091 Cohere
# embed), 038 (0092 TEI /embed over-cap chunk-and-stitch). None are shipped
# -- same deferral rationale as 013-027. (0060a ships the RerankProvider
# protocol + the Cohere-shape reference reranker for the protocol fixtures
# 006-012; the 0090 Cohere wire-mapping fixtures 028-031 assert the request-
# side wire contract this runner cannot capture, so they stay deferred.)
_DEFERRED_FIXTURES: dict[str, str] = {
    **{
        p.stem: "TEI wire mapping (proposal 0077) not implemented"
        for p in CONFORMANCE_DIR.glob("[0-9][0-9][0-9]-*.yaml")
        if 13 <= int(p.stem[:3]) <= 17
    },
    **{
        p.stem: "Jina wire mapping (proposal 0078) not implemented"
        for p in CONFORMANCE_DIR.glob("[0-9][0-9][0-9]-*.yaml")
        if 18 <= int(p.stem[:3]) <= 22
    },
    **{
        p.stem: (
            "OpenAI-compatible embed wire ships via the bundled OpenAIEmbeddingProvider "
            "(proposal 0059); deferred because the harness lacks a wire-capture primitive "
            "(expected_wire_request / url / headers) and 0079's dimensions / input_type "
            "request knobs are unimplemented"
        )
        for p in CONFORMANCE_DIR.glob("[0-9][0-9][0-9]-*.yaml")
        if 23 <= int(p.stem[:3]) <= 27
    },
    **{
        p.stem: "Cohere rerank wire mapping (proposal 0090) not implemented"
        for p in CONFORMANCE_DIR.glob("[0-9][0-9][0-9]-*.yaml")
        if 28 <= int(p.stem[:3]) <= 31
    },
    **{
        p.stem: "Cohere embed wire mapping (proposal 0091) not implemented"
        for p in CONFORMANCE_DIR.glob("[0-9][0-9][0-9]-*.yaml")
        if 32 <= int(p.stem[:3]) <= 37
    },
    **{
        p.stem: (
            "embedding batch-chunking general rule (proposal 0092); the TEI "
            "/embed over-cap chunk-and-stitch fixture rides the unshipped TEI "
            "embed wire mapping (proposal 0077)"
        )
        for p in CONFORMANCE_DIR.glob("[0-9][0-9][0-9]-*.yaml")
        if int(p.stem[:3]) == 38
    },
}


def _fixture_paths() -> list[Path]:
    return sorted(CONFORMANCE_DIR.glob("[0-9][0-9][0-9]-*.yaml"))


def _fixture_id(path: Path) -> str:
    return path.stem


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return cast("dict[str, Any]", yaml.safe_load(f))


def _build_handler(
    responses: list[Mapping[str, Any]],
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """MockTransport handing back the configured responses in arrival order."""
    captured: list[httpx.Request] = []
    iterator = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        try:
            spec = next(iterator)
        except StopIteration as exc:
            raise AssertionError(f"mock provider exhausted at request {len(captured)}") from exc
        status = int(spec.get("status", 200))
        body = spec.get("body")
        if body is None:
            return httpx.Response(status)
        return httpx.Response(status, content=json.dumps(body).encode("utf-8"))

    return httpx.MockTransport(handler), captured


def _build_provider(
    responses: list[Mapping[str, Any]],
    *,
    model: str,
) -> tuple[OpenAIEmbeddingProvider, list[httpx.Request]]:
    transport, captured = _build_handler(responses)
    provider = OpenAIEmbeddingProvider(
        base_url="http://mock-embed.test",
        model=model,
        api_key="test-key",
        transport=transport,
    )
    return provider, captured


def _build_config(config_block: Mapping[str, Any] | None) -> EmbeddingRuntimeConfig | None:
    if not config_block:
        return None
    block = dict(config_block)
    extras = cast("Mapping[str, Any] | None", block.pop("extras", None))
    if extras:
        block.update(extras)
    return EmbeddingRuntimeConfig(**block)


def _build_rerank_provider(
    responses: list[Mapping[str, Any]],
    *,
    model: str,
) -> tuple[CohereRerankProvider, list[httpx.Request]]:
    transport, captured = _build_handler(responses)
    provider = CohereRerankProvider(
        base_url="http://mock-rerank.test",
        model=model,
        api_key="test-key",
        transport=transport,
    )
    return provider, captured


def _build_rerank_config(config_block: Mapping[str, Any] | None) -> RerankRuntimeConfig | None:
    if not config_block:
        return None
    block = dict(config_block)
    extras = cast("Mapping[str, Any] | None", block.pop("extras", None))
    if extras:
        block.update(extras)
    return RerankRuntimeConfig(**block)


def _assert_embedding_response(
    response: Any,
    expected: Mapping[str, Any],
) -> None:
    """Assert the response against the expected.final_state.embedding_response
    block. The block uses assertion keys (vectors_length / dimensions / ...)
    and/or a literal ``vectors`` list."""
    for key, val in expected.items():
        if key == "vectors":
            assert response.vectors == [[float(x) for x in v] for v in val], (
                f"vectors mismatch: {response.vectors} != {val}"
            )
        elif key == "vectors_length":
            assert len(response.vectors) == val
        elif key == "dimensions":
            assert response.dimensions == val
        elif key == "inner_vector_lengths_all_equal":
            assert all(len(v) == val for v in response.vectors)
        elif key == "model":
            assert response.model == val
        elif key == "response_id":
            assert response.response_id == val
        elif key == "usage":
            assert response.usage.input_tokens == cast("Mapping[str, Any]", val)["input_tokens"]
        # Unknown keys are skipped (tolerant, matching the llm-provider
        # harness): a new spec assertion key is a no-op here until wired,
        # rather than breaking the whole parametrized suite.


def _check_success_invariants(
    response: Any,
    input_strings: list[str],
    invariants: Mapping[str, Any],
) -> None:
    for key, val in invariants.items():
        if not val:
            continue
        if key == "vectors_length_matches_input_length":
            assert len(response.vectors) == len(input_strings)
        elif key == "all_vectors_same_dimensionality":
            assert len({len(v) for v in response.vectors}) <= 1
        elif key == "dimensions_field_matches_inner_vector_length":
            assert not response.vectors or response.dimensions == len(response.vectors[0])
        elif key.startswith("vector_at_index_"):
            # Order assertions; covered by the literal final_state.vectors check.
            continue
        # Unknown invariants are skipped (tolerant); the structural invariants
        # above plus the baseline count assertion cover the core contract.


def _assert_rerank_response(
    response: Any,
    expected: Mapping[str, Any],
) -> None:
    """Assert the rerank response against the expected.final_state.<stored>
    block: the sorted results (index / relevance_score / document), model,
    response_id, and usage (search_units / input_tokens, null-aware)."""
    for key, val in expected.items():
        if key == "results":
            results = cast("list[Mapping[str, Any]]", val)
            assert len(response.results) == len(results), (
                f"result count mismatch: {len(response.results)} != {len(results)}"
            )
            for got, want in zip(response.results, results, strict=True):
                assert got.index == want["index"], f"index mismatch: {got.index} != {want['index']}"
                assert got.relevance_score == want["relevance_score"], (
                    f"relevance_score mismatch: {got.relevance_score} != {want['relevance_score']}"
                )
                # ``document`` is only asserted when the expected entry carries
                # the key (null included -- the omitted-echo case asserts None).
                if "document" in want:
                    assert got.document == want["document"], (
                        f"document mismatch: {got.document!r} != {want['document']!r}"
                    )
        elif key == "model":
            assert response.model == val
        elif key == "response_id":
            assert response.response_id == val
        elif key == "usage":
            if val is None:
                assert response.usage is None
            else:
                usage = cast("Mapping[str, Any]", val)
                assert response.usage is not None, "expected a usage record, got None"
                if "search_units" in usage:
                    assert response.usage.search_units == usage["search_units"]
                if "input_tokens" in usage:
                    assert response.usage.input_tokens == usage["input_tokens"]
        # Unknown keys are skipped (tolerant), matching the embedding harness.


def _check_rerank_success_invariants(
    response: Any,
    documents: list[str],
    invariants: Mapping[str, Any],
) -> None:
    for key, val in invariants.items():
        if not val:
            continue
        if key == "results_sorted_by_relevance_descending":
            scores = [r.relevance_score for r in response.results]
            assert scores == sorted(scores, reverse=True), f"results not sorted descending: {scores}"
        elif key == "each_index_valid_into_input_documents":
            assert all(0 <= r.index < len(documents) for r in response.results)
        elif key == "result_count_at_most_document_count":
            assert len(response.results) <= len(documents)
        # Unknown invariants are skipped (tolerant); the structural invariants
        # above cover the core §6 contract.


@pytest.mark.parametrize("fixture_path", _fixture_paths(), ids=_fixture_id)
async def test_retrieval_provider_fixture(fixture_path: Path) -> None:
    fixture_id = fixture_path.stem
    skip_if_deferred(fixture_id, _DEFERRED_FIXTURES)
    spec = _load(fixture_path)
    for case in cast("list[dict[str, Any]]", spec.get("cases", [])):
        case_name = case.get("name", "<unnamed>")
        try:
            await _run_one_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_one_case(case: Mapping[str, Any]) -> None:
    entry = cast("str", case["entry"])
    node = cast("Mapping[str, Any]", cast("Mapping[str, Any]", case["nodes"])[entry])
    if "calls_rerank" in node:
        await _run_rerank_case(case, cast("Mapping[str, Any]", node["calls_rerank"]))
        return
    calls_embed = cast("Mapping[str, Any]", node["calls_embed"])
    input_strings = cast("list[str]", calls_embed["input"])
    model = cast("str", calls_embed.get("model", _DEFAULT_MODEL))
    config = _build_config(cast("Mapping[str, Any] | None", calls_embed.get("config")))
    responses = cast("list[Mapping[str, Any]]", case.get("mock_embedding") or [])
    provider, _captured = _build_provider(responses, model=model)
    expected_error = cast("Mapping[str, Any] | None", case.get("expected_error"))
    expected = cast("Mapping[str, Any]", case.get("expected") or {})
    invariants = cast("Mapping[str, Any]", expected.get("invariants") or {})
    try:
        if expected_error is not None:
            with pytest.raises(LlmProviderError) as excinfo:
                await provider.embed(input_strings, config=config)
            assert excinfo.value.category == expected_error["category"], (
                f"expected {expected_error['category']}, got {excinfo.value.category}"
            )
        else:
            response = await provider.embed(input_strings, config=config)
            # Baseline assertion: every success case checks the core
            # one-vector-per-input invariant, so a fixture without a
            # final_state / invariants block still asserts something real.
            assert len(response.vectors) == len(input_strings), (
                f"expected {len(input_strings)} vectors, got {len(response.vectors)}"
            )
            final_state = cast("Mapping[str, Any]", expected.get("final_state") or {})
            stored = cast("str | None", calls_embed.get("stores_response_in"))
            if stored is not None and stored in final_state:
                _assert_embedding_response(response, cast("Mapping[str, Any]", final_state[stored]))
            _check_success_invariants(response, input_strings, invariants)
    finally:
        await provider.aclose()


async def _run_rerank_case(case: Mapping[str, Any], calls_rerank: Mapping[str, Any]) -> None:
    query = cast("str", calls_rerank["query"])
    documents = cast("list[str]", calls_rerank["documents"])
    top_k = cast("int | None", calls_rerank.get("top_k"))
    model = cast("str", calls_rerank.get("model", _DEFAULT_RERANK_MODEL))
    config = _build_rerank_config(cast("Mapping[str, Any] | None", calls_rerank.get("config")))
    responses = cast("list[Mapping[str, Any]]", case.get("mock_rerank") or [])
    provider, _captured = _build_rerank_provider(responses, model=model)
    expected_error = cast("Mapping[str, Any] | None", case.get("expected_error"))
    expected = cast("Mapping[str, Any]", case.get("expected") or {})
    invariants = cast("Mapping[str, Any]", expected.get("invariants") or {})
    try:
        if expected_error is not None:
            with pytest.raises(LlmProviderError) as excinfo:
                await provider.rerank(query, documents, top_k=top_k, config=config)
            assert excinfo.value.category == expected_error["category"], (
                f"expected {expected_error['category']}, got {excinfo.value.category}"
            )
        else:
            response = await provider.rerank(query, documents, top_k=top_k, config=config)
            # Baseline assertion: every success case satisfies the §6
            # at-most-len(documents) bound, so a fixture without a
            # final_state / invariants block still asserts something real.
            assert len(response.results) <= len(documents), (
                f"expected at most {len(documents)} results, got {len(response.results)}"
            )
            final_state = cast("Mapping[str, Any]", expected.get("final_state") or {})
            stored = cast("str | None", calls_rerank.get("stores_response_in"))
            if stored is not None and stored in final_state:
                _assert_rerank_response(response, cast("Mapping[str, Any]", final_state[stored]))
            _check_rerank_success_invariants(response, documents, invariants)
    finally:
        await provider.aclose()
