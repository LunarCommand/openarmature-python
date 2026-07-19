# Copilot instructions for openarmature-python

This is the reference Python implementation of OpenArmature, a workflow
framework for LLM pipelines and tool-calling agents. Behavior is defined by
the language-agnostic specification in `openarmature-spec` and verified by
conformance fixtures under `tests/conformance/`; the spec and its fixtures are
the source of truth, not local style preference.

## Conventions to respect when reviewing

- **User-facing copy uses no em dashes.** `README.md`, `docs/**`, PR
  descriptions, and `examples/**` avoid the em dash (`U+2014`) as an
  LLM-output tell; use commas, colons, parentheses, or sentence splits instead. `src/`
  code comments are exempt. Do not suggest adding an em dash. New `CHANGELOG.md`
  entries also avoid them (older entries are intentionally left as-is).

- **`examples/**` are teaching artifacts, not production code.** They
  deliberately use the codebase's house idioms, construct providers lazily
  (so importing the module opens no network client), close providers in a
  `finally`, carry moon-themed subject matter, and reference no spec sections.
  Each exposes a `build_graph()` factory (a smoke test depends on it). Prefer
  consistency with the sibling examples over generic style preferences.

- **Documentation code snippets are illustrative** unless they live under
  `docs/getting-started/`, `docs/model-providers/`, or
  `docs/retrieval-providers/` (those are executed for drift). Concept-page
  snippets reference names defined out of band and are not standalone programs;
  do not flag them for missing imports or incompleteness.

- **Typed default factories are intentional.** On a Pydantic `State`, a
  `list[int]` field uses `Field(default_factory=list[int])`, not a bare
  `default_factory=list`: strict pyright flags the bare `list` factory as
  `list[Unknown]` on a `list[int]` field (a `list[str]` field happens to pass).
  Both the typed factory and a bare `[]` default are safe here because Pydantic
  copies the default per instance.

- **`usage = None` is the contract, not a bug.** When a provider reports no
  token usage, `EmbeddingResponse.usage` / `RerankResponse.usage` and the
  corresponding event `usage` are `None`. A mapping must never fabricate a
  usage record, a zero, or a client-side estimate; a malformed usage figure is
  treated as not-reported, not as an error.

- **`input_type` is a wire no-op on symmetric embedding providers** (for
  example the OpenAI mapping). Setting `input_type="query"` / `"document"`
  keeps a pipeline portable to asymmetric providers (TEI, Cohere, Jina); it is
  not dead code.

## Toolchain

- Python `>=3.12`, strict pyright, and ruff (line length 110). A suggestion
  that would break strict typing or ruff formatting is not useful; the CI gate
  runs all three.

- Provider error handling maps to the canonical `Provider*` categories from
  `openarmature.llm.errors` (authentication, rate-limit, invalid-model,
  invalid-request, unavailable, invalid-response). A non-200 response is
  classified by status; a 200 with a malformed body maps to
  `ProviderInvalidResponse`.

## Generated files (do not review as hand-written)

- `src/openarmature/AGENTS.md` and `src/openarmature/_patterns/` are generated
  by `scripts/build_agents_md.py`; edit the generator, not the output.
