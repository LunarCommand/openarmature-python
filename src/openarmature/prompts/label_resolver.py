# Spec §7 (new in proposal 0033): LabelResolver primitive that lets a
# PromptManager map prompt names to labels at deployment time, without
# code changes. Three-step resolve precedence is normative; the
# storage shape behind the resolver is impl-defined (mapping, JSON
# file, remote service, env vars).

"""LabelResolver Protocol and reference mapping-backed implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

# Spec §6 step-3 fallback when neither a resolver-supplied per-name
# override nor a resolver-supplied default is available.
SPEC_FALLBACK_LABEL = "production"

# Reserved key in the MappingLabelResolver shape. A `"default"` entry
# in the mapping is the resolver's default-override (step 2 in the
# fallback chain); any other key is a per-name override (step 1).
_DEFAULT_KEY = "default"


@runtime_checkable
class LabelResolver(Protocol):
    """Resolves a prompt name to the label to fetch under.

    Implementations MUST follow the fallback chain in
    :meth:`resolve`: per-name override > default override > the
    ``"production"`` fallback.
    """

    # Spec prompt-management §7: label fallback chain.

    def resolve(self, name: str) -> str:
        """Return the label to fetch ``name`` under.

        Synchronous; deterministic for given resolver state.
        """
        ...


class MappingLabelResolver:
    """Reference resolver backed by a static name → label mapping.

    The mapping recognizes one reserved key, ``"default"``, as the
    resolver's default-override; every other key is a per-name
    override. Construct from a literal dict in code or from a parsed
    JSON file at startup; the resolver is immutable after
    construction.

        >>> r = MappingLabelResolver({"default": "production", "experimental": "staging"})
        >>> r.resolve("experimental")
        'staging'
        >>> r.resolve("anything-else")
        'production'
    """

    def __init__(self, mapping: Mapping[str, str]) -> None:
        self._mapping: dict[str, str] = dict(mapping)

    def resolve(self, name: str) -> str:
        # Step 1: per-name override (any non-`default` key).
        if name in self._mapping and name != _DEFAULT_KEY:
            return self._mapping[name]
        # Step 2: default override (a `default` entry in the mapping).
        if _DEFAULT_KEY in self._mapping:
            return self._mapping[_DEFAULT_KEY]
        # Step 3: spec fallback.
        return SPEC_FALLBACK_LABEL


__all__ = [
    "SPEC_FALLBACK_LABEL",
    "LabelResolver",
    "MappingLabelResolver",
]
