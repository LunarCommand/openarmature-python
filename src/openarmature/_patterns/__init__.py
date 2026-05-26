"""Auto-generated package holding the programmatic patterns API's
transformed markdown payload.

``openarmature.patterns.list()`` / ``get(name)`` resolve the
per-pattern ``<slug>.md`` files in this package via
``importlib.resources``. The files are generated artifacts —
regenerate with ``uv run python scripts/build_agents_md.py``.

Source: ``docs/patterns/*.md`` (excluding ``index.md``) with
the programmatic-API transforms applied — relative
``../concepts/...md`` / ``../examples/...md`` links rewritten
to absolute ``openarmature.ai`` URLs, intra-pattern bare-name
``.md`` links rewritten to absolute
``openarmature.ai/patterns/...`` URLs (see
``_transform_pattern_content_for_programmatic`` in
``scripts/build_agents_md.py``). No heading demotion: each
pattern stands alone when read via the programmatic API.
"""
