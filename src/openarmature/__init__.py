"""OpenArmature: workflow framework for LLM pipelines and tool-calling agents.

AI agents: three discovery surfaces are available, pick whichever
your environment can reach:

1. **Bundled reference** at ``openarmature/AGENTS.md`` — capability
   contracts, common patterns, non-obvious shapes, and an example
   index. Path resolves via::

       python -c "import openarmature; print(openarmature.__path__[0] + '/AGENTS.md')"

   Or via the CLI: ``openarmature docs`` prints the same path.

2. **Programmatic patterns catalog** at ``openarmature.patterns`` —
   ``list()`` returns the available pattern names; ``get(name)``
   returns the canonical recipe as a markdown string. Useful in
   sandboxed environments that can ``import openarmature`` but
   can't freely read arbitrary package paths.

3. **CLI** registered as ``openarmature`` (and reachable as
   ``python -m openarmature`` where script entry points don't land
   cleanly). ``openarmature init`` writes a discovery pointer block
   into the project's ``AGENTS.md`` / ``CLAUDE.md`` so future agent
   sessions opening the project find the bundled docs automatically.
"""

__version__ = "0.10.0"
__spec_version__ = "0.27.1"
