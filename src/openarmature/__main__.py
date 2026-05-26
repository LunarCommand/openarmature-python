"""Allow ``python -m openarmature`` to invoke the CLI.

Provides a path-independent way to reach :func:`openarmature.cli.main`
in environments where the ``[project.scripts]`` entry point doesn't
land cleanly — some ``pip install --target`` layouts, path-shadowed
venvs, etc. As long as ``import openarmature`` works,
``python -m openarmature`` works too.
"""

from __future__ import annotations

import sys

from openarmature.cli import main

if __name__ == "__main__":
    sys.exit(main())
