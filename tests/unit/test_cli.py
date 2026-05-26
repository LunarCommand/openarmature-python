"""Unit tests for the ``openarmature`` CLI.

Covers the two-subcommand surface (``init`` and ``docs``) via the
in-process :func:`openarmature.cli.main` entry point — same surface
the ``[project.scripts]`` shim and ``python -m openarmature`` dispatch
to, so tests don't need to spawn a subprocess.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openarmature.cli import INIT_MARKER, main


def test_docs_prints_bundled_agents_md_path(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["docs"])
    captured = capsys.readouterr()
    assert code == 0
    printed = Path(captured.out.strip())
    # The bundled file ships at the installed package root.
    assert printed.name == "AGENTS.md"
    assert printed.is_file()


def test_init_creates_files_when_absent(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["init", "--cwd", str(tmp_path)])
    assert code == 0

    agents = tmp_path / "AGENTS.md"
    claude = tmp_path / "CLAUDE.md"
    assert agents.is_file()
    assert claude.is_file()
    # Both have the marker so re-run detects them.
    assert INIT_MARKER in agents.read_text()
    assert INIT_MARKER in claude.read_text()
    # Both start with the ``## OpenArmature`` section (no leading
    # blank lines on a freshly-created file).
    assert agents.read_text().startswith("## OpenArmature")

    out = capsys.readouterr().out
    assert "create:" in out
    # Reports both files.
    assert "AGENTS.md" in out
    assert "CLAUDE.md" in out


def test_init_appends_to_existing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    existing = "# Project AGENTS.md\n\nSome existing notes.\n"
    (tmp_path / "AGENTS.md").write_text(existing)

    code = main(["init", "--cwd", str(tmp_path)])
    assert code == 0

    result = (tmp_path / "AGENTS.md").read_text()
    # Existing content is preserved.
    assert result.startswith("# Project AGENTS.md")
    assert "Some existing notes." in result
    # And the new section is appended.
    assert "## OpenArmature" in result
    assert INIT_MARKER in result
    # A blank line separates the original content from the new
    # section (no jammed-up "notes.## OpenArmature").
    assert "notes.\n\n## OpenArmature" in result

    out = capsys.readouterr().out
    assert "append:" in out


def test_init_is_idempotent_via_marker(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # First run creates.
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()  # Discard first-run output.

    first_content = (tmp_path / "AGENTS.md").read_text()
    code = main(["init", "--cwd", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "AGENTS.md").read_text() == first_content, (
        "init should be a no-op when the marker is already present"
    )

    out = capsys.readouterr().out
    assert "skip:" in out
    assert INIT_MARKER in out


def test_init_force_reappends_block(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["init", "--cwd", str(tmp_path)])
    capsys.readouterr()  # Discard first-run output.

    initial = (tmp_path / "AGENTS.md").read_text()
    code = main(["init", "--cwd", str(tmp_path), "--force"])
    assert code == 0

    forced = (tmp_path / "AGENTS.md").read_text()
    # ``--force`` re-appends, so the file grew.
    assert len(forced) > len(initial)
    # And there are now two ``## OpenArmature`` headings.
    assert forced.count("## OpenArmature") == 2
    assert forced.count(INIT_MARKER) == 2

    out = capsys.readouterr().out
    assert "force-append:" in out


def test_init_dry_run_does_not_modify_disk(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["init", "--cwd", str(tmp_path), "--dry-run"])
    assert code == 0

    # No files created.
    assert not (tmp_path / "AGENTS.md").exists()
    assert not (tmp_path / "CLAUDE.md").exists()

    out = capsys.readouterr().out
    # Output is prefixed so the user can tell it's a preview.
    assert "[dry-run]" in out
    assert "create:" in out


def test_init_rejects_nonexistent_cwd(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "does-not-exist"
    code = main(["init", "--cwd", str(missing)])
    assert code == 2
    err = capsys.readouterr().err
    assert "not a directory" in err


def test_init_dry_run_on_existing_file_reports_append(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "AGENTS.md").write_text("# Existing\n")
    code = main(["init", "--cwd", str(tmp_path), "--dry-run"])
    assert code == 0

    # File unchanged.
    assert (tmp_path / "AGENTS.md").read_text() == "# Existing\n"

    out = capsys.readouterr().out
    assert "[dry-run] append:" in out
    # CLAUDE.md doesn't exist, so dry-run reports a create for it.
    assert "[dry-run] create:" in out
