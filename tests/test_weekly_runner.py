"""Regression coverage for the native launchd runner bootstrap."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_weekly.sh"
WEEKLY_REQUIREMENTS = ROOT / "requirements-weekly.txt"


def test_runner_requires_python_311_or_newer():
    source = RUNNER.read_text()

    assert "python3.11" in source
    assert "STALLIONS_PYTHON" in source
    assert "requires Python 3.11+" in source
    assert '"$VENV/bin/python3" -c' in source
    assert 'rm -rf "$VENV"' in source


def test_weekly_dependencies_match_the_runner_import_surface():
    requirements = [
        line for line in WEEKLY_REQUIREMENTS.read_text().splitlines()
        if line and not line.startswith("#")
    ]

    assert requirements == [
        "pandas>=1.5",
        "pyarrow>=10.0",
        "numpy>=1.23",
    ]
    source = RUNNER.read_text()
    assert "import pandas, pyarrow, numpy" in source
    assert "scipy" not in source
    assert "scikit-learn" not in requirements
    assert "joblib" not in requirements


def test_runner_installs_weekly_specific_dependencies():
    source = RUNNER.read_text()

    assert '"$REPO/requirements-weekly.txt"' in source


# --------------------------------------------------------------------------- #
# Executed, not read.  The assertions above prove the lines exist; a refactor
# can keep every one of those strings and still spend the API budget on Python
# 3.9.  These run the script.  Both cases exit on the version gate, so neither
# builds a venv and neither needs the network.
# --------------------------------------------------------------------------- #
def _stage(tmp_path):
    """A minimal repo the runner can be executed against."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(RUNNER, repo / "scripts" / "run_weekly.sh")
    os.chmod(repo / "scripts" / "run_weekly.sh", 0o755)
    shutil.copy(WEEKLY_REQUIREMENTS, repo / "requirements-weekly.txt")
    # Tripwire: if the runner ever reaches the loop it leaves this behind.
    (repo / "scripts" / "weekly_loop.py").write_text(
        "import pathlib\n"
        "pathlib.Path(__file__).with_name('LOOP_RAN').write_text('x')\n")
    return repo


def _fake_interpreter(path, *, ok):
    """Stand-in for `python3`: the `-c` version probe exits 1 when too old."""
    path.write_text("#!/bin/bash\n"
                    "if [ \"${1:-}\" = '-c' ]; then exit %d; fi\nexit 0\n"
                    % (0 if ok else 1))
    os.chmod(path, 0o755)
    return path


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX runner")
def test_runner_exits_before_any_api_work_on_python_below_311(tmp_path):
    repo = _stage(tmp_path)
    old = _fake_interpreter(tmp_path / "python3.9", ok=False)

    proc = subprocess.run(
        ["bash", str(repo / "scripts" / "run_weekly.sh"), "pregame"],
        cwd=repo, capture_output=True, text=True,
        env={**os.environ, "STALLIONS_PYTHON": str(old)}, timeout=120)

    assert proc.returncode == 1
    assert "requires Python 3.11+" in proc.stdout + proc.stderr
    assert not (repo / ".venv").exists(), \
        "the version gate must refuse before the venv is built"
    assert not (repo / "scripts" / "LOOP_RAN").exists(), \
        "no CFBD-capable code may run on an unsupported interpreter"


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX runner")
def test_runner_exits_when_no_python_311_can_be_found(tmp_path):
    repo = _stage(tmp_path)
    binhome = tmp_path / "bin"
    binhome.mkdir()
    for tool in ("bash", "date", "mkdir", "tee", "ls", "tail", "rm", "cat",
                 "dirname", "sed", "grep"):
        found = shutil.which(tool)
        if found:
            os.symlink(found, binhome / tool)

    proc = subprocess.run(
        ["bash", str(repo / "scripts" / "run_weekly.sh"), "pregame"],
        cwd=repo, capture_output=True, text=True,
        env={"PATH": str(binhome), "HOME": str(tmp_path),
             "STALLIONS_PYTHON": str(tmp_path / "absent-python3.11")},
        timeout=120)

    assert proc.returncode == 1
    assert "requires Python 3.11+" in proc.stdout + proc.stderr
    assert not (repo / ".venv").exists()
    assert not (repo / "scripts" / "LOOP_RAN").exists()
