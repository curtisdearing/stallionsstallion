"""Regression coverage for the native launchd runner bootstrap."""

from pathlib import Path


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
