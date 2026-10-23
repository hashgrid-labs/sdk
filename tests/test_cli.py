import subprocess
import sys

import hashgrid


def test_version():
    out = subprocess.run(
        [sys.executable, "-c", "from hashgrid.cli import main; main(['--version'])"],
        capture_output=True, text=True,
    )
    assert out.stdout.strip() == f"hashgrid {hashgrid.__version__}" == "hashgrid 1.3.0"


def test_version_matches_the_package():
    import re
    from pathlib import Path

    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert re.search(r'^version = "([^"]+)"', pyproject, re.M)[1] == hashgrid.__version__
