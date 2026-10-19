import subprocess
import sys

import hashgrid


def test_version():
    out = subprocess.run(
        [sys.executable, "-c", "from hashgrid.cli import main; main(['--version'])"],
        capture_output=True, text=True,
    )
    assert out.stdout.strip() == f"hashgrid {hashgrid.__version__}" == "hashgrid 1.2.0"
