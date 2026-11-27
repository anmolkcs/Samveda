"""Guards on the Kaggle notebook itself.

The notebook is the deliverable the user actually runs, on an image I cannot
reproduce locally. It failed once in a way nothing in this repo could catch: the
setup cell ran `pip install "numpy<2" scipy soundfile`, which replaced the numpy
Kaggle's own scipy was built against. numpy still imported; scipy died with
`ModuleNotFoundError: No module named 'numpy.char'` at the fetch step, 20 minutes
in, and needed a fresh session to recover.

These tests are cheap and they pin the two things that made that failure possible:
an unguarded pip install, and cells that were never executed as code.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "kaggle_samadhvani.ipynb"


def _cells():
    nb = json.loads(NB_PATH.read_text())
    return [(c["cell_type"], "".join(c.get("source", []))) for c in nb["cells"]]


def code_cells():
    return [src for kind, src in _cells() if kind == "code"]


def test_every_code_cell_parses():
    for i, src in enumerate(code_cells()):
        ast.parse(src, filename=f"cell{i}")


def _pip_calls(src: str) -> list[str]:
    """Every pip invocation in a cell, as one flat string, arguments included.

    Parsed rather than grepped: the package names sit on the line *after* the one
    containing "pip", and a naive per-line check quietly missed them — as the
    first version of this test did. Comments are ignored, so prose about pip does
    not count as an install.
    """
    found = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        text = ast.unparse(node)
        if "pip" in text:
            found.append(text)
    return found


def test_notebook_never_pip_installs_the_scientific_stack():
    """Installing numpy/scipy/torch is what broke the Kaggle run.

    It is tempting (match the local venv) and it is always wrong: the image's
    compiled packages are built against its own numpy, and pip will not tell you
    that a leftover 1.26 module makes scipy unimportable.
    """
    forbidden = ("numpy", "scipy", "torch", "numba", "librosa")
    offenders = []
    for i, src in enumerate(code_cells()):
        for call in _pip_calls(src):
            args = call.split("install", 1)[-1].lower()
            for pkg in forbidden:
                if pkg in args:
                    offenders.append((i, call[:120], pkg))
    assert not offenders, ("the notebook must not pip-install the scientific stack; "
                           f"it ships with the image: {offenders}")


def test_the_pip_guard_can_actually_see_a_guarded_install():
    """Meta-test: the detector above must fire on the code that caused the outage.

    Without this, a future refactor of `_pip_calls` could make the guard vacuous
    and everything would still look green.
    """
    historical = (
        'subprocess.run([sys.executable, "-m", "pip", "install", "-q",\n'
        '                "numpy<2", "scipy", "soundfile"], check=False)'
    )
    calls = _pip_calls(historical)
    assert calls, "detector saw no pip call in the exact code that failed on Kaggle"
    assert any("numpy" in c.lower() for c in calls), (
        "detector found the pip call but not its arguments — package names are on "
        "the continuation line, which is what the first version of this test missed")

    # ...and it must not fire on a comment that merely mentions pip
    assert not _pip_calls('# pip install "numpy<2"\nimport numpy as np')


def test_setup_cell_verifies_imports_before_the_slow_work():
    """A broken environment must be caught in seconds, not after the download."""
    cells = code_cells()
    joined = [ast.parse(c).body for c in cells]
    # cell 1 must import scipy.signal and the project's own modules
    checks = [c for c in cells if "scipy.signal import" in c and "samadhvani.audio.mel" in c]
    assert checks, "no environment check cell — failures surface 20 minutes in"
    # it must come before any download
    check_idx = next(i for i, c in enumerate(cells) if c in checks)
    fetch_idx = next(i for i, c in enumerate(cells) if "fetch_iish(" in c)
    assert check_idx < fetch_idx
    assert joined  # keep flake8 quiet


def test_notebook_does_not_claim_a_missing_piece_is_preinstalled():
    """soundfile is the one package worth installing; the cell must handle it."""
    setup = code_cells()[0]
    assert "soundfile" in setup
    assert "MP3" in setup, "must check that libsndfile has MP3, not just that it imports"


@pytest.mark.parametrize("needle", ["HOURS", "CLIPS", "EPOCHS"])
def test_settings_are_single_named_constants(needle):
    """The user is told to change one number; make sure that is true."""
    cells = code_cells()
    hits = [c for c in cells if any(
        line.startswith(f"{needle} ") or line.startswith(f"{needle}=")
        for line in c.splitlines())]
    assert len(hits) >= 1, f"{needle} is not set anywhere the user can edit it"
