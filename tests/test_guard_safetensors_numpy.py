"""Top-level NumPy safetensors reads must retain operator-level weight metering."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from weightbench.harness import sandbox

st_numpy = pytest.importorskip("safetensors.numpy")


def _run_guarded(root: Path, code: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PYTHONPATH": str(root / "lib"),
        "WB_TRACE_DIR": str(root / "trace"),
        "WB_SHAPES": str(root / "lib" / "shapes.json"),
    }
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_safe_open_numpy_tensor_and_slice_reads_are_metered(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    weights = public / "weights.safetensors"
    st_numpy.save_file(
        {
            "W1": np.ones((8, 4), dtype=np.float32),
            "W2": np.ones((3, 8), dtype=np.float32),
        },
        str(weights),
    )
    guard = sandbox.install_guard(tmp_path / "guard", public)
    code = (
        "import numpy as np\n"
        "from safetensors import safe_open\n"
        f"with safe_open({str(weights)!r}, framework='np') as f:\n"
        "    w1 = f.get_tensor('W1')\n"
        "    w2 = f.get_slice('W2')[:]\n"
        "x = np.ones((5, 4), dtype=np.float32)\n"
        "h = np.maximum(x @ w1.T, 0)\n"
        "y = h @ w2.T\n"
        "print(type(w1).__name__, type(w2).__name__, type(y).__name__, float(y.sum()))\n"
    )

    result = _run_guarded(guard, code)

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["WeightArray", "WeightArray", "WeightArray", "480.0"]
    traces = [json.loads(path.read_text()) for path in (guard / "trace").glob("*.json")]
    assert len(traces) == 1
    trace = traces[0]
    assert trace["flops"] == 560
    assert trace["weight_matmuls"] == 2
    assert trace["forward_chains"] == 1
    assert trace["metering_escapes"] == 0
    assert trace["unsupported_contractions"] == 0
