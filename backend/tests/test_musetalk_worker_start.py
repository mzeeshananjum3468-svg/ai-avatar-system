import json
import os
import select
import subprocess
import sys

import pytest


@pytest.mark.integration
@pytest.mark.slow
def test_musetalk_worker_starts_and_reports_ready():
    """Verify that the persistent worker can be spawned and emits a READY line.

    The test uses the ``MUSETALK_PATH`` environment variable to locate the
    checkout.  It starts the worker as a subprocess, sends the minimal init JSON
    payload, and asserts that the first line on stdout is ``READY``.
    """

    musetalk_dir = os.getenv("MUSETALK_PATH", "/media/muhammadfaisal/Data3/Zeeshan/MuseTalk")
    worker_script = os.path.join(musetalk_dir, "scripts", "musetalk_worker.py")
    required_files = [
        worker_script,
        os.path.join(musetalk_dir, "models", "musetalkV15", "unet.pth"),
        os.path.join(musetalk_dir, "models", "musetalkV15", "musetalk.json"),
        os.path.join(musetalk_dir, "models", "whisper", "config.json"),
        os.path.join(musetalk_dir, "models", "sd-vae", "config.json"),
    ]

    missing = [path for path in required_files if not os.path.exists(path)]
    if missing:
        pytest.skip("MuseTalk assets are missing:\n" + "\n".join(missing))

    # Launch the worker
    proc = subprocess.Popen(
        [sys.executable, worker_script],
        cwd=musetalk_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Build the init payload – only the fields required by the worker
    init_cfg = {
        "unet_model_path": os.path.join(musetalk_dir, "models", "musetalkV15", "unet.pth"),
        "unet_config": os.path.join(musetalk_dir, "models", "musetalkV15", "musetalk.json"),
        "whisper_dir": os.path.join(musetalk_dir, "models", "whisper"),
        "vae_type": os.path.join(musetalk_dir, "models", "sd-vae"),
        "use_float16": False,
    }

    proc.stdin.write(json.dumps(init_cfg) + "\n")
    proc.stdin.flush()

    ready_line = ""
    try:
        # Give the worker time to print READY while still failing fast if it
        # crashes during model load.
        if not select.select([proc.stdout], [], [], 600)[0]:
            stderr_output = proc.stderr.read() or ""
            raise AssertionError(f"Worker timed out waiting for READY. stderr:\n{stderr_output}")
        ready_line = proc.stdout.readline().strip()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

    stderr_output = proc.stderr.read() or ""

    assert ready_line == "READY", f"Worker did not emit READY, got: {ready_line}\nstderr:\n{stderr_output}"