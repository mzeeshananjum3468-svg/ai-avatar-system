"""
test_musetalk_worker.py — talk to musetalk_worker.py directly, no backend needed.

Usage:
    python test_musetalk_worker.py <path_to_avatar.jpg> <path_to_audio.wav> <output.mp4>

Run this from anywhere; it only needs MUSETALK_PATH to find the checkout.
Set env vars first if your setup differs from defaults, e.g.:
    export MUSETALK_PATH=/path/to/MuseTalk
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <avatar.jpg> <audio.wav> <output.mp4>")
        sys.exit(1)

    image_path, audio_path, output_path = (str(Path(p).resolve()) for p in sys.argv[1:4])

    for p, label in [(image_path, "avatar image"), (audio_path, "audio")]:
        if not os.path.exists(p):
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)

    musetalk_dir = Path(os.environ.get("MUSETALK_PATH", "./backend/models/MuseTalk")).resolve()
    worker_script = musetalk_dir / "scripts" / "musetalk_worker.py"
    if not worker_script.exists():
        print(f"ERROR: worker script not found at {worker_script}")
        print("Set MUSETALK_PATH env var to your MuseTalk checkout, or copy the file there.")
        sys.exit(1)

    print(f"MuseTalk dir : {musetalk_dir}")
    print(f"Worker script: {worker_script}")
    print(f"Avatar image : {image_path}")
    print(f"Audio        : {audio_path}")
    print(f"Output       : {output_path}")
    print()

    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(musetalk_dir) + (":" + existing_pp if existing_pp else "")

    use_float16 = env.get("MUSETALK_FLOAT16", "1") == "1"

    init_msg = {
        "unet_model_path": str(musetalk_dir / "models" / "musetalkV15" / "unet.pth"),
        "unet_config": str(musetalk_dir / "models" / "musetalkV15" / "musetalk.json"),
        "whisper_dir": str(musetalk_dir / "models" / "whisper"),
        "vae_type": str(musetalk_dir / "models" / "sd-vae"),
        "use_float16": use_float16,
    }

    for key in ("unet_model_path", "unet_config", "whisper_dir", "vae_type"):
        if not os.path.exists(init_msg[key]):
            print(f"WARNING: {key} does not exist at {init_msg[key]}")
            print("  (did scripts/setup_musetalk.sh finish downloading models?)")

    print("Starting worker process...")
    proc = subprocess.Popen(
        [sys.executable, str(worker_script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(musetalk_dir),
        env=env,
        text=True,
        bufsize=1,
    )

    proc.stdin.write(json.dumps(init_msg) + "\n")
    proc.stdin.flush()

    print("Waiting for models to load (this can take 1-10 min first run)...")
    print("--- worker stderr will stream below (this is normal, not an error) ---")

    # Stream stderr in the background so you can see loading/progress logs live,
    # while we block on stdout waiting for the READY line.
    import threading

    def pump_stderr():
        for line in proc.stderr:
            print(f"[worker] {line.rstrip()}")

    t = threading.Thread(target=pump_stderr, daemon=True)
    t.start()

    ready_line = proc.stdout.readline()
    if not ready_line.strip().startswith("READY"):
        print(f"\nERROR: worker did not send READY (got: {ready_line!r}). Check stderr above.")
        proc.kill()
        sys.exit(1)

    print("\n--- worker ready, sending job ---\n")

    job = {
        "image": image_path,
        "audio": audio_path,
        "output": output_path,
        "coord_cache": str(musetalk_dir / "results" / "coords" / "test_avatar.pkl"),
    }
    os.makedirs(os.path.dirname(job["coord_cache"]), exist_ok=True)

    t0 = time.time()
    proc.stdin.write(json.dumps(job) + "\n")
    proc.stdin.flush()

    result_line = proc.stdout.readline()
    elapsed = time.time() - t0

    if not result_line:
        print("ERROR: worker exited without returning a result (check stderr above).")
        proc.kill()
        sys.exit(1)

    result = json.loads(result_line)
    print(f"\n--- result ({elapsed:.1f}s) ---")
    print(result)

    if result.get("status") == "ok" and os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"\nSUCCESS: {output_path} ({size_mb:.1f} MB)")
    else:
        print("\nFAILED — see 'msg' above and worker stderr for details.")

    proc.stdin.close()
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


if __name__ == "__main__":
    main()