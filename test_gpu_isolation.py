#!/usr/bin/env python3
"""
GPU isolation test for pytest-fkit.

Tests that subprocesses with per-worker ROCR/HIP/CUDA env vars
can correctly see and use their assigned GPU.

This simulates exactly what pytest-fkit does when running tests:
  1. Detect GPUs from env vars
  2. Allocate physical GPU IDs to workers
  3. Spawn a subprocess per worker with the corrected env vars
  4. Each subprocess verifies torch.cuda.is_available() and runs a GPU op

Tests both the OLD (broken) env var approach and the NEW (fixed) approach
to demonstrate the bug and the fix.
"""
import os
import sys
import subprocess
import json

# =========================================================================
# GPU probe script: runs inside each subprocess
# =========================================================================
PROBE_SCRIPT = r'''
import os, sys, json, glob

result = {
    "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES", ""),
    "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", ""),
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "kfd_exists": os.path.exists("/dev/kfd"),
    "render_nodes": sorted(glob.glob("/dev/dri/renderD*")),
}

try:
    import torch
    result["torch_version"] = torch.__version__
    result["hip_version"] = getattr(torch.version, "hip", None)
    result["cuda_available"] = torch.cuda.is_available()
    result["device_count"] = torch.cuda.device_count() if result["cuda_available"] else 0

    if result["cuda_available"] and result["device_count"] > 0:
        # Actually use the GPU: allocate a tensor and do a computation
        t = torch.randn(256, 256, device="cuda:0")
        r = torch.mm(t, t)
        result["gpu_compute_ok"] = True
        result["gpu_name"] = torch.cuda.get_device_name(0)
        del t, r
        torch.cuda.empty_cache()
    else:
        result["gpu_compute_ok"] = False
        result["gpu_name"] = None
except Exception as e:
    result["error"] = str(e)
    result["cuda_available"] = False
    result["device_count"] = 0
    result["gpu_compute_ok"] = False

print(json.dumps(result))
'''


def run_probe(env_overrides: dict, label: str) -> dict:
    """Spawn a subprocess with specific env vars and probe GPU access."""
    env = os.environ.copy()
    env.update(env_overrides)

    try:
        r = subprocess.run(
            [sys.executable, "-c", PROBE_SCRIPT],
            capture_output=True, text=True, timeout=60, env=env,
        )
        if r.returncode != 0:
            return {"label": label, "error": f"exit={r.returncode}", 
                    "stderr": r.stderr.strip()[-500:],
                    "cuda_available": False, "gpu_compute_ok": False}
        data = json.loads(r.stdout.strip().split("\n")[-1])
        data["label"] = label
        return data
    except Exception as e:
        return {"label": label, "error": str(e),
                "cuda_available": False, "gpu_compute_ok": False}


def detect_gpus():
    """Detect GPUs from env or rocm-smi."""
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        val = os.environ.get(var, "").strip()
        if val:
            return [x.strip() for x in val.split(",") if x.strip()]

    import shutil
    if shutil.which("rocm-smi"):
        import re
        r = subprocess.run(["rocm-smi", "--showid"], capture_output=True, text=True, timeout=10)
        ids = set()
        for line in r.stdout.split("\n"):
            m = re.search(r"GPU\[(\d+)\]", line)
            if m:
                ids.add(m.group(1))
        if ids:
            return sorted(ids, key=int)
    return []


def main():
    gpu_ids = detect_gpus()
    if not gpu_ids:
        print("No GPUs detected. Cannot run isolation test.")
        sys.exit(1)

    print(f"Detected {len(gpu_ids)} GPUs: {gpu_ids}")
    print(f"/dev/kfd exists: {os.path.exists('/dev/kfd')}")
    print()

    # =====================================================================
    # Test 1: Baseline — no env var overrides, use parent env
    # =====================================================================
    print("=" * 70)
    print("TEST 1: Baseline (parent env, no overrides)")
    print("=" * 70)
    res = run_probe({}, "baseline")
    print(f"  cuda_available={res.get('cuda_available')} "
          f"device_count={res.get('device_count')} "
          f"gpu_compute={res.get('gpu_compute_ok')} "
          f"gpu={res.get('gpu_name', 'N/A')}")
    if res.get("error"):
        print(f"  ERROR: {res['error']}")
    print()

    # =====================================================================
    # Test 2: OLD approach (broken) — all three vars set to physical ID
    # =====================================================================
    print("=" * 70)
    print("TEST 2: OLD approach (ROCR=physical, HIP=physical, CUDA=physical)")
    print("        This is the BROKEN approach — fails for GPU index > 0")
    print("=" * 70)
    old_results = []
    for gpu_id in gpu_ids:
        env = {
            "ROCR_VISIBLE_DEVICES": gpu_id,
            "HIP_VISIBLE_DEVICES": gpu_id,
            "CUDA_VISIBLE_DEVICES": gpu_id,
        }
        res = run_probe(env, f"old_gpu{gpu_id}")
        status = "PASS" if res.get("gpu_compute_ok") else "FAIL"
        print(f"  GPU {gpu_id}: {status} "
              f"(cuda_available={res.get('cuda_available')}, "
              f"device_count={res.get('device_count')}, "
              f"compute={res.get('gpu_compute_ok')})")
        if res.get("error"):
            print(f"         ERROR: {res['error'][:200]}")
        old_results.append(res)
    print()

    # =====================================================================
    # Test 3: NEW approach (fixed) — ROCR=physical, HIP=0, CUDA=0
    # =====================================================================
    print("=" * 70)
    print("TEST 3: NEW approach (ROCR=physical, HIP=0-based, CUDA=0-based)")
    print("        This is the FIXED approach — HIP re-indexed within ROCR set")
    print("=" * 70)
    new_results = []
    for gpu_id in gpu_ids:
        env = {
            "ROCR_VISIBLE_DEVICES": gpu_id,
            "HIP_VISIBLE_DEVICES": "0",
            "CUDA_VISIBLE_DEVICES": "0",
        }
        res = run_probe(env, f"new_gpu{gpu_id}")
        status = "PASS" if res.get("gpu_compute_ok") else "FAIL"
        print(f"  GPU {gpu_id}: {status} "
              f"(cuda_available={res.get('cuda_available')}, "
              f"device_count={res.get('device_count')}, "
              f"compute={res.get('gpu_compute_ok')}, "
              f"name={res.get('gpu_name', 'N/A')})")
        if res.get("error"):
            print(f"         ERROR: {res['error'][:200]}")
        new_results.append(res)
    print()

    # =====================================================================
    # Test 4: Multi-GPU worker (2 GPUs per worker)
    # =====================================================================
    if len(gpu_ids) >= 2:
        print("=" * 70)
        print("TEST 4: Multi-GPU worker (ROCR=phys0,phys1, HIP=0,1, CUDA=0,1)")
        print("=" * 70)
        pair = f"{gpu_ids[0]},{gpu_ids[1]}"
        env = {
            "ROCR_VISIBLE_DEVICES": pair,
            "HIP_VISIBLE_DEVICES": "0,1",
            "CUDA_VISIBLE_DEVICES": "0,1",
        }
        res = run_probe(env, "multi_gpu")
        status = "PASS" if res.get("gpu_compute_ok") else "FAIL"
        print(f"  GPUs {pair}: {status} "
              f"(cuda_available={res.get('cuda_available')}, "
              f"device_count={res.get('device_count')}, "
              f"compute={res.get('gpu_compute_ok')})")
        if res.get("error"):
            print(f"         ERROR: {res['error'][:200]}")
        print()

    # =====================================================================
    # Summary
    # =====================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    old_pass = sum(1 for r in old_results if r.get("gpu_compute_ok"))
    new_pass = sum(1 for r in new_results if r.get("gpu_compute_ok"))
    print(f"  OLD approach: {old_pass}/{len(old_results)} GPUs passed")
    print(f"  NEW approach: {new_pass}/{len(new_results)} GPUs passed")
    if old_pass < len(old_results) and new_pass == len(new_results):
        print(f"\n  ✓ FIX CONFIRMED: Old approach broke GPU {','.join(gpu_ids[1:])} "
              f"but new approach works for all GPUs")
    elif new_pass == len(new_results):
        print(f"\n  ✓ All GPUs accessible with new approach")
    else:
        print(f"\n  ⚠ Some GPUs still not accessible — check device permissions")


if __name__ == "__main__":
    main()
