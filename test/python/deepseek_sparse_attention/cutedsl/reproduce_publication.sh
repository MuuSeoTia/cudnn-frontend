#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Python handles archive preparation and report validation; bash is the entry point.
exec "${PYTHON:-python3.12}" - "$0" "$@" <<'PY'
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import ssl
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import venv
import xml.etree.ElementTree as ET

BASE = "f243150ad4079311da2c59497e9a1b514efd91ae"
ARCHIVE_SHA256 = "6d4cf2ffdfee42b4d72f1e9400167895d7bf019dd0a71ad546cac0328b23a5fb"
KERNEL = "python/cudnn/deepseek_sparse_attention/score_recompute/dense_score_recompute_sm100.py"
DSA = "deepseek_sparse_attention/cutedsl/"
INDEXER = DSA + "test_DSA_indexer_forward.py"
COMPRESSED = DSA + "test_DSA_indexer_forward_compressed.py"
DENSE = DSA + "test_DSA_dense_score_recompute.py"
INPUTS = {
    KERNEL: "7d21c0ff3cdb560d841ab76400d077fbd140d483cdab97fa982106663a0e147e",
    "test/python/" + INDEXER: "3e43d15c627cad69903c9e50e6f3b5ab8c713cb0660cd136512883cac942fc86",
}
SUITES = {
    "fp32": (36, [
        INDEXER + "::test_DSA_indexer_forward_wrapper",
        INDEXER + "::test_DSA_indexer_forward_wrapper_thd_varlen_tails",
        INDEXER + "::test_DSA_indexer_forward_class_head_weight_contract",
        COMPRESSED + "::test_DSA_compressed_indexer_forward_bshd_preallocated_lse",
        COMPRESSED + "::test_DSA_compressed_indexer_forward_thd_preallocated_global_indices",
    ]),
    "publication": (6, [INDEXER + "::test_DSA_indexer_forward_per_head_publication"]),
    "attention": (2, [
        DENSE + "::test_DSA_dense_score_recompute_wrapper[attention-32-128-acc_dtype0-dtype0]",
        DENSE + "::test_DSA_dense_score_recompute_thd_execute_uses_plan_envelope[attention]",
    ]),
    "shared": (39, [
        DENSE,
        INDEXER + "::test_DSA_indexer_forward_wrapper_thd_varlen_tails",
        COMPRESSED + "::test_DSA_compressed_indexer_forward_bshd_preallocated_lse",
        COMPRESSED + "::test_DSA_compressed_indexer_forward_thd_preallocated_global_indices",
        "-k", "not mxfp8 and not mxfp4 and not sm90",
    ]),
}
CHILD = r'''
import faulthandler, importlib, importlib.metadata, json, os, pathlib, sys
faulthandler.dump_traceback_later(3540)
root = pathlib.Path(os.environ["REPRO_ROOT"])
sys.path[:0] = [str(root / "python"), str(root / "test/python")]
import pytest
class Receipt:
    def pytest_sessionstart(self, session):
        import cudnn, torch
        kernel = importlib.import_module(
            "cudnn.deepseek_sparse_attention.score_recompute.dense_score_recompute_sm100")
        for module in (cudnn, kernel):
            if not pathlib.Path(module.__file__).resolve().is_relative_to(root):
                raise RuntimeError(f"Wrong source imported: {module.__file__}; expected {root}")
        cc = torch.cuda.get_device_capability()
        if cc[0] != 10 or cudnn.backend_version() != 92200:
            raise RuntimeError(f"Need SM100-family GPU and cuDNN 9.22; got {cc}, {cudnn.backend_version()}")
        record = {"cudnn": cudnn.__file__, "kernel": kernel.__file__,
            "backend": cudnn.backend_version(), "gpu": torch.cuda.get_device_name(),
            "cc": cc, "sms": torch.cuda.get_device_properties(0).multi_processor_count,
            "packages": {n: importlib.metadata.version(n) for n in
                ("torch", "nvidia-cutlass-dsl", "cuda-bindings", "apache-tvm-ffi", "pytest")}}
        pathlib.Path(os.environ["REPRO_RUNTIME"]).write_text(json.dumps(record, indent=2) + "\n")
sys.exit(pytest.main(sys.argv[1:], plugins=[Receipt()]))
'''


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(root):
    return {
        str(p.relative_to(root)): "symlink:" + os.readlink(p) if p.is_symlink() else sha256(p)
        for p in sorted(root.rglob("*")) if p.is_file() or p.is_symlink()
    }


def only_change(before, after, expected):
    changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
    if changed != {expected}:
        raise RuntimeError(f"Expected only {expected} to change; found {sorted(changed)}")


def stop_process_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return process.wait()


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def main():
    script = Path(sys.argv[1]).resolve()
    parser = argparse.ArgumentParser(prog=script.name, description="Reproduce DSA publication RED/GREEN and independent FP32 qualification.")
    parser.add_argument("--output", type=Path, help="new output directory (default: dsa-publication-* in the current directory)")
    parser.add_argument("--suite", choices=["all", *SUITES], default="all")
    parser.add_argument("--existing-env", action="store_true", help="use PYTHON's installed packages instead of creating a pinned virtualenv")
    parser.add_argument("--source-archive", type=Path, help="offline copy of the pinned NVIDIA codeload archive (SHA256 checked)")
    args = parser.parse_args(sys.argv[2:])
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
        parser.error("use Linux and Python 3.12 (set PYTHON=/path/to/python3.12)")
    sanitizer = shutil.which(os.environ.get("COMPUTE_SANITIZER", "compute-sanitizer"))
    if not sanitizer:
        parser.error("compute-sanitizer is required; install the CUDA toolkit or set COMPUTE_SANITIZER=/path/to/compute-sanitizer")
    checkout = script.parents[4]
    for name, expected in INPUTS.items():
        if not (checkout / name).is_file() or sha256(checkout / name) != expected:
            parser.error(f"{name} does not match the qualified review source; run from the publication review checkout")
    output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="dsa-publication-", dir=Path.cwd()))
    if args.output:
        try:
            output.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            parser.error(f"--output must name a new directory: {output}")
    reports = output / "reports"
    reports.mkdir()
    print(f"Results: {reports}", flush=True)
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "CUTE_DSL_ARCH"):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    python = sys.executable
    if not args.existing_env:
        venv.EnvBuilder(with_pip=True).create(output / "venv")
        python = str(output / "venv/bin/python")
        packages = [
            "torch==2.13.0", "nvidia-cudnn-frontend==1.32.1", "nvidia-cutlass-dsl[cu13]==4.8.1",
            "cuda-bindings==13.4.4", "apache-tvm-ffi==0.1.12", "numpy==1.26.4",
            "pytest==8.4.2", "pytest-xdist==3.8.0", "looseversion==1.3.0",
            "certifi==2026.10.23",
        ]
        with (reports / "install.log").open("w") as log:
            print("Installing pinned packages into a fresh virtualenv (see reports/install.log)...", flush=True)
            subprocess.run([python, "-m", "pip", "install", *packages], env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
            # Torch pins 9.20; the separately measured runtime uses this explicit override.
            subprocess.run([python, "-m", "pip", "install", "--no-deps", "nvidia-cudnn-cu13==9.22.0.52"],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
    with (reports / "packages.txt").open("w") as log:
        subprocess.run([python, "-m", "pip", "freeze", "--all"], env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    with (reports / "sanitizer-version.txt").open("w") as log:
        subprocess.run([sanitizer, "--version"], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    # Locate the wheel's native extension without importing its Python package.
    binding = subprocess.check_output([python, "-c", '''
import importlib.metadata
d = importlib.metadata.distribution("nvidia-cudnn-frontend")
if d.version != "1.32.1":
    raise RuntimeError(f"Need nvidia-cudnn-frontend 1.32.1; got {d.version}")
print(next(d.locate_file(f) for f in d.files if str(f).startswith("cudnn/_compiled_module") and str(f).endswith(".so")))
'''], env=env, text=True, timeout=60).strip()
    archive = args.source_archive.resolve() if args.source_archive else output / "main.tar.gz"
    if not args.source_archive:
        print("Downloading pinned NVIDIA main source...", flush=True)
        try:
            cafile = subprocess.check_output([python, "-c", "import certifi; print(certifi.where())"],
                                            env=env, text=True, stderr=subprocess.PIPE, timeout=60).strip()
        except subprocess.CalledProcessError:
            parser.error("PYTHON's environment needs certifi for the source download; install certifi==2026.10.23 or pass --source-archive")
        context = ssl.create_default_context()
        context.load_verify_locations(cafile=cafile)
        with urllib.request.urlopen(f"https://codeload.github.com/NVIDIA/cudnn-frontend/tar.gz/{BASE}", context=context, timeout=120) as response:
            with archive.open("wb") as target:
                shutil.copyfileobj(response, target)
    if sha256(archive) != ARCHIVE_SHA256:
        raise RuntimeError(f"Source archive checksum mismatch: {archive}")
    sources = output / "sources"
    with tarfile.open(archive) as tar:
        tar.extractall(sources, filter="data")
    pristine = sources / f"cudnn-frontend-{BASE}"
    shutil.copy2(binding, pristine / "python/cudnn")
    baseline, fixed = sources / "baseline", sources / "fixed"
    shutil.copytree(pristine, baseline, symlinks=True)
    shutil.copy2(checkout / "test/python" / INDEXER, baseline / "test/python" / INDEXER)
    shutil.copytree(baseline, fixed, symlinks=True)
    shutil.copy2(checkout / KERNEL, fixed / KERNEL)
    roots = {"pristine": pristine, "baseline": baseline, "fixed": fixed}
    hashes = {arm: manifest(root) for arm, root in roots.items()}
    only_change(hashes["pristine"], hashes["baseline"], "test/python/" + INDEXER)
    only_change(hashes["baseline"], hashes["fixed"], KERNEL)
    receipt = {"base": BASE, "archive_sha256": sha256(archive), "script_sha256": sha256(script),
               "source_hashes": hashes, "existing_env": args.existing_env, "runs": []}
    receipt_path = reports / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    env.update(CUDNN_TEST_NO_ISOLATION="1", CUTE_DSL_LINEINFO="1", OMP_NUM_THREADS="1",
               CUDNN_FRONTEND_DISABLE_COMPILED_CACHE="1", CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO="0")
    signal.signal(signal.SIGTERM, interrupted)
    failed = []
    for suite in SUITES if args.suite == "all" else [args.suite]:
        expected, tests = SUITES[suite]
        arms = ["pristine"] if suite == "fp32" else ["fixed"] if suite == "shared" else ["baseline", "fixed"]
        for arm in arms:
            for tool in ["numerics"] if suite in ("fp32", "shared") else ["numerics", "racecheck", "synccheck"]:
                label = f"{arm}-{suite}-{tool}"
                xml = reports / f"{label}.xml"
                command = [python, "-c", CHILD, "-p", "xdist.plugin", "-n", "0", "-m", "L0 or L1", "-v", "-ra",
                           "--junitxml=" + str(xml), *tests]
                if tool != "numerics":
                    command = [sanitizer, "--tool", tool, "--report-api-errors", "extended", "--error-exitcode", "86", *command]
                run_env = env | {"REPRO_ROOT": str(roots[arm]), "REPRO_RUNTIME": str(reports / f"{label}-runtime.json"),
                                 "CUDA_CACHE_PATH": str(output / "cache" / label)}
                print(f"Running {label}...", flush=True)
                log_path = reports / f"{label}.log"
                timed_out = False
                report_error = None
                with log_path.open("w") as log:
                    with subprocess.Popen(command, cwd=roots[arm] / "test/python", env=run_env,
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True) as process:
                        try:
                            returncode = process.wait(timeout=3600)
                        except subprocess.TimeoutExpired:
                            timed_out = True
                            returncode = stop_process_group(process)
                            log.write("\nQualification process exceeded 3600 seconds.\n")
                        except BaseException:
                            stop_process_group(process)
                            raise
                text = log_path.read_text()
                totals = dict.fromkeys(("tests", "failures", "errors", "skipped"), 0)
                if xml.is_file():
                    try:
                        for node in ET.parse(xml).getroot().iter("testsuite"):
                            for key in totals:
                                totals[key] += int(node.get(key, 0))
                    except (ET.ParseError, ValueError) as error:
                        report_error = str(error)
                else:
                    report_error = "JUnit XML was not produced"
                ok = not timed_out and report_error is None and totals == {"tests": expected, "failures": 0, "errors": 0, "skipped": 0}
                summaries = re.findall(r"(?:RACECHECK|ERROR) SUMMARY: [^\n]+", text)
                red = arm == "baseline" and tool == "racecheck"
                ok &= returncode == (86 if red else 0)
                if tool == "racecheck":
                    match = re.search(r"RACECHECK SUMMARY: \d+ hazards displayed \((\d+) errors, (\d+) warnings\)", text)
                    ok &= bool(match and int(match[2]) == 0 and (int(match[1]) > 0 if red else int(match[1]) == 0))
                    if red:
                        # An unrelated sanitizer failure is not publication RED evidence.
                        sites = ["dense_score_recompute_sm100.py:598", "dense_score_recompute_sm100.py:1347"] if suite == "attention" else [
                            "dense_score_recompute_sm100.py:600", "indexer_score_unified_sm100.py:170"]
                        ok &= all(site in text for site in sites)
                elif tool == "synccheck":
                    ok &= "ERROR SUMMARY: 0 errors" in text
                receipt["runs"].append({"label": label, "command": command, "exit_code": returncode,
                                        "timed_out": timed_out, "report_error": report_error,
                                        "pytest": totals, "sanitizer": summaries, "passed": ok})
                receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
                print(f"{'PASS' if ok else 'FAIL'} {label}: {totals}; {'; '.join(summaries)}", flush=True)
                if not ok:
                    failed.append(label)
                    print(text[-4000:], flush=True)
                if timed_out:
                    raise RuntimeError(f"{label} timed out; qualification stopped. Logs: {reports}")
    if failed:
        raise RuntimeError(f"Qualification failed: {', '.join(failed)}. Logs: {reports}")
    print(f"PASS: all requested checks completed without skips. Evidence: {reports}", flush=True)


main()
PY
