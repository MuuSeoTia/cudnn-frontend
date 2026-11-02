# DSA per-head publication reproduction

Base: NVIDIA `main`, `f243150ad4079311da2c59497e9a1b514efd91ae` (1.32.1).

## Setup

Use Linux with an SM100-family GPU (B200/GB200/GB300), a CUDA 13 toolkit and
compatible driver, cuDNN 9 headers/libraries, and a C++17 compiler. Put `nvcc`
and `compute-sanitizer` on `PATH`. As in [Building from Source](../../../../README.md#building-from-source),
set `CUDNN_PATH` and `CUDAToolkit_ROOT` if they are installed outside the default locations.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
git clone --branch fix-dsa-publication-main https://github.com/MuuSeoTia/cudnn-frontend.git test_fix
cd test_fix
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install -e .
```

Run both sections below from this fresh checkout. The checker uses the installed
editable package, disables compiled caches, and runs six numerical regression
cases under racecheck. It writes `publication-results.xml` at the repository root.

## Before: without the fix

Replace only the kernel with the version from the pinned NVIDIA main commit:

```bash
git restore --source=f243150ad4079311da2c59497e9a1b514efd91ae -- \
  python/cudnn/deepseek_sparse_attention/score_recompute/dense_score_recompute_sm100.py
python test/python/deepseek_sparse_attention/cutedsl/reproduce_publication.py
```

Expected: **6 passed**, but racecheck reports shared-memory conflicts and the
checker exits **86**. Recorded on GB300: **126 racecheck errors**. Numerical
checks alone do not expose this race; hazard counts can vary by GPU and runtime.

## After: with the fix

Restore the patched kernel from this branch and run the same checker:

```bash
git restore --source=HEAD -- \
  python/cudnn/deepseek_sparse_attention/score_recompute/dense_score_recompute_sm100.py
python test/python/deepseek_sparse_attention/cutedsl/reproduce_publication.py
```

Expected: **6 passed, no skips, zero racecheck errors**, exit **0**. The editable
install picks up the Python kernel change without reinstalling.

## Independent changes

FP32 head weights and dtype-aware compile keys already exist on this base
([NVIDIA #1311](https://github.com/NVIDIA/cudnn-frontend/pull/1311)). The 36
existing FP32/BF16, cache-order, packed and compressed cases were independently
qualified on pristine main. This PR adds only the publication fix.

The load warp's ordinary W/LSE stores need a handoff to epilogue readers;
Q/K tensor-core completion does not publish every load lane's stores. A
two-slot `PipelineAsync` uses all 32 load lanes as producers and all 128
epilogue lanes as consumers. Consumer release prevents premature slot reuse.
Public arguments, weight types, math and the two-slot data buffer are unchanged.
Separate MXFP8/MXFP4 implementations are outside this qualification.

## Recorded qualification

The Python checker was rerun on 2026-11-02 with the runtime below: six passes
per arm, racecheck **126 errors → 0**, exit **86 → 0**, and no skipped cases.

GB300 SM103 (152 SMs), Python 3.12, Torch 2.13.0, CuTeDSL 4.8.1,
cuda-bindings 13.4.4, TVM-FFI 0.1.12, cuDNN 9.22.0.52, pytest 8.4.2 and
Compute Sanitizer 2026.3.0.0. Source Python used the official Frontend 1.32.1
wheel's native extension.

### Earlier full matrix

The pre-simplification harness produced the matrix below. The baseline received
only the regression test; the fixed tree additionally received the kernel change.

| Check | Unfixed | Fixed |
| --- | --- | --- |
| Existing FP32/BF16 support | 36 passed | Independent baseline check |
| Publication numerics | 6 passed | 6 passed |
| Publication racecheck | 126 errors, 100 displayed hazards | 0 errors |
| Focused attention numerics | 2 passed | 2 passed |
| Focused attention racecheck | 8 errors | 0 errors |
| Synccheck, publication and focused attention | 0 errors | 0 errors |
| Shared-mainloop cases | Not selected | 39 passed, 44 deselected |

No selected case skipped. These are kernel/API correctness results, not model
accuracy or performance measurements.
