# DSA per-head publication qualification

Base: NVIDIA `main`, `f243150ad4079311da2c59497e9a1b514efd91ae` (1.32.1).

## Independent changes

FP32 head weights are already supported on this base, including dtype-aware
compile keys ([NVIDIA #1311](https://github.com/NVIDIA/cudnn-frontend/pull/1311)).
The existing tests below check both weight dtypes, both cache-population orders,
FP32 values distinguishable from BF16 rounding, packed sequences, and compressed
indexing. Run them on unchanged `main`; no publication fix is needed to establish
that API support. Numerical success does not establish race freedom.

The publication fix orders the load warp's ordinary shared-memory stores of
W/LSE before epilogue reads. Q/K tensor-core pipeline completion does not publish
every load lane's ordinary stores. A two-slot `PipelineAsync` uses all 32 load
lanes as producers and all 128 epilogue lanes as consumers. Consumer release
also protects each slot from reuse by a later persistent tile. Public arguments,
weight types, math and the existing two-slot data buffer are unchanged.
This change covers the BF16 Q/K shared mainloop and its indexer/attention
callers. The MXFP8/MXFP4 kernels have separate implementations and are outside
this qualification; their per-head synchronization needs a separate audit.

## Runtime

Use Linux, Python 3.12, a Blackwell SM100-family GPU and Compute Sanitizer.
The qualification uses Torch 2.13.0+cu130, CuTe DSL 4.8.1 (CUDA 13.4),
cuda-bindings 13.4.4, apache-tvm-ffi 0.1.12, NumPy 1.26.4, cuDNN 9.22.0.52,
pytest 8.4.2 and pytest-xdist 3.8.0. Install these in a dedicated environment.
Torch's package metadata pins cuDNN 9.20.0.48; 9.22.0.52 is an explicit
`--no-deps` override in this measured environment.

The commands below use the official `nvidia-cudnn-frontend==1.32.1` wheel's
native extension with the checkout's Python sources. This qualifies the Python
kernel change, not a fresh C++ extension build. Keep the wheel installed in the
environment and avoid an editable install pointing at another checkout.

## Prepare the two source trees

From a fresh clone of the committed review branch:

```bash
base=f243150ad4079311da2c59497e9a1b514efd91ae
fixed=$(pwd)
baseline="${fixed}-baseline"
git worktree add --detach "$baseline" "$base"

# Copy only the new test into the unfixed tree, never its kernel implementation.
git diff "$base" HEAD -- test/python/deepseek_sparse_attention/cutedsl/test_DSA_indexer_forward.py > "${fixed}-publication-test.patch"
git -C "$baseline" apply "${fixed}-publication-test.patch"

# Resolve the installed wheel before selecting either source overlay.
binding=$(env -u PYTHONPATH python -c 'import cudnn, pathlib; print(next(pathlib.Path(cudnn.__file__).parent.glob("_compiled_module*.so")))')
cp "$binding" "$fixed/python/cudnn/"
cp "$binding" "$baseline/python/cudnn/"
export CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1
export CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO=0
export CUTE_DSL_LINEINFO=1
export OMP_NUM_THREADS=1
```

Select a tree explicitly for each run. Read `cuDNN Frontend Path:` in pytest's
banner and verify that the concrete kernel also resolves there:

```bash
tree="$baseline"  # repeat with "$fixed"
export PYTHONPATH="$tree/python:$tree/test/python"
cd "$tree/test/python"
python -c 'import cudnn; import cudnn.deepseek_sparse_attention.score_recompute.dense_score_recompute_sm100 as k; print(cudnn.__file__, k.__file__, sep="\n")'
```

## FP32 support on the baseline

These tests are unchanged from `main`. The additional publication test is not
selected, and the baseline kernel is unmodified.

```bash
dsa=deepseek_sparse_attention/cutedsl
python -m pytest -n 0 -m 'L0 or L1' -v -ra \
  "$dsa/test_DSA_indexer_forward.py::test_DSA_indexer_forward_wrapper" \
  "$dsa/test_DSA_indexer_forward.py::test_DSA_indexer_forward_wrapper_thd_varlen_tails" \
  "$dsa/test_DSA_indexer_forward.py::test_DSA_indexer_forward_class_head_weight_contract" \
  "$dsa/test_DSA_indexer_forward_compressed.py::test_DSA_compressed_indexer_forward_bshd_preallocated_lse" \
  "$dsa/test_DSA_indexer_forward_compressed.py::test_DSA_compressed_indexer_forward_thd_preallocated_global_indices"
```

## Publication RED/GREEN

Run each command in both trees with the same environment. The six cases cover
BF16/FP32 W, partial tiles, persistent slot reuse and tiles with no visible keys.
They compare against a chunked FP64 oracle, repeat the call, and change W.
The unfixed racecheck run must fail with shared W publication hazards; the fixed
run must pass. A failed import, skipped suite or numerical failure is not RED
evidence for the synchronization bug.

```bash
case=deepseek_sparse_attention/cutedsl/test_DSA_indexer_forward.py::test_DSA_indexer_forward_per_head_publication
python -m pytest -n 0 -m 'L0 or L1' -v -ra "$case"
compute-sanitizer --tool racecheck --report-api-errors extended --error-exitcode 86 \
  python -m pytest -n 0 -m 'L0 or L1' -v -ra "$case"
compute-sanitizer --tool synccheck --report-api-errors extended --error-exitcode 86 \
  python -m pytest -n 0 -m 'L0 or L1' -v -ra "$case"
```

## Shared-mainloop coverage

In the fixed tree, run existing BF16 dense attention/indexer and packed and
compressed indexer tests. Inspect skips as well as pytest's final summary.

```bash
dsa=deepseek_sparse_attention/cutedsl
python -m pytest -n 0 -m 'L0 or L1' -v -ra \
  -k 'not mxfp8 and not mxfp4 and not sm90' \
  "$dsa/test_DSA_dense_score_recompute.py" \
  "$dsa/test_DSA_indexer_forward.py::test_DSA_indexer_forward_wrapper_thd_varlen_tails" \
  "$dsa/test_DSA_indexer_forward_compressed.py::test_DSA_compressed_indexer_forward_bshd_preallocated_lse" \
  "$dsa/test_DSA_indexer_forward_compressed.py::test_DSA_compressed_indexer_forward_thd_preallocated_global_indices"
```
