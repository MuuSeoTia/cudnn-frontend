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

## Run

From the review checkout on Linux with a Blackwell SM100-family GPU, a working
CUDA 13-capable driver, Python 3.12 (including `venv`), and `compute-sanitizer`
on `PATH`:

```bash
bash test/python/deepseek_sparse_attention/cutedsl/reproduce_publication.sh
```

The script creates a fresh virtualenv, installs pinned public packages, downloads
and checks the exact NVIDIA main archive, prepares isolated source trees, and
runs all four suites below. Allow internet access and disk space for several GB
of Python/CUDA packages. Compute Sanitizer 2026.3.0.0 was used for qualification.

Results go to a new `dsa-publication-*` directory. Its `reports/` contains the
installation log, resolved packages, source hashes, exact commands, exit codes,
raw logs, JUnit XML and per-run runtime records. The script exits nonzero if any
required check fails or skips. Expected racecheck failures on the unfixed code
count as success only when the numerical tests pass and the reported source
locations match the publication conflict. Hazard counts can vary by GPU.
Each test process has a one-hour limit; a timeout stops qualification and keeps
its logs and receipt.

Optional arguments: `--output /path/to/new-directory`,
`--suite fp32|publication|attention|shared` (default: all), and `--existing-env`
to use an already prepared Python environment. Set `PYTHON` or
`COMPUTE_SANITIZER` to select executables outside `PATH`; `--help` lists options.
`--source-archive` accepts an offline copy of the pinned NVIDIA codeload archive.

## Independent checks

| Suite | Source and acceptance |
| --- | --- |
| `fp32` | 36 existing FP32/BF16, cache-order, packed and compressed cases on pristine main |
| `publication` | Same six cases on main plus the regression test, then on the kernel fix; numerics, racecheck and synccheck |
| `attention` | Two existing dense/packed attention cases on both versions under the same three checks |
| `shared` | 39 existing shared-mainloop cases on the fix, with 44 unrelated cases deselected |

The six publication cases cover BF16/FP32 W, partial tiles, persistent slot reuse
and tiles with no visible keys. They use a chunked FP64 oracle, repeated calls
and changed W. The script checks that the baseline overlay changes only the
test file and that the fixed tree differs only in the kernel file. It verifies
imported package/kernel paths and disables compiled caches in every child.

Pinned runtime: Torch 2.13.0, CuTe DSL 4.8.1 (CUDA 13.4), cuda-bindings 13.4.4,
apache-tvm-ffi 0.1.12, NumPy 1.26.4, cuDNN 9.22.0.52, pytest 8.4.2,
pytest-xdist 3.8.0, looseversion 1.3.0 and certifi 2026.10.23 (source-download
CA bundle, also needed with `--existing-env`). The script explicitly overrides
Torch's cuDNN 9.20 dependency
with `--no-deps` to match the measured runtime. It uses the official Frontend
1.32.1 wheel's native extension with isolated checkout Python sources; it does
not build the C++ extension.

## Recorded source-recreation result

On 2026-11-02, a fresh clone supplied the base archive and the committed
kernel/test changes. The source manifests differed only by the kernel file
between unfixed and fixed arms. GB300 SM103 (152 SMs), runtime above:

| Check | Unfixed | Fixed |
| --- | --- | --- |
| Existing FP32/BF16 support tests | 36 passed | Independent baseline check |
| Publication numerics | 6 passed | 6 passed |
| Publication racecheck | 126 errors, 100 displayed hazards | 0 errors |
| Focused attention numerics | 2 passed | 2 passed |
| Focused attention racecheck | 8 errors | 0 errors |
| Synccheck, publication and focused attention | 0 errors | 0 errors |
| Shared-mainloop suite | Not selected | 39 passed, 44 deselected |

No selected case skipped. These are kernel/API correctness results, not model
accuracy or performance measurements.

The one-command entry point reproduced this table on 2026-11-02 with a fresh
virtualenv and online source download, using only `--output`. All 14 processes
met their acceptance criteria in 1,097 seconds on GB300, with no selected-test
skips. The executed script's SHA256 was
`069b33878c13d26a42775c4329fcddb0e6db06493e3a76ce6b8da9e8458687b4`.
