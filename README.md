# WePocketModel 1.0-2B Benchmark Suite

A local benchmark runner for a Mac mini M4 with 16 GB unified memory. The benchmark code, monitoring, test fixtures and report generation in this project are original. The actual model runtime and pretrained checkpoint remain Tencent's.

## Fixed in this release

- `python main.py` runs the benchmark suite.
- `python main.py benchmark` runs the same suite.
- All required image, prompt and decision fixtures are generated automatically.
- The harness prefers Tencent's own decision request fixture from the cloned repository when available.
- A single failed test is recorded and the suite continues. It no longer aborts on a missing long-context input.
- Reports omit absolute paths, local usernames, raw commands and filesystem details.
- Every test records process memory, process CPU, system RAM and swap samples.
- macOS telemetry includes memory pressure, VM statistics and thermal status when available.

## Setup on macOS

Use Python 3.10–3.13.

```bash
unzip wepocket-fixed-benchmark.zip
cd wepocket-fixed-benchmark
chmod +x setup.sh
./setup.sh
```

Setup creates `.venv`, installs PyTorch and dependencies, clones Tencent's inference runtime, and downloads the 2B checkpoint into this project.

## Run

```bash
.venv/bin/python main.py
```

or:

```bash
.venv/bin/python main.py benchmark
```

Optional settings:

```bash
.venv/bin/python main.py benchmark --repeats 3 --sustain-runs 3
.venv/bin/python main.py benchmark --threads 4 --long-repetitions 20
```

Do not append words to the command. `benchmarkclear` is not a valid command. Use `benchmark`.

## Tests

1. Text generation.
2. Image understanding with an automatically generated sample PNG.
3. Structured decision using Tencent's own example request where available.
4. Text embeddings at 64, 128, 256, 512, 1024 and 2048 dimensions.
5. Image embedding at 512 dimensions.
6. Repeated generation.
7. Long-input generation.
8. Sequential sustained-load runs.

## Resource metrics

- Inference process-tree peak and average RSS.
- Inference process-tree average and peak CPU utilization.
- System RAM average and peak percentage during each run.
- System swap high-water mark during each run.
- macOS `memory_pressure`, `vm_stat` and `pmset -g thermlog` output when available.
- Hardware architecture, logical/physical CPU count and installed memory.

GPU utilization is not fabricated. macOS GPU telemetry requires separate privileged collection. Exact tokens/second and time-to-first-token are not reported because the documented inference CLI does not expose streaming token timestamps.

## Reports

The suite writes:

- `reports/benchmark-<timestamp>.json`
- `reports/benchmark-<timestamp>.csv`
- `reports/benchmark-<timestamp>.md`

Reports contain no absolute local paths, usernames, raw command lines or filesystem statistics. They include model timing and hardware resource metrics. Failed tests include sanitized diagnostics.

## Interpretation

The official Tencent CLI loads the model, runs one request, writes JSON and exits. Therefore, repeated runs include model startup/loading time. Sustained-load runs are sequential process launches, not concurrent server load.

A PASS means the inference process exited successfully and produced a non-empty output. It does not prove answer correctness. Use a labeled dataset for accuracy testing.

## Upstream

- Tencent runtime: https://github.com/Tencent/WePocketModel
- Model: https://huggingface.co/tencent/WePocketModel1.0-2B
- Inference CLI guide: https://github.com/Tencent/WePocketModel/blob/main/inference/README.md

Review the model-weight terms separately from the runtime repository's Apache-2.0 code license.
