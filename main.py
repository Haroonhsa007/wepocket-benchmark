#!/usr/bin/env python3
"""
WePocketModel 1.0-2B local benchmark runner.

Run:
    python main.py
    python main.py benchmark
    python main.py benchmark --repeats 3 --sustain-runs 3

The benchmark runner and report generation are original project code.
Actual model inference is delegated to Tencent's documented inference CLI.
"""
from __future__ import annotations

import argparse
import binascii
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import struct
import sys
import threading
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:
    raise SystemExit("psutil is missing. Run: python -m pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parent
MODEL_ID = "tencent/WePocketModel1.0-2B"
MODEL_DIR = ROOT / "models" / "WePocketModel1.0-2B"
VENDOR_DIR = ROOT / "vendor" / "WePocketModel"
INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
EMBED_DIMS = (64, 128, 256, 512, 1024, 2048)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_inputs() -> None:
    """Create every local fixture. Benchmark never relies on manually-created files."""
    (INPUTS / "decisions").mkdir(parents=True, exist_ok=True)
    (INPUTS / "images").mkdir(parents=True, exist_ok=True)
    (INPUTS / "prompts").mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    decision_fixtures = {
        "choice.json": {
            "state": "A customer received the wrong product and requests a replacement.",
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "Select the best response for the customer.",
                    "criteria": {
                        "replace": "Apologize and arrange shipment of the correct replacement.",
                        "refund": "Offer a refund.",
                        "ignore": "Ignore the customer request."
                    }
                }
            }
        },
        "score.json": {
            "state": "The agent acknowledged a service outage, gave a status update, and provided a restoration estimate.",
            "questions": {
                "q": {
                    "type": "score",
                    "instructions": "Rate the quality of the support response from 1 to 5.",
                    "criteria": {
                        "1": "Poor, no useful information.",
                        "2": "Limited acknowledgement.",
                        "3": "Adequate but missing next steps.",
                        "4": "Clear update with useful next steps.",
                        "5": "Clear, accurate, complete and actionable update."
                    }
                }
            }
        },
        "binary.json": {
            "state": "The ticket contains a customer email address and a password reset request. It contains no password or authentication token.",
            "questions": {
                "q": {
                    "type": "noul",
                    "instructions": "Does the ticket contain a plaintext password or authentication token?"
                }
            }
        }
    }
    for name, data in decision_fixtures.items():
        path = INPUTS / "decisions" / name
        if not path.exists():
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    long_prompt = INPUTS / "prompts" / "long_context.txt"
    if not long_prompt.exists():
        long_prompt.write_text(
            "This is a controlled long-input test for a local multimodal model. "
            "A fictional company network outage affected office Wi-Fi, internal DNS, "
            "and access to a customer portal. Monitoring detected packet loss at 09:10. "
            "The network team isolated an upstream switch issue, restored connectivity, "
            "checked DNS resolution, and validated application health. Summarize the "
            "incident, user impact, mitigation, and remaining risks without inventing facts.\n",
            encoding="utf-8"
        )

    image_path = INPUTS / "images" / "benchmark_sample.png"
    if not image_path.exists():
        create_sample_png(image_path)


def create_sample_png(path: Path) -> None:
    """Create a deterministic, dependency-free sample image for image tests."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data) & 0xffffffff)

    width, height = 640, 420
    pixels = bytearray()
    for y in range(height):
        pixels.append(0)
        for x in range(width):
            rgb = (245, 247, 250)
            if 100 < x < 540 and 95 < y < 315:
                rgb = (35, 65, 105)
            if 155 < x < 210 and 150 < y < 185:
                rgb = (70, 220, 130)
            if 225 < x < 280 and 150 < y < 185:
                rgb = (250, 185, 55)
            if 295 < x < 350 and 150 < y < 185:
                rgb = (225, 75, 75)
            if 145 < x < 490 and 225 < y < 240:
                rgb = (220, 230, 240)
            if 145 < x < 420 and 255 < y < 270:
                rgb = (220, 230, 240)
            pixels.extend(rgb)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(pixels), 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(data)


def find_runtime(backend: str) -> Path:
    runner = VENDOR_DIR / "inference" / ("infer_cpu.py" if backend == "cpu" else "infer.py")
    if not runner.is_file():
        raise FileNotFoundError(
            "Tencent inference runtime is missing. Run setup.sh to install it."
        )
    return runner


def resolve_model(model_arg: str) -> str:
    if model_arg != "local":
        return model_arg
    if not MODEL_DIR.is_dir() or not any(MODEL_DIR.iterdir()):
        raise FileNotFoundError(
            "Local model checkpoint is missing. Run setup.sh or use --model tencent/WePocketModel1.0-2B."
        )
    return str(MODEL_DIR)


def clean_text(value: str) -> str:
    """Remove absolute local paths from public-facing console/report text."""
    value = value.replace(str(ROOT), "<project>")
    value = re.sub(r'/Users/\S+', '<local-path>', value)
    value = re.sub(r'/home/\S+', '<local-path>', value)
    return value


def mac_telemetry() -> dict[str, Any]:
    result: dict[str, Any] = {}
    if sys.platform != "darwin":
        return result
    commands = {
        "memory_pressure": ["memory_pressure"],
        "vm_stat": ["vm_stat"],
        "thermal_status": ["pmset", "-g", "thermlog"],
    }
    for key, command in commands.items():
        if not shutil.which(command[0]):
            result[key] = None
            continue
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=6)
            result[key] = clean_text((proc.stdout or proc.stderr).strip())[:1800] or None
        except Exception as exc:
            result[key] = f"Unavailable: {type(exc).__name__}"
    # GPU utilization is intentionally not guessed from CPU or memory readings.
    result["gpu_utilization"] = "Not collected. macOS powermetrics requires separate privileged telemetry."
    return result


def system_snapshot() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "platform": platform.system(),
        "os_version": platform.mac_ver()[0] if sys.platform == "darwin" else platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor() or "Not reported",
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_total_gib": round(vm.total / 1024**3, 2),
        "memory_available_gib": round(vm.available / 1024**3, 2),
        "memory_used_gib": round(vm.used / 1024**3, 2),
        "memory_percent": vm.percent,
        "swap_total_gib": round(swap.total / 1024**3, 2),
        "swap_used_gib": round(swap.used / 1024**3, 2),
        "swap_percent": swap.percent,
        "mac_telemetry": mac_telemetry()
    }


def monitor_process(proc: subprocess.Popen, interval: float) -> dict[str, Any]:
    try:
        root_proc = psutil.Process(proc.pid)
    except psutil.Error:
        root_proc = None

    peak_rss = 0
    rss_samples: list[int] = []
    cpu_samples: list[float] = []
    ram_samples: list[float] = []
    swap_samples: list[int] = []
    while proc.poll() is None:
        rss = 0
        cpu = 0.0
        targets = []
        if root_proc:
            targets = [root_proc]
            try:
                targets.extend(root_proc.children(recursive=True))
            except psutil.Error:
                pass
        for item in targets:
            try:
                rss += item.memory_info().rss
                cpu += item.cpu_percent(interval=None)
            except psutil.Error:
                continue
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        peak_rss = max(peak_rss, rss)
        rss_samples.append(rss)
        cpu_samples.append(cpu)
        ram_samples.append(vm.percent)
        swap_samples.append(swap.used)
        time.sleep(interval)

    def mean(values):
        return round(sum(values) / len(values), 2) if values else None

    return {
        "sample_count": len(rss_samples),
        "sample_interval_seconds": interval,
        "process_peak_rss_gib": round(peak_rss / 1024**3, 3),
        "process_average_rss_gib": round((mean(rss_samples) or 0) / 1024**3, 3),
        "process_average_cpu_percent": mean(cpu_samples),
        "process_peak_cpu_percent": round(max(cpu_samples), 2) if cpu_samples else None,
        "system_ram_percent_average": mean(ram_samples),
        "system_ram_percent_peak": max(ram_samples) if ram_samples else None,
        "system_swap_peak_gib": round((max(swap_samples) if swap_samples else 0) / 1024**3, 3)
    }


def command_for(task: str, args, output: Path, prompt: str | None = None,
                image: Path | None = None, decision_file: Path | None = None,
                dimension: int | None = None) -> list[str]:
    command = [
        sys.executable, str(find_runtime(args.backend)),
        "--model", resolve_model(args.model),
        "--task", task,
        "--output", str(output),
        "--timeout", str(args.timeout)
    ]
    if args.backend == "cpu":
        command += ["--threads", str(args.threads)]
    if task == "decision":
        command += ["--decision-json", str(decision_file)]
    else:
        if prompt:
            command += ["--prompt", prompt]
        if image:
            command += ["--image", str(image)]
        if task == "generate":
            command += ["--max-new-tokens", str(args.max_new_tokens)]
        if task == "embed" and dimension:
            command += ["--embedding-dim", str(dimension)]
    return command


def run_one(label: str, task: str, args, prompt: str | None = None,
            image: Path | None = None, decision_file: Path | None = None,
            dimension: int | None = None) -> dict[str, Any]:
    output = OUTPUTS / f"{timestamp()}-{label}.json"
    command = command_for(task, args, output, prompt, image, decision_file, dimension)
    before = system_snapshot()
    started = time.perf_counter()
    try:
        proc = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env={**os.environ, "TOKENIZERS_PARALLELISM": "false"}
        )
        metrics = monitor_process(proc, args.sample_interval)
        stdout, stderr = proc.communicate()
        elapsed = time.perf_counter() - started
        valid = output.is_file() and output.stat().st_size > 0
        status = "PASS" if proc.returncode == 0 and valid else "FAIL"
        detail = {
            "test": label, "task": task, "status": status,
            "elapsed_seconds": round(elapsed, 3),
            "return_code": proc.returncode,
            "output_size_bytes": output.stat().st_size if valid else 0,
            "resource_metrics": metrics,
            "system_before": before,
            "system_after": system_snapshot(),
            "stdout_excerpt": clean_text(stdout[-1600:]),
            "stderr_excerpt": clean_text(stderr[-2500:]),
            "failure_reason": None if status == "PASS" else "Inference returned an error or did not produce a non-empty JSON result."
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        detail = {
            "test": label, "task": task, "status": "FAIL",
            "elapsed_seconds": round(elapsed, 3),
            "return_code": None, "output_size_bytes": 0,
            "resource_metrics": {},
            "system_before": before, "system_after": system_snapshot(),
            "stdout_excerpt": "", "stderr_excerpt": "",
            "failure_reason": f"{type(exc).__name__}: {clean_text(str(exc))}"
        }
    return detail


def official_decision_fixture() -> Path:
    """Prefer a request shipped with Tencent's cloned examples if present."""
    candidates = [
        VENDOR_DIR / "examples" / "cpu" / "demo_data" / "decision" / "request.json",
        VENDOR_DIR / "examples" / "gpu" / "demo_data" / "decision" / "request.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # If the repository layout changes, search only the official examples tree.
    examples = VENDOR_DIR / "examples"
    if examples.is_dir():
        found = list(examples.glob("**/decision/request.json"))
        if found:
            return found[0]
    return INPUTS / "decisions" / "choice.json"


def run_benchmark(args) -> dict[str, Any]:
    ensure_inputs()
    if not VENDOR_DIR.exists():
        raise FileNotFoundError("Tencent runtime missing. Run setup.sh first.")
    if args.model == "local":
        resolve_model(args.model)

    tests: list[dict[str, Any]] = []

    def execute(label, task, **kwargs):
        print(f"\n[RUN] {label}", flush=True)
        result = run_one(label, task, args, **kwargs)
        tests.append(result)
        print(f"[{result['status']}] {label}: {result['elapsed_seconds']:.2f}s", flush=True)
        if result["status"] == "FAIL":
            print("  " + (result.get("failure_reason") or "See report diagnostics."), flush=True)

    # Functional tests
    execute("text_generation", "generate",
            prompt="Answer in one sentence: what is semantic search?")
    execute("image_understanding", "generate",
            prompt="Describe the main object in this image and the visible indicator lights.",
            image=INPUTS / "images" / "benchmark_sample.png")

    decision_path = official_decision_fixture()
    execute("structured_decision", "decision", decision_file=decision_path)

    # Embedding dimension sweep
    for dim in EMBED_DIMS:
        execute(f"text_embedding_{dim}d", "embed",
                prompt="A network router is offline and office users cannot access internal services.",
                dimension=dim)

    execute("image_embedding_512d", "embed",
            prompt="Create an embedding for semantic retrieval of this image.",
            image=INPUTS / "images" / "benchmark_sample.png", dimension=512)

    # Repeated inference, cold process starts. This includes model load time.
    for index in range(1, args.repeats + 1):
        execute(f"repeated_generation_{index}", "generate",
                prompt="Explain three uses of a vector database.")

    # Long input test. The harness creates the prompt fixture automatically.
    base_prompt = (INPUTS / "prompts" / "long_context.txt").read_text(encoding="utf-8")
    long_prompt = "\n".join([base_prompt] * args.long_repetitions)
    execute("long_input_generation", "generate", prompt=long_prompt)

    # Sustained sequential load test, not concurrent load.
    for index in range(1, args.sustain_runs + 1):
        execute(f"sustained_load_run_{index}", "generate",
                prompt="Summarize the purpose of CPU, RAM and swap monitoring in local AI inference.")

    passed = sum(test["status"] == "PASS" for test in tests)
    failed = len(tests) - passed
    total_time = round(sum(test["elapsed_seconds"] for test in tests), 2)
    hardware = system_snapshot()
    created = datetime.now(timezone.utc).isoformat()
    report = {
        "report_title": "WePocketModel 1.0-2B Local Benchmark",
        "created_utc": created,
        "model": MODEL_ID if args.model == "local" else args.model,
        "backend": args.backend,
        "hardware": hardware,
        "configuration": {
            "cpu_threads": args.threads,
            "max_new_tokens": args.max_new_tokens,
            "repeat_count": args.repeats,
            "sustained_runs": args.sustain_runs,
            "long_input_repetitions": args.long_repetitions,
            "resource_sample_interval_seconds": args.sample_interval,
            "embedding_dimensions_tested": list(EMBED_DIMS)
        },
        "summary": {
            "total_tests": len(tests), "passed": passed, "failed": failed,
            "pass_rate_percent": round(passed * 100 / len(tests), 1) if tests else 0,
            "total_inference_time_seconds": total_time
        },
        "tests": tests,
        "measurement_notes": [
            "Each official CLI invocation loads the model, performs one request, writes a result and exits. Repeated tests therefore include cold model startup/loading overhead.",
            "Process RSS is the measured inference process tree resident set. On Apple Silicon it is not an exact attribution of physical unified memory.",
            "System RAM and swap are sampled during each inference process.",
            "macOS memory_pressure, vm_stat and thermal status are captured where the operating system exposes them.",
            "GPU utilization is not reported unless separately collected through privileged macOS telemetry.",
            "The official CLI does not expose streaming token timestamps. Exact time-to-first-token and tokens-per-second are omitted rather than estimated.",
            "A PASS means the inference command completed and produced a non-empty result file. It does not certify answer accuracy.",
            "The long-input test uses a configurable repeated prompt. It does not claim to fill the model's maximum context window.",
            "Sustained-load tests are sequential process launches, not concurrent requests."
        ]
    }

    stamp = timestamp()
    json_path = REPORTS / f"benchmark-{stamp}.json"
    csv_path = REPORTS / f"benchmark-{stamp}.csv"
    md_path = REPORTS / f"benchmark-{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_fields = [
        "test", "task", "status", "elapsed_seconds", "process_peak_rss_gib",
        "process_average_rss_gib", "process_average_cpu_percent",
        "process_peak_cpu_percent", "system_ram_percent_average",
        "system_ram_percent_peak", "system_swap_peak_gib", "return_code"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for test in tests:
            metrics = test.get("resource_metrics", {})
            writer.writerow({
                "test": test["test"], "task": test["task"], "status": test["status"],
                "elapsed_seconds": test["elapsed_seconds"],
                "process_peak_rss_gib": metrics.get("process_peak_rss_gib"),
                "process_average_rss_gib": metrics.get("process_average_rss_gib"),
                "process_average_cpu_percent": metrics.get("process_average_cpu_percent"),
                "process_peak_cpu_percent": metrics.get("process_peak_cpu_percent"),
                "system_ram_percent_average": metrics.get("system_ram_percent_average"),
                "system_ram_percent_peak": metrics.get("system_ram_percent_peak"),
                "system_swap_peak_gib": metrics.get("system_swap_peak_gib"),
                "return_code": test.get("return_code")
            })

    lines = [
        "# WePocketModel 1.0-2B Local Benchmark", "",
        f"- Run time (UTC): {created}",
        f"- Hardware: {hardware['architecture']}, {hardware['logical_cpu_count']} logical CPUs",
        f"- Installed memory: {hardware['memory_total_gib']} GiB",
        f"- Backend: {args.backend}",
        f"- Tests: {len(tests)} | PASS: {passed} | FAIL: {failed}",
        f"- Pass rate: {report['summary']['pass_rate_percent']}%",
        f"- Total inference time: {total_time} seconds", "",
        "| Test | Status | Time (s) | Peak process RAM (GiB) | Avg CPU (%) | Peak CPU (%) | Avg system RAM (%) | Peak swap (GiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for test in tests:
        m = test.get("resource_metrics", {})
        lines.append(
            f"| {test['test']} | {test['status']} | {test['elapsed_seconds']:.2f} | "
            f"{m.get('process_peak_rss_gib', 'n/a')} | "
            f"{m.get('process_average_cpu_percent', 'n/a')} | "
            f"{m.get('process_peak_cpu_percent', 'n/a')} | "
            f"{m.get('system_ram_percent_average', 'n/a')} | "
            f"{m.get('system_swap_peak_gib', 'n/a')} |"
        )
    lines += ["", "## Measurement notes", ""]
    lines += [f"- {note}" for note in report["measurement_notes"]]
    lines += ["", "## Test diagnostics", ""]
    for test in tests:
        if test["status"] == "FAIL":
            lines += [f"### {test['test']}", "", f"- Reason: {test.get('failure_reason')}",
                      f"- Error: {test.get('stderr_excerpt', '')}", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n========== BENCHMARK SUMMARY ==========")
    for test in tests:
        m = test.get("resource_metrics", {})
        print(
            f"{test['status']:4} {test['test']:28} "
            f"{test['elapsed_seconds']:8.2f}s "
            f"RAM={m.get('process_peak_rss_gib', 'n/a')}GiB "
            f"CPU={m.get('process_average_cpu_percent', 'n/a')}%"
        )
    print("\nReports created:")
    print(f"  JSON: reports/{json_path.name}")
    print(f"  CSV:  reports/{csv_path.name}")
    print(f"  MD:   reports/{md_path.name}")
    print("No absolute local paths are included in the reports.")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="WePocketModel local benchmark suite")
    parser.add_argument("command", nargs="?", choices=["benchmark"], default="benchmark",
                        help="Benchmark suite. Default when omitted.")
    parser.add_argument("--model", default="local",
                        help="local checkpoint, or Hugging Face model ID")
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=max(1, min(6, os.cpu_count() or 4)))
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--sustain-runs", type=int, default=2)
    parser.add_argument("--long-repetitions", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.threads < 1 or args.timeout < 1 or args.sample_interval <= 0:
        raise SystemExit("threads, timeout and sample interval must be positive.")
    if not 1 <= args.max_new_tokens <= 512:
        raise SystemExit("--max-new-tokens must be between 1 and 512.")
    if args.repeats < 0 or args.sustain_runs < 0 or args.long_repetitions < 1:
        raise SystemExit("repeat counts must be non-negative and long repetitions at least 1.")
    ensure_inputs()
    report = run_benchmark(args)
    raise SystemExit(0 if report["summary"]["failed"] == 0 else 2)


if __name__ == "__main__":
    main()
