"""
results_summary.py
==================
Reads logs/runs.log.jsonl and prints a formatted performance summary report.

Usage:
    python results_summary.py                        # summarise all runs
    python results_summary.py --log logs/runs.log.jsonl
    python results_summary.py --model deepseek-r1:8b # filter by model
    python results_summary.py --last 20              # last N runs only
    python results_summary.py --compare              # compare models side by side

Dependencies:
    pip install psutil
    psutil is used for RAM detection. If not installed, RAM will show as "Unknown"
    but the rest of the report will work correctly without it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Hardware detection
# ─────────────────────────────────────────────────────────────────────────────

def _get_cpu_info() -> str:
    """Best-effort CPU name across platforms."""
    system = platform.system()
    try:
        if system == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return name.strip()
        elif system == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL, text=True
            )
            return out.strip()
        elif system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def _get_ram_gb() -> str:
    """Total system RAM in GB."""
    try:
        import psutil
        return f"{psutil.virtual_memory().total / 1024**3:.1f} GB"
    except ImportError:
        pass
    # Fallback for Linux
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return f"{kb / 1024**2:.1f} GB"
    except Exception:
        pass
    return "Unknown"


def _get_gpu_info() -> str:
    """Best-effort GPU name."""
    # Try nvidia-smi first (works on Windows/Linux with NVIDIA GPU)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True
        )
        gpus = [g.strip() for g in out.strip().splitlines() if g.strip()]
        if gpus:
            return ", ".join(gpus)
    except Exception:
        pass
    # Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                stderr=subprocess.DEVNULL, text=True
            )
            for line in out.splitlines():
                if "Chipset Model" in line or "Chip" in line:
                    return line.split(":", 1)[1].strip() + " (Metal/MPS)"
        except Exception:
            pass
        return "Apple Silicon GPU (MPS)"
    return "No GPU detected / CPU only"


def _get_ollama_info(model: str) -> str:
    """Check if model is served via Ollama and get info."""
    if not model:
        return ""
    # Check if it looks like an Ollama model (no prefix)
    if model.startswith("gemini:") or model.startswith("hf:"):
        return ""
    try:
        import urllib.request
        url = "http://127.0.0.1:11434/api/tags"
        with urllib.request.urlopen(url, timeout=2) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            if any(model in m for m in models):
                return "Ollama (local)"
    except Exception:
        pass
    return "Ollama (local, server not running)"


def _detect_backend(model: str) -> str:
    """Detect which backend is being used from the model string."""
    if not model:
        return "Unknown"
    if model.startswith("gemini:"):
        return f"Gemini API ({model[7:]})"
    if model.startswith("hf:"):
        return f"HuggingFace API ({model[3:]})"
    if model.startswith("ollama:"):
        return f"Ollama local ({model[7:]})"
    return f"Ollama local ({model})"


# ─────────────────────────────────────────────────────────────────────────────
# Log reading
# ─────────────────────────────────────────────────────────────────────────────

def load_runs(log_path: str, model_filter: Optional[str] = None,
              last_n: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load and optionally filter run records from JSONL log."""
    path = Path(log_path)
    if not path.exists():
        print(f"[ERROR] Log file not found: {log_path}")
        sys.exit(1)

    runs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if model_filter:
        runs = [r for r in runs if model_filter.lower() in r.get("model", "").lower()]

    if last_n:
        runs = runs[-last_n:]

    return runs


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pct(part: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{100 * part / total:.1f}%"


def compute_stats(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics from a list of run records."""
    total = len(runs)
    if total == 0:
        return {}

    successes = [r for r in runs if r.get("success")]
    failures  = [r for r in runs if not r.get("success")]

    times_all     = [r["elapsed_s"] for r in runs if "elapsed_s" in r]
    times_success = [r["elapsed_s"] for r in successes if "elapsed_s" in r]
    times_failure = [r["elapsed_s"] for r in failures  if "elapsed_s" in r]

    # Unique goals
    goals = [r.get("goal", "") for r in runs]
    unique_goals = set(goals)

    # Depth distribution
    depths = [r.get("depth_reached", 0) for r in runs]

    # Timestamp range
    timestamps = sorted([r["ts"] for r in runs if "ts" in r])
    first_run = datetime.fromtimestamp(timestamps[0]).strftime("%Y-%m-%d %H:%M") if timestamps else "?"
    last_run  = datetime.fromtimestamp(timestamps[-1]).strftime("%Y-%m-%d %H:%M") if timestamps else "?"

    # Models used
    models_used = list({r.get("model", "unknown") for r in runs})

    # Isabelle call counts
    isa_calls = [r.get("use_theories_calls", 0) for r in runs]

    return {
        "total":           total,
        "n_success":       len(successes),
        "n_failure":       len(failures),
        "success_rate":    _pct(len(successes), total),
        "failure_rate":    _pct(len(failures), total),
        "unique_goals":    len(unique_goals),
        "median_time_all":     _median(times_all),
        "mean_time_all":       _mean(times_all),
        "median_time_success": _median(times_success),
        "median_time_failure": _median(times_failure),
        "min_time":        min(times_all) if times_all else 0,
        "max_time":        max(times_all) if times_all else 0,
        "avg_depth":       _mean(depths),
        "max_depth_seen":  max(depths) if depths else 0,
        "models_used":     models_used,
        "first_run":       first_run,
        "last_run":        last_run,
        "total_isa_calls": sum(isa_calls),
        "avg_isa_calls":   _mean(isa_calls),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────

SEP  = "=" * 68
SEP2 = "-" * 68


def _row(label: str, value: Any, width: int = 30) -> str:
    return f"  {label:<{width}} {value}"


def print_hardware_section() -> None:
    print(SEP)
    print("  HARDWARE & ENVIRONMENT")
    print(SEP2)
    print(_row("OS:",          f"{platform.system()} {platform.release()} ({platform.machine()})"))
    print(_row("Python:",      platform.python_version()))
    print(_row("CPU:",         _get_cpu_info()))
    print(_row("RAM:",         _get_ram_gb()))
    print(_row("GPU:",         _get_gpu_info()))
    print()


def print_model_section(models_used: List[str]) -> None:
    print(SEP)
    print("  LLM BACKEND")
    print(SEP2)
    for m in models_used:
        backend = _detect_backend(m)
        ollama_status = _get_ollama_info(m)
        print(_row("Model:", m))
        print(_row("Backend:", backend))
        if ollama_status:
            print(_row("Ollama status:", ollama_status))
    print()


def print_stats_section(stats: Dict[str, Any], title: str = "RESULTS SUMMARY") -> None:
    print(SEP)
    print(f"  {title}")
    print(SEP2)
    print(_row("Total runs:",          stats["total"]))
    print(_row("Unique goals:",        stats["unique_goals"]))
    print(_row("Successful proofs:",   f"{stats['n_success']}  ({stats['success_rate']})"))
    print(_row("Failed proofs:",       f"{stats['n_failure']}  ({stats['failure_rate']})"))
    print()
    print(_row("Median time (all):",   f"{stats['median_time_all']:.1f}s"))
    print(_row("Mean time (all):",     f"{stats['mean_time_all']:.1f}s"))
    print(_row("Median time (success):", f"{stats['median_time_success']:.1f}s"))
    print(_row("Median time (failure):", f"{stats['median_time_failure']:.1f}s"))
    print(_row("Fastest run:",         f"{stats['min_time']:.1f}s"))
    print(_row("Slowest run:",         f"{stats['max_time']:.1f}s"))
    print()
    print(_row("Avg search depth:",   f"{stats['avg_depth']:.1f}"))
    print(_row("Max depth reached:",  stats["max_depth_seen"]))
    print(_row("Total Isabelle calls:", stats["total_isa_calls"]))
    print(_row("Avg Isabelle calls:", f"{stats['avg_isa_calls']:.1f}"))
    print()
    total_time_s = stats.get("mean_time_all", 0) * stats.get("total", 0)
    h = int(total_time_s // 3600)
    m = int((total_time_s % 3600) // 60)
    s = int(total_time_s % 60)
    total_time_str = f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s"
    print(_row("Total time (all runs):", total_time_str))
    print()


def print_goal_breakdown(runs: List[Dict[str, Any]]) -> None:
    """Show per-goal success/failure breakdown."""
    print(SEP)
    print("  PER-GOAL BREAKDOWN")
    print(SEP2)

    by_goal: Dict[str, List[Dict]] = defaultdict(list)
    for r in runs:
        by_goal[r.get("goal", "unknown")].append(r)

    # Sort by success rate descending
    goal_stats = []
    for goal, goal_runs in by_goal.items():
        n = len(goal_runs)
        s = sum(1 for r in goal_runs if r.get("success"))
        avg_t = _mean([r.get("elapsed_s", 0) for r in goal_runs])
        goal_stats.append((goal, n, s, avg_t))
    goal_stats.sort(key=lambda x: (-x[2]/x[1], x[0]))

    for goal, n, s, avg_t in goal_stats:
        status = "PASS" if s > 0 else "FAIL"
        short_goal = goal[:45] + "..." if len(goal) > 45 else goal
        print(f"  [{status}] {short_goal}")
        print(f"         attempts={n}  solved={s}  ({_pct(s,n)})  avg={avg_t:.1f}s")
    print()


def print_model_comparison(runs: List[Dict[str, Any]]) -> None:
    """Side-by-side comparison when multiple models were used."""
    models = list({r.get("model", "unknown") for r in runs})
    if len(models) < 2:
        return

    print(SEP)
    print("  MODEL COMPARISON")
    print(SEP2)

    for model in sorted(models):
        model_runs = [r for r in runs if r.get("model") == model]
        stats = compute_stats(model_runs)
        if not stats:
            continue
        print(f"  {model}")
        print(f"    Runs:         {stats['total']}")
        print(f"    Success rate: {stats['success_rate']}")
        print(f"    Median time:  {stats['median_time_all']:.1f}s")
        print(f"    Avg depth:    {stats['avg_depth']:.1f}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Print a performance summary from runs.log.jsonl"
    )
    ap.add_argument("--log",     default="logs/runs.log.jsonl",
                    help="Path to JSONL log file (default: logs/runs.log.jsonl)")
    ap.add_argument("--model",   default=None,
                    help="Filter to runs using this model name (substring match)")
    ap.add_argument("--last",    type=int, default=None,
                    help="Only summarise the last N runs")
    ap.add_argument("--compare", action="store_true",
                    help="Show side-by-side model comparison")
    ap.add_argument("--goals",   action="store_true",
                    help="Show per-goal breakdown")
    ap.add_argument("--all",     action="store_true",
                    help="Show all sections (hardware, goals, comparison)")
    args = ap.parse_args()

    runs = load_runs(args.log, model_filter=args.model, last_n=args.last)

    if not runs:
        print("[INFO] No runs found matching the given filters.")
        sys.exit(0)

    stats = compute_stats(runs)

    print()
    print(SEP)
    print("  ISABELLM — RESULTS SUMMARY REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP)
    print()

    # Always show hardware
    print_hardware_section()

    # Always show model info
    print_model_section(stats.get("models_used", []))

    # Always show overall stats
    print_stats_section(stats)

    # Optional sections
    if args.goals or args.all:
        print_goal_breakdown(runs)

    if args.compare or args.all:
        print_model_comparison(runs)

    print(SEP)
    print("  END OF REPORT")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
