# results_summary.py

A standalone performance summary report tool for the Isabellm LLM theorem prover.

Reads `logs/runs.log.jsonl` and prints a formatted report including hardware specs,
LLM backend info, proof success rates, timing statistics, and per-goal breakdowns.

---

## Dependencies

```bash
pip install psutil
```

`psutil` is used for RAM detection. If not installed, RAM will show as `Unknown`
but the rest of the report will work correctly without it.

---

## Usage

```bash
# Basic summary (hardware + model + results)
python results_summary.py

# Show per-goal pass/fail breakdown
python results_summary.py --goals

# Side-by-side model comparison (when multiple models used)
python results_summary.py --compare

# Show all sections at once
python results_summary.py --all

# Only summarise the last 20 runs
python results_summary.py --last 20

# Filter to a specific model
python results_summary.py --model deepseek-r1:8b
python results_summary.py --model gemini

# Use a different log file
python results_summary.py --log logs/planner.log.jsonl
```

---

## Output Sections

### Hardware & Environment
Dynamically detects and displays:
- OS and architecture
- Python version
- CPU model (Windows registry / macOS sysctl / Linux /proc/cpuinfo)
- Total system RAM (via psutil, falls back to /proc/meminfo on Linux)
- GPU (NVIDIA via nvidia-smi, Apple Silicon via system_profiler)

### LLM Backend
Detects which backend is being used from the model string:
- `gemini:model-name` → Gemini API
- `hf:repo/name` → HuggingFace API
- `model-name` or `ollama:model-name` → Ollama local
- Also checks if the Ollama server is currently running

### Results Summary
- Total runs and unique goals attempted
- Successful vs failed proof counts and percentages
- Timing statistics (median, mean, min, max, separated by success/failure)
- Search depth statistics
- Total and average Isabelle theory call counts
- Date range of runs

### Per-Goal Breakdown (--goals)
Shows each unique goal with:
- Pass/Fail status
- Number of attempts
- Success rate
- Average time per attempt

### Model Comparison (--compare)
Side-by-side table when runs from multiple models are present:
- Success rate per model
- Median time per model
- Average search depth per model

---

## Notes

- The log file path defaults to `logs/runs.log.jsonl`
- Planner runs are logged separately in `logs/planner.log.jsonl` if it exists
- All timestamps in the log are Unix epoch; the report converts them to readable dates
- Run the script from the project root directory
