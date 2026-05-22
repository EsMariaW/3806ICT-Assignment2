from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from planner.driver import plan_and_fill
from prover import config as CFG  # NEW: live switches for premise/context
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _parse_temps(s: Optional[str]) -> Optional[List[float]]:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            raise argparse.ArgumentTypeError(f"Invalid temperature: {p!r}")
    return out or None


def _read_goals_file(path: str) -> List[str]:
    """Read goals one-per-line as UTF-8, stripping a 'lemma \"...\"' wrapper and
    surrounding quotes if present. Blank lines are dropped.

    Reading the file here (rather than piping via the shell) keeps Unicode logic
    symbols (∀, ∧, ∪, ⟶, …) intact — the Windows/PowerShell console pipe corrupts
    them to '?', but an explicit UTF-8 file handle does not.
    """
    goals: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('lemma "') and line.endswith('"'):
                line = line[len('lemma "'):-1].strip()
            elif line.startswith('"') and line.endswith('"') and len(line) >= 2:
                line = line[1:-1].strip()
            if line:
                goals.append(line)
    return goals


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Planner: Plan → Sketch → Fill (Isabelle/HOL)")

    # Accept BOTH a --goal flag and a positional goal (backwards-compatible)
    ap.add_argument("--goal", dest="goal_flag",
                    help='Lemma statement, e.g., map f (xs @ ys) = map f xs @ map f ys')
    ap.add_argument("goal_pos", nargs="?",
                    help='Lemma statement without the --goal flag')

    ap.add_argument("--model", default=None,
                    help="Model id (e.g., 'ollama:qwen2.5:14b', 'hf:meta-llama/...', 'gemini:gemini-2.5-pro')")
    ap.add_argument("--timeout", type=int, default=120,
                    help="Total wall-clock seconds for planning + filling")
    ap.add_argument("--mode", choices=["auto", "outline"], default="auto",
                    help="auto: allow whole proofs; outline: force placeholders and fill")

    # Diverse-outline controls
    ap.add_argument("--diverse-outlines", dest="diverse", action="store_true",
                    help="Enable diverse outline sampling + quick sketch check")
    ap.add_argument("--single-outline", dest="diverse", action="store_false",
                    help="Disable diversity; use a single low-temp outline")
    ap.set_defaults(diverse=True)
    ap.add_argument("--k", type=int, default=3,
                    help="Number of outline candidates (when --diverse-outlines)")
    ap.add_argument("--temps", type=_parse_temps, default=None,
                    help="Comma-separated temps for outline sampling, e.g. '0.35,0.55,0.85'")

    # Local repair controls
    ap.add_argument("--repairs", dest="repairs", action="store_true",
                    help="Enable local LLM-guided repairs if a hole fails to fill (default).")
    ap.add_argument("--no-repairs", dest="repairs", action="store_false",
                    help="Disable local repairs; only try direct fill for each hole.")
    ap.set_defaults(repairs=True)
    ap.add_argument("--max-repairs-per-hole", type=int, default=2,
                    help="Max repair ops to try for each failing hole (default: 2).")
    # Unified tracing: --trace (preferred)
    ap.add_argument("--trace", dest="trace", action="store_true",
                    help="Print planner progress AND repair details.")
    ap.add_argument("--repair-trace", dest="trace", action="store_true",
                    help="(deprecated) Same as --trace.")
    ap.add_argument("--verbose", dest="trace", action="store_true",
                    help="(deprecated) Same as --trace.")

    # Context hints & priors / scoring knobs (all optional; defaults keep old behavior)
    ap.add_argument("--context-hints", dest="context_hints", action="store_true",
                    help="Mine local Isabelle state and feed top facts/defs as outline hints.")
    ap.add_argument("--no-context-hints", dest="context_hints", action="store_false")
    ap.set_defaults(context_hints=False)

    ap.add_argument("--lib-templates", dest="lib_templates", action="store_true",
                    help="Prepend a few tiny library outline templates when they match obvious tokens.")
    ap.add_argument("--no-lib-templates", dest="lib_templates", action="store_false")
    ap.set_defaults(lib_templates=False)

    ap.add_argument("--priors", default=None,
                    help="Optional JSON file of pattern priors / rules.")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Weight on subgoal count (keep ≥1.0).")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="Weight on pattern penalty.")
    ap.add_argument("--gamma", type=float, default=0.2,
                    help="Weight on hint bonus.")

    # micro-RAG hint lexicon (optional)
    ap.add_argument("--hintlex", default=None,
                    help="Path to token→hints JSON (from planner.priors aggregate).")
    ap.add_argument("--hintlex-top", type=int, default=8,
                    help="Max hints to take per token from the lexicon.")

    # === New flags for CEGIS enhancements ===
    # Tiny beam over local repairs (adaptive inside the CEGIS loop).
    ap.add_argument("--beam-k", type=int, default=2,
                    help="Tiny beam size for local repairs (default: 2). Actual beam adapts down to 1 when time is low.")
    # Whole-proof fallback toggle.
    ap.add_argument("--whole-fallback", dest="whole_fallback", action="store_true",
                    help="Allow whole-proof repair as a final fallback when time remains (default: on).")
    ap.add_argument("--no-whole-fallback", dest="whole_fallback", action="store_false",
                    help="Disable whole-proof fallback (skip Baldur-like stage).")
    ap.set_defaults(whole_fallback=True)

    # --- NEW: premise selection & file-aware context controls (default ON) ---
    ap.add_argument("--premises", dest="premises", action="store_true",
                    help="Enable premise retrieval (default).")
    ap.add_argument("--no-premises", dest="premises", action="store_false",
                    help="Disable premise retrieval.")
    ap.set_defaults(premises=True)

    ap.add_argument("--context", dest="context", action="store_true",
                    help="Enable file-aware context window (default).")
    ap.add_argument("--no-context", dest="context", action="store_false",
                    help="Disable file-aware context window.")
    ap.set_defaults(context=True)

    ap.add_argument("--context-files", type=str, default="",
                    help="Space/comma-separated .thy files to seed context (e.g., 'A.thy B.thy' or 'A.thy,B.thy').")

    # --- Batch / file input (reads UTF-8 directly, bypassing terminal encoding) ---
    ap.add_argument("--goals-file", dest="goals_file", default=None,
                    help="Path to a file with one lemma statement per line (UTF-8). "
                         "Without --line, runs the whole file and prints a Success: N/M tally. "
                         "With --line N, proves only that line (full --trace honoured).")
    ap.add_argument("--line", dest="line_no", type=int, default=None,
                    help="1-based line number to pick from --goals-file (proves just that one goal).")
    args = ap.parse_args(argv)

    # Resolve goal: --goals-file (+ optional --line) > flag > positional > stdin
    file_goals: List[str] = []
    if args.goals_file:
        try:
            file_goals = _read_goals_file(args.goals_file)
        except FileNotFoundError:
            print(f"Goals file not found: {args.goals_file}", file=sys.stderr)
            return 2
        except Exception as ex:
            print(f"Could not read goals file {args.goals_file}: {ex}", file=sys.stderr)
            return 2
        if not file_goals:
            print(f"No goals found in {args.goals_file}.", file=sys.stderr)
            return 2

    goal = args.goal_flag or args.goal_pos
    if not goal and not args.goals_file:
        data = sys.stdin.read().strip()
        if data.startswith('lemma "') and data.endswith('"'):
            data = data[len('lemma "'): -1]
        goal = data.strip()

    if not goal and not args.goals_file:
        print("No goal provided. Use --goal '…', a positional goal, --goals-file, or pipe via stdin.", file=sys.stderr)
        return 2

    # Apply NEW premise/context flags to the prover's live config (read inside prove_goal).
    # The prover checks: CFG.PROVER_CONTEXT_ENABLE / CFG.PROVER_CONTEXT_FILES / CFG.PREMISES_ENABLE
    CFG.PREMISES_ENABLE = bool(args.premises)
    CFG.PROVER_CONTEXT_ENABLE = bool(args.context)
    if args.context_files:
        # split on commas and/or whitespace, keep order & drop empties
        raw = args.context_files.replace(",", " ").split()
        CFG.PROVER_CONTEXT_FILES = [s for s in raw if s]

    # NOTE: Current driver hard-codes beam_k=2 and whole_fallback=True when calling CEGIS.
    # We keep the flags here for forward compatibility; warn if users deviate from the current defaults.
    if args.beam_k != 2 or (args.whole_fallback is False):
        print(
            "[cli] Note: driver currently uses beam_k=2 and whole_fallback=True. "
            "These flags are accepted for forward compatibility and will take effect once driver wiring is updated.",
            file=sys.stderr,
        )

    def _run_one(g: str, *, trace: bool):
        return plan_and_fill(
            g,
            model=args.model,
            timeout=args.timeout,
            mode=args.mode,
            outline_k=args.k if args.diverse else 1,
            outline_temps=args.temps,
            legacy_single_outline=(not args.diverse),
            repairs=args.repairs,
            max_repairs_per_hole=args.max_repairs_per_hole,
            trace=trace,
            # planner scoring/context
            priors_path=args.priors,
            context_hints=args.context_hints,
            lib_templates=args.lib_templates,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            # hintlex
            hintlex_path=args.hintlex,
            hintlex_top=args.hintlex_top,
        )

    # ---- Goals-file modes ----
    if args.goals_file:
        # Single-pick: prove one line, full trace, print the proof like single-goal mode.
        if args.line_no is not None:
            if args.line_no < 1 or args.line_no > len(file_goals):
                print(f"--line {args.line_no} out of range (file has {len(file_goals)} goals).", file=sys.stderr)
                return 2
            g = file_goals[args.line_no - 1]
            res = _run_one(g, trace=args.trace)
            print(res.outline, end="" if res.outline.endswith("\n") else "\n")
            if not res.success:
                print("[planner] NOTE: this proof did NOT verify (returned as a failed attempt).", file=sys.stderr)
            return 0 if res.success else 1

        # Batch: compact per-line pass/fail, then a Success: N/M tally.
        n_ok = 0
        total = len(file_goals)
        for i, g in enumerate(file_goals, start=1):
            ok = False
            try:
                res = _run_one(g, trace=False)
                ok = bool(res.success)
            except Exception as ex:
                print(f"[{i}/{total}] CRASH  {type(ex).__name__}: {ex}", file=sys.stderr)
            status = "PASS" if ok else "FAIL"
            print(f"[{i}/{total}] {status}  {g}", flush=True)
            if ok:
                n_ok += 1
        print(f"\nBatch done. Success: {n_ok}/{total} ({100.0 * n_ok / total:.1f}%).", flush=True)
        return 0 if n_ok == total else 1

    # ---- Single-goal mode (unchanged behaviour) ----
    res = _run_one(goal, trace=args.trace)
    if args.trace and ("--verbose" in (argv or sys.argv) or "--repair-trace" in (argv or sys.argv)):
        print("[planner] Note: --verbose/--repair-trace are deprecated; use --trace.", flush=True)
    print(res.outline, end="" if res.outline.endswith("\n") else "\n")
    if not res.success:
        print("[planner] NOTE: this proof did NOT verify (returned as a failed attempt).", file=sys.stderr)
    return 0 if res.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
