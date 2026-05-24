import re
import os
import json
import requests
from typing import Dict, List, Optional, Tuple, Set, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
from prover.isabelle_api import build_theory, run_theory, last_print_state_block, finished_ok

# ========== Configuration ==========
_ISA_FAST_TIMEOUT_S = int(os.getenv("ISABELLE_FAST_TIMEOUT_S", "12"))
_ISA_VERIFY_TIMEOUT_S = int(os.getenv("ISABELLE_VERIFY_TIMEOUT_S", "30"))
_SESSION = requests.Session()

# ========== Regex Patterns ==========
_HEADER_RE = re.compile(r"^\s*(proof\s*\(|proof\b|case\s+|then\s+show\b)")
_APPLY_OR_BY = re.compile(r"^\s*(apply|by)\b")

def _clamp_line_index(lines: List[str], idx: int) -> int:
    if not lines:
        return -1
    return max(0, min(idx, len(lines) - 1))

def _run_theory_with_timeout(isabelle, session: str, thy: List[str], *, timeout_s: Optional[int]) -> List:
    if not timeout_s or timeout_s <= 0:
        return run_theory(isabelle, session, thy)
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(run_theory, isabelle, session, thy)
        try:
            return fut.result(timeout=timeout_s)
        except _FuturesTimeout:
            if hasattr(isabelle, "interrupt"):
                try:
                    isabelle.interrupt()
                except Exception:
                    pass
            raise TimeoutError("isabelle_run_timeout")

def _earliest_failure_anchor(isabelle, session: str, full_text: str, *, default_line_0: int) -> Tuple[int, str]:
    try:
        lines = full_text.splitlines()
        _, errs = _quick_state_and_errors(isabelle, session, full_text)
        err_lines = sorted(_extract_error_lines(errs))
        if err_lines:
            pos0 = err_lines[0] - 1
            if 0 <= pos0 < len(lines):
                return pos0, "error_line"
            for i, L in enumerate(lines):
                if "sorry" in L:
                    return i, "first_sorry_from_error"
            return _nearest_structural_head_before(lines, len(lines) - 1), "error_line_out_of_range"
        thy = build_theory(lines, add_print_state=False, end_with=None)
        ok, _ = finished_ok(_run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S))
        if not ok:
            for i, L in enumerate(lines):
                if "sorry" in L:
                    return i, "first_sorry"
        return default_line_0, "default"
    except Exception:
        return default_line_0, "default"

def _nearest_structural_head_before(lines: List[str], idx: int) -> int:
    if not lines:
        return -1
    i = _clamp_line_index(lines, idx)
    head_re = re.compile(r"^\s*(?:have|show|obtain|case\b|proof\b)\b")
    for j in range(i, -1, -1):
        if head_re.match(lines[j]):
            return j
    return i

# ========== Isabelle Interaction ==========

def _extract_print_state_from_responses(resps: List) -> str:
    standard = last_print_state_block(resps) or ""
    llm_lines: List[str] = []
    for resp in (resps or []):
        if str(getattr(resp, "response_type", "")).upper() != "FINISHED":
            continue
        body = getattr(resp, "response_body", None)
        # Fix: handle non-string response types
        if body is None:
            continue
        if isinstance(body, (bytes, bytearray)):
            body = body.decode(errors="replace")
        elif not isinstance(body, str):
            try:
                body = str(body)
            except Exception:
                continue
        # body is now guaranteed to be a string
        try:
            data = json.loads(body) if isinstance(body, str) and body.strip().startswith("{") else None
            if not isinstance(data, dict):
                continue
            for node in data.get("nodes", []):
                for msg in node.get("messages", []):
                    if msg.get("kind") != "writeln":
                        continue
                    text = msg.get("message", "")
                    if text.startswith(("[LLM_SUBGOAL]", "[LLM_VARS]")):
                        llm_lines.append(text)
                    elif "goal" in text and "subgoal" in text and not standard:
                        standard = text
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    if llm_lines and standard:
        return standard + "\n" + "\n".join(llm_lines)
    return standard or "\n".join(llm_lines)

def _quick_state_and_errors(isabelle, session: str, full_text: str) -> Tuple[str, List[dict]]:
    try:
        thy = build_theory(full_text.splitlines(), add_print_state=True, end_with=None)
        resps = _run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_FAST_TIMEOUT_S)
        state = _extract_print_state_from_responses(resps)
        errors: List[dict] = []
        
        for r in resps or []:
            # Fix: handle non-string response types
            raw = getattr(r, "response_body", None)
            if raw is None:
                continue
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode(errors="replace")
            elif not isinstance(raw, str):
                try:
                    raw = str(raw)
                except Exception:
                    continue
            
            # Try structured JSON first
            try:
                data = json.loads(raw) if isinstance(raw, str) and raw.strip() else None
                if isinstance(data, dict):
                    for node in data.get("nodes", []):
                        for msg in node.get("messages", []):
                            if str(msg.get("kind", "")).lower() == "error":
                                txt = str(msg.get("message", "") or "").strip()
                                if txt:
                                    errors.append({"text": txt, "line": msg.get("line")})
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
            
            # Fallback: raw text parsing for error markers
            if isinstance(raw, str):
                for pattern in ["*** Error:", "*** Outer syntax error", "*** Failed"]:
                    if pattern in raw:
                        for line in raw.split('\n'):
                            if pattern in line:
                                errors.append({"text": line.strip()})
                                break
        
        # Deduplicate by text
        seen = set()
        deduped = []
        for e in errors:
            txt = e.get("text", "")
            if txt and txt not in seen:
                seen.add(txt)
                deduped.append(e)
        
        return state, deduped[:5]
    except Exception as e:
        return "", [{"text": f"extraction_error: {type(e).__name__}"}]

def _print_state_before_hole(isabelle, session: str, full_text: str, hole_span: Tuple[int, int], trace: bool = False) -> str:
    hole_line, indent, lines = _hole_line_bounds(full_text, hole_span)
    if not (0 <= hole_line < len(lines) and "sorry" in lines[hole_line]):
        nearest = _find_first_hole(lines)
        if nearest is not None:
            hole_line = nearest
            indent = len(lines[hole_line]) - len(lines[hole_line].lstrip(" "))
    pad = " " * max(2, indent)
    injected = [f"{pad}prefer 1", f"{pad}print_state", f"{pad}(* REPAIR-PRINT-STATE *)"]
    variant_lines = lines[:hole_line] + injected + lines[hole_line:]
    variant = "\n".join(variant_lines) + ("\n" if full_text.endswith("\n") else "")
    try:
        thy = build_theory(variant.splitlines(), add_print_state=False, end_with=None)
        resps = _run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_FAST_TIMEOUT_S)
        return _extract_print_state_from_responses(resps)
    except Exception:
        return ""

# ========== Counterexample Hints ==========

def _normalize_isabelle_symbols(s: str) -> str:
    """Make Isabelle symbols friendlier in short snippets (e.g., a\<^sub>1 → a_1)."""
    if not s:
        return s
    # subscripts: x\<^sub>12 → x_12
    s = re.sub(r"\\<\^sub>(\d+)", lambda m: "_" + m.group(1), s)
    # superscripts: x\<^sup>2 → x^2
    s = re.sub(r"\\<\^sup>(\d+)", lambda m: "^" + m.group(1), s)
    # collapse extra spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _extract_nitpick_text_from_responses(resps_text: str) -> str:
    """Extract Nitpick/Quickcheck output from mixed JSON/plain-text logs."""
    if not resps_text:
        return ""
    messages: List[str] = []
    for line in resps_text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if s.startswith("{"):
            try:
                if isinstance(s, str) and s.strip().startswith("{"):
                    data = json.loads(s)
                    if isinstance(data, dict):
                        msg = data.get("message")
                        if isinstance(msg, str) and msg.strip():
                            messages.append(msg)
                        for node in data.get("nodes", []):
                            for m in node.get("messages", []):
                                if isinstance(m, dict):
                                    t = m.get("message", "")
                                    if isinstance(t, str) and t:
                                        messages.append(t)
                        for err in data.get("errors", []):
                            if isinstance(err, dict):
                                t = err.get("message", "")
                                if isinstance(t, str) and t:
                                    messages.append(t)
                continue
            except json.JSONDecodeError:
                pass
        # plain text fallback
        messages.append(s)
    return "\n".join(messages)

# --- helpers for parsing bindings ---
_BINDING_RE = re.compile(
    # var = value (value may include type annotation :: ...; stop before comma/semicolon/paren or EOL)
    r"\b([A-Za-z][A-Za-z0-9_']*)\s*=\s*([^,;\)\n]+)")
_TRAIL_PUNCT_RE = re.compile(r"[:\s]+$")

_CEX_MARKERS = (
    "nitpick found a counterexample",
    "nitpick found a potential counterexample",
    "quickcheck found a counterexample",
    "quickcheck found a potential counterexample",
    # extra robustness
    "counterexample:",
    "counterexample found",
    "falsified",
)

def _nitpick_state_hints_from_text(text: str) -> Dict[str, List[str]]:
    """Extract counterexample info; stitch multi-line values; normalize symbols."""
    import sys
    if not text:
        return {"bindings": [], "def_hints": []}

    extracted_text = _extract_nitpick_text_from_responses(text) or text
    t_lower = extracted_text.lower()
    has_cex = any(m in t_lower for m in _CEX_MARKERS)
    if not has_cex:
        return {"bindings": [], "def_hints": []}

    #print("[DEBUG] Found counterexample in output", file=sys.stderr)

    # Prefer the region starting at the first marker
    start_idx = min((t_lower.find(m) for m in _CEX_MARKERS if m in t_lower), default=-1)
    scan_region = extracted_text[start_idx:] if start_idx >= 0 else extracted_text

    lines = scan_region.split("\n")
    bindings: List[str] = []
    seen_vars: Set[str] = set()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        m = _BINDING_RE.search(line)
        if not m:
            i += 1
            continue
        var, head = m.groups()
        val = head

        # stitch continuation if we started a composite value
        opener = val.lstrip()[:1]
        closer = {"[": "]", "(": ")", "{": "}"} .get(opener)
        if closer and closer not in val:
            j = i + 1
            while j < len(lines) and not re.match(r"^\s*\w+\s*=\s*", lines[j]):
                piece = lines[j].strip()
                if piece:
                    val += " " + piece
                if closer in lines[j]:
                    j += 1
                    break
                j += 1
            i = j
        else:
            i += 1

        val = _TRAIL_PUNCT_RE.sub('', val.strip())
        var_n = _normalize_isabelle_symbols(var)
        val_n = _normalize_isabelle_symbols(val)
        if var_n not in {"line", "offset", "file", "kind", "message"} and var_n not in seen_vars:
            seen_vars.add(var_n)
            bindings.append(f"{var_n} = {val_n}")
        if len(bindings) >= 8:
            break

    # Fallback: scan whole text if nothing found
    if not bindings:
        for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9_']*)\s*=\s*(.+)", extracted_text):
            var, val = m.groups()
            var_n = _normalize_isabelle_symbols(var)
            val_n = _normalize_isabelle_symbols(_TRAIL_PUNCT_RE.sub('', val.strip()))
            if var_n not in seen_vars:
                seen_vars.add(var_n)
                bindings.append(f"{var_n} = {val_n}")
            if len(bindings) >= 8:
                break

    # *_def unfolding hints
    defs: List[str] = []
    seen_defs: Set[str] = set()
    for match in re.finditer(r"\b([A-Za-z_]\w*'*)_def\b", extracted_text):
        name = match.group(1)
        if name not in seen_defs:
            seen_defs.add(name)
            defs.append(name)
            if len(defs) >= 12:
                break
    def_hints = [f"unfolding {d}_def" for d in defs]

    result = {"bindings": bindings[:8], "def_hints": def_hints[:12]}
    #print(f"[DEBUG] Extracted counterexample hints: {result}", file=sys.stderr)
    return result

# --- state parsing ---

def _parse_goal_from_state(state_block: str) -> Optional[str]:
    """Extract the primary goal from a state block."""
    if not state_block:
        return None
    goal_match = re.search(r'goal\s*\([^)]+\):\s*\d+\.\s*(.+?)(?:\n\s*\d+\.|$)', state_block, re.DOTALL)
    if goal_match:
        goal = goal_match.group(1).strip()
        goal = re.sub(r'\s+', ' ', goal)
        return goal
    return None

def _parse_assumptions_from_state(state_block: str) -> List[str]:
    """Extract assumptions from "using this:" section of state block."""
    if not state_block:
        return []
    using_match = re.search(r'using this:\s*(.+?)(?=goal|\Z)', state_block, re.DOTALL)
    if not using_match:
        return []
    assumptions: List[str] = []
    for line in using_match.group(1).split('\n'):
        line = line.strip()
        if line and not line.startswith(('goal', 'proof')):
            assumptions.append(line)
    return assumptions[:5]

def _strip_all_forall_prefixes(prop: str) -> str:
    """Remove leading ⋀x y z. / \<And>x y. prefixes (possibly many)."""
    if not prop:
        return prop
    # Match one or more names (incl. type/commas) up to the first dot after a ⋀/\<And>
    q = re.compile(r'^(?:\\<And>|⋀)\s*[^.]*\.\s*')
    while True:
        m = q.match(prop)
        if not m:
            break
        prop = prop[m.end():]
    return prop


def _counterexample_hints_from_state(
    isabelle, 
    session: str, 
    state_block: str,
    timeout_s: int = 10
) -> Dict[str, List[str]]:
    """
    Extract counterexample hints by recreating the failing goal in a clean context.
    """
    import sys
    
    goal = _parse_goal_from_state(state_block)
    if not goal:
        #print("[DEBUG] Could not extract goal from state block", file=sys.stderr)
        return {"bindings": [], "def_hints": []}
    
    #print(f"[DEBUG] Extracted goal: {goal[:100]}...", file=sys.stderr)
    
    assumptions = _parse_assumptions_from_state(state_block)
    
    assumes_clause = ""
    if assumptions:
        clean_assumptions = []
        for i, assum in enumerate(assumptions):
            assum = re.sub(r'^\s*\d+\.\s*', '', assum).strip()
            if assum:
                clean_assumptions.append(f'A{i}: "{assum}"')
        if clean_assumptions:
            assumes_clause = f"assumes {' and '.join(clean_assumptions)}\n  "
    
    clean_goal = _strip_all_forall_prefixes(goal)
    
    test_theory = f"""
theory CounterexampleTest
imports Main Nitpick
begin

lemma counterexample_test:
  {assumes_clause}shows "{clean_goal}"
  quickcheck[timeout={max(1, timeout_s//3)}]
  nitpick[timeout={max(1, timeout_s)}, verbose, show_all]
  oops

end
"""
    
    #print(f"[DEBUG] Testing with simplified theory", file=sys.stderr)
    
    try:
        thy = build_theory(test_theory.splitlines(), add_print_state=False, end_with=None)
        resps = _run_theory_with_timeout(isabelle, session, thy, timeout_s=timeout_s + 5)
        
        full_output: List[str] = []
        for r in resps or []:
            body = getattr(r, "response_body", None)
            if body is None:
                continue
            if isinstance(body, bytes):
                body = body.decode(errors="replace")
            elif not isinstance(body, str):
                # Fix: handle nonstring types
                try:
                    body = str(body)
                except Exception:
                    continue
            full_output.append(str(body))
        
        result = "\n".join(full_output)
        #print(f"[DEBUG] Counterexample check output: {len(result)} chars", file=sys.stderr)
        
        hints = _nitpick_state_hints_from_text(result)
        return hints
        
    except Exception as e:
        #print(f"[DEBUG] Counterexample check failed: {type(e).__name__}: {e}", file=sys.stderr)
        return {"bindings": [], "def_hints": []}


def get_counterexample_hints_for_repair(
    isabelle,
    session: str,
    state_block: str,
    full_text: str = None,
    hole_span: Tuple[int, int] = None,
    timeout_s: int = 10
) -> Dict[str, List[str]]:
    """
    Main entry point for getting counterexample hints during repair.
    Returns dict with 'bindings' and 'def_hints' keys.
    """
    return _counterexample_hints_from_state(
        isabelle, 
        session, 
        state_block,
        timeout_s=timeout_s
    )

# [FIX] adding this function for repair.py
def _run_nitpick_at_line(
    isabelle,
    session: str,
    full_text_lines: List[str],
    *,
    inject_before_1based: int,
    timeout_s: int = 6,
) -> Dict[str, Any]:
    """Run Quickcheck+Nitpick on the subgoal *at* a specific proof line.

    Unlike `_counterexample_hints_from_state`, which recreates the lemma's
    overall goal in a fresh theory, this injects the diagnostics inline right
    before `inject_before_1based`, so it probes the *local* obligation that the
    failing tactic on that line is trying to discharge. This is what tells us
    whether an intermediate `have` the planner invented is actually FALSE (in
    which case no tactic repair can save it) versus merely hard to prove (where
    a stronger tactic / more facts would help).

    Returns a dict:
      {
        "found_cex": bool,        # True if a (potential) counterexample was reported
        "bindings": List[str],    # e.g. ["xs = [a]", "ys = []"]
        "def_hints": List[str],   # e.g. ["unfolding foo_def"]
        "raw": str,               # trimmed raw output, for logging
      }

    Never raises; on any failure returns found_cex=False with empty hints.
    """
    empty: Dict[str, Any] = {"found_cex": False, "bindings": [], "def_hints": [], "raw": ""}
    try:
        n = len(full_text_lines)
        if n == 0:
            return empty
        # Clamp to a valid 0-based insertion index.
        idx0 = max(0, min(inject_before_1based - 1, n - 1))

        # Match the indentation of the line we're probing so the injected
        # diagnostic sits at the same proof depth.
        target = full_text_lines[idx0] or ""
        indent = target[: len(target) - len(target.lstrip(" "))]
        pad = indent if indent else "  "

        # Quickcheck first (fast, concrete values); Nitpick second (deeper,
        # finds counterexamples Quickcheck misses). Both are non-destructive
        # diagnostics — they report and the proof continues — so even if the
        # surrounding proof later fails, we still capture their messages.
        qc_to = max(1, timeout_s // 3)
        np_to = max(1, timeout_s)
        injected = [
            f"{pad}quickcheck[timeout={qc_to}]",
            f"{pad}nitpick[timeout={np_to}, verbose, show_all]",
        ]

        variant_lines = full_text_lines[:idx0] + injected + full_text_lines[idx0:]

        thy = build_theory(variant_lines, add_print_state=False, end_with=None)
        resps = _run_theory_with_timeout(
            isabelle, session, thy, timeout_s=timeout_s + 5
        )

        full_output: List[str] = []
        for r in resps or []:
            body = getattr(r, "response_body", None)
            if isinstance(body, (bytes, bytearray)):
                body = body.decode(errors="replace")
            elif body is None:
                continue
            full_output.append(str(body))
        result = "\n".join(full_output)

        hints = _nitpick_state_hints_from_text(result)
        bindings = hints.get("bindings", []) if isinstance(hints, dict) else []
        def_hints = hints.get("def_hints", []) if isinstance(hints, dict) else []

        # `_nitpick_state_hints_from_text` already gates on the counterexample
        # markers, so non-empty bindings imply a CEX was found. Also re-check the
        # markers directly in case a CEX was reported with no parseable bindings.
        extracted = _extract_nitpick_text_from_responses(result) or result
        found = bool(bindings) or any(m in extracted.lower() for m in _CEX_MARKERS)

        # Trim raw output for logging sanity.
        raw = extracted.strip()
        if len(raw) > 1200:
            raw = raw[:600] + "\n…\n" + raw[-600:]

        return {
            "found_cex": found,
            "bindings": bindings,
            "def_hints": def_hints,
            "raw": raw,
        }
    except (TimeoutError, _FuturesTimeout):
        # Diagnostic timed out — treat as inconclusive, not as "no counterexample".
        return {"found_cex": False, "bindings": [], "def_hints": [], "raw": "(nitpick timeout)"}
    except Exception as e:
        return {"found_cex": False, "bindings": [], "def_hints": [], "raw": f"(nitpick error: {type(e).__name__})"}

# ========== Context Analysis ==========

def _hole_line_bounds(full_text: str, hole_span: Tuple[int, int]) -> Tuple[int, int, List[str]]:
    lines = full_text.splitlines()
    hole_line = full_text[:hole_span[0]].count("\n")
    line_text = lines[hole_line] if 0 <= hole_line < len(lines) else ""
    indent = len(line_text) - len(line_text.lstrip(" "))
    return hole_line, indent, lines

def _find_first_hole(lines: List[str]) -> Optional[int]:
    for i, line in enumerate(lines):
        if "sorry" in line:
            return i
    return None

def _snippet_window(lines: List[str], hole_line: int, radius: int = 12) -> Tuple[int, int]:
    return max(0, hole_line - radius), min(len(lines), hole_line + radius + 1)

def _facts_from_state(state_block: str, limit: int = 16) -> List[str]:
    if not state_block:
        return []
    facts: List[str] = []
    seen: Set[str] = set()
    # Priority 1: propositions under "using this:"
    m = re.search(r"using this:\n((?:[ \t].*\n)+)", state_block)
    if m:
        for L in m.group(1).splitlines():
            s = L.strip()
            if s and s not in seen:
                seen.add(s)
                facts.append(s)
                if len(facts) >= limit:
                    return facts
    # Priority 2: quoted propositions
    for q in re.findall(r'(?m)^\s*"(.*?)"\s*$', state_block):
        s = q.strip()
        if s and s not in seen:
            seen.add(s)
            facts.append(s)
            if len(facts) >= limit:
                return facts
    # Priority 3: *_def names
    for d in re.findall(r"\b([A-Za-z_][A-Za-z0-9_']*)_def\b", state_block):
        if d and d not in seen:
            seen.add(d)
            facts.append(f"{d}_def")
            if len(facts) >= limit:
                break
    return facts

def _nearest_header(lines: List[str], hole_line: int) -> str:
    for i in range(hole_line, -1, -1):
        if _HEADER_RE.match(lines[i].strip()):
            return lines[i].strip()
    return ""

def _recent_steps(lines: List[str], hole_line: int, max_lines: int = 5) -> List[str]:
    steps: List[str] = []
    for i in range(hole_line - 1, -1, -1):
        if _APPLY_OR_BY.match(lines[i]):
            steps.append(lines[i].strip())
            if len(steps) >= max_lines:
                break
        if lines[i].strip().startswith(("case ", "proof", "qed", "lemma ")):
            break
    return list(reversed(steps))

def _extract_error_lines(errs) -> list[int]:
    out: List[int] = []
    for e in errs:
        ln = e.get("line") if isinstance(e, dict) else getattr(e, "line", None)
        if isinstance(ln, int):
            out.append(ln)
    return out

def _normalize_error_texts(errs) -> List[str]:
    return [str(e.get("text", "") if isinstance(e, dict) else e).strip() 
            for e in (errs or []) 
            if str(e.get("text", "") if isinstance(e, dict) else e).strip()][:8]