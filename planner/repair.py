from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
import requests
from typing import Callable
from planner.repair_inputs import _find_first_hole, _hole_line_bounds, _APPLY_OR_BY, _snippet_window, _clamp_line_index, _quick_state_and_errors, _extract_error_lines, _run_theory_with_timeout, _print_state_before_hole, _nearest_header, _recent_steps, _normalize_error_texts, _facts_from_state, get_counterexample_hints_for_repair, _earliest_failure_anchor, _run_nitpick_at_line
from planner.prompts import _LOCAL_SYSTEM, _LOCAL_USER, _BLOCK_SYSTEM, _BLOCK_USER
from planner.goals import _verify_full_proof
from prover.config import MODEL as DEFAULT_MODEL, OLLAMA_HOST, TIMEOUT_S as OLLAMA_TIMEOUT_S, OLLAMA_NUM_PREDICT, TEMP as OLLAMA_TEMP, TOP_P as OLLAMA_TOP_P
from prover.isabelle_api import build_theory, run_theory, last_print_state_block, finished_ok

# ========== Configuration ==========
_ISA_FAST_TIMEOUT_S = int(os.getenv("ISABELLE_FAST_TIMEOUT_S", "12"))
_ISA_VERIFY_TIMEOUT_S = int(os.getenv("ISABELLE_VERIFY_TIMEOUT_S", "30"))
_SESSION = requests.Session()
_REPAIR_RULES_JSON = os.getenv("REPAIR_RULES_JSON", "").strip()  # optional, declarative fallback rules

# #Fix: Inter-request delay (seconds) inserted after every Gemini call to avoid
# #Fix: tripping the API's requests-per-minute quota when _repair_block loops
# #Fix: over multiple rounds (up to 3 rounds × 3 stages = up to 9 rapid calls).
# #Fix: Tune this value to match your Gemini quota tier.
_GEMINI_INTER_REQUEST_DELAY_S: float = float(os.getenv("GEMINI_INTER_REQUEST_DELAY_S", "1.0"))
# [FIX]
# Lines (1-based, in the current full_text) whose local obligation Nitpick/Quickcheck
# proved FALSE. A false obligation can never be discharged by tactic repair, so the
# driver should skip leaf-level repair for these and escalate straight to regeneration.
# Best-effort signal: it is repopulated as repair runs and read by the driver via
# `pop_false_subgoal_lines()`. Cleared on read so stale entries don't leak across holes.
_FALSE_SUBGOAL_LINES: Set[int] = set()

def pop_false_subgoal_lines() -> Set[int]:
    """Return and clear the set of lines found to have a FALSE local obligation."""
    global _FALSE_SUBGOAL_LINES
    out = set(_FALSE_SUBGOAL_LINES)
    _FALSE_SUBGOAL_LINES = set()
    return out

# ========== Regex Patterns ==========
_CTX_HEAD = re.compile(r"^\s*(?:using|from|with|then|ultimately|finally|also|moreover)\b")
_HAS_BODY = re.compile(r"^\s*(?:by\b|apply\b|proof\b|sorry\b|done\b)")
_INLINE_BY_TAIL = re.compile(r"\s+by\s+.+$")
_TACTIC_LINE = re.compile(r"^\s*(?:apply|by)\b|(?:\s)by\s+\S")
_STRUCTURAL_LINE = re.compile(r"^\s*(?:lemma|theorem|qed|next|proof|case|have|show|assume|fix|from|using|thus|hence|ultimately|finally|also|moreover|let|where)\b")
_HEAD_CMD_RE = re.compile(r"^\s*(have|show|obtain|then\s+show|thus|hence)\b")
_PROOF_RE = re.compile(r"^\s*proof\b")
_QED_RE = re.compile(r"^\s*qed\b")
_CASE_LINE_RE = re.compile(r"^\s*case\b")
_NEXT_OR_QED_RE = re.compile(r"^\s*(?:next|qed)\b")
_WRAPPED_THEOREM_HEAD = re.compile(r"(?mx)\A(?:[ \t]*(?:\(\*.*?\*\)|\<comment\>.*?\<\/comment\>)[ \t]*\n|[ \t]*\n)*[ \t]*(?:lemma|theorem|corollary)\b")
# Outline-level strategies we want to ban on whole-proof regen
_OUTLINE_PROOF_LINE   = re.compile(r"(?m)^\s*proof(?:\s*\(([^)]*)\))?\s*$")
_OUTLINE_BARE         = re.compile(r"(?m)^\s*(?:induction|cases|coinduction)\b.*$")

# ========== Utility Functions ==========

def _log(prefix: str, label: str, content: str, trace: bool = True) -> None:
    if trace and content:
        print(f"[{prefix}] {label} (len={len(content)}):\n{content if content.strip() else '  (empty)'}", flush=True)

def _sanitize_llm_block(text: str) -> str:
    if not text:
        return text
    patterns = [
        r"^\s*<<<BLOCK\s*$",
        r"^\s*BLOCK\s*$",
        r"^\s*<<<PROOF\s*$",
        r"^\s*PROOF\s*$",
        r"^\s*```\s*$",
        r"^\s*```isabelle\s*$",
        r"^\s*```isar\s*$",
        # strip stray fence markers sometimes emitted by LLMs
        r"^\s*<<<\s*$",
        r"^\s*>>>\s*$",
    ]
    # Also drop accidental headers LLMs sometimes leak mid-repair
    header_patterns = [
        r"^\s*lemma\b.*$",
        r"^\s*theorem\b.*$",
        r"^\s*corollary\b.*$",
        r"^\s*proposition\b.*$",
        r"^\s*---\s*$",
    ]
    compiled = [re.compile(p) for p in (patterns + header_patterns)]
    lines = [l for l in text.splitlines() if not any(p.match(l) for p in compiled)]

    # Balance 'proof'/'qed' and cut off any text after the final balanced 'qed'
    balance = 0
    last_closed_idx = -1
    for i, l in enumerate(lines):
        if re.match(r"^\s*proof\b", l):
            balance += 1
        elif re.match(r"^\s*qed\b", l):
            if balance > 0:
                balance -= 1
                if balance == 0:
                    last_closed_idx = i
    if last_closed_idx != -1 and last_closed_idx + 1 < len(lines):
        lines = lines[: last_closed_idx + 1]

    return "\n".join(lines).strip()

def _is_effective_block(text: str) -> bool:
    return bool(_sanitize_llm_block(text or "").strip())

def _fingerprint_block(text: str) -> str:
    """Canonicalize a block to detect duplicates across rounds."""
    if not text:
        return ""
    # Collapse whitespace, drop zero-width and backticks, normalize quotes.
    t = re.sub(r"\s+", " ", text.strip())
    t = t.replace("`", "").replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    return t

def _trim_block_for_prompt(text: str, max_chars: int = 800) -> str:
    """Keep prompt sizes sane by trimming long blocks."""
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    head = t[: max_chars // 2].rstrip()
    tail = t[- max_chars // 2 :].lstrip()
    return head + "\n…\n" + tail

def _is_tactic_line(s: str) -> bool:
    return bool(_TACTIC_LINE.search(s)) and not bool(_STRUCTURAL_LINE.match(s))

def _extract_proof_context(full_text: str, block_start_line: int) -> str:
    """
    Extract the lemma header and all proof content before the block.
    Returns everything from the lemma line up to (but not including) the block.
    """
    lines = full_text.splitlines()
    
    # Find the lemma/theorem header
    lemma_line = -1
    for i in range(min(block_start_line, len(lines) - 1), -1, -1):
        if re.match(r"^\s*(?:lemma|theorem|corollary|proposition)\b", lines[i]):
            lemma_line = i
            break
    
    if lemma_line < 0:
        # No lemma found, return a small window before the block
        start = max(0, block_start_line - 10)
        return "\n".join(lines[start:block_start_line]).strip()
    
    # Return from lemma header to just before the block
    context_lines = lines[lemma_line:block_start_line]
    return "\n".join(context_lines).strip()

# ========== LLM Generation ==========

# #Fix: Track whether we issued a Gemini call this process so we can insert a
# #Fix: delay before the *next* one. Using a module-level flag avoids coupling
# #Fix: the sleep to the call site (callers don't need to know about rate limits).
_gemini_last_call_time: float = 0.0

def _generate_simple(prompt: str, model: Optional[str] = None, *, timeout_s: Optional[int] = None) -> str:
    m = model or DEFAULT_MODEL
    timeout = timeout_s if timeout_s is not None else OLLAMA_TIMEOUT_S
    display_model = m
    dump = os.getenv("LLM_DUMP", "").strip().lower() in ("1", "true", "yes", "on")

    # if dump:
    #     print(f"{'='*60}", flush=True)
    #     print(f"[Repair] LLM Prompt:\n{prompt.rstrip()}", flush=True)
    #     print(f"{'-'*60}", flush=True)
    
    if m.startswith("hf:"):
        raw = _hf_generate(prompt, m[3:], timeout)
    elif m.startswith("gemini:"):
        #Fix: Enforce a minimum gap between successive Gemini REST calls.
        #Fix: _repair_block drives up to 3 rounds × 3 stages so without this
        #Fix: guard all iterations land within the same second, causing 429s.
        global _gemini_last_call_time
        elapsed = time.monotonic() - _gemini_last_call_time
        if elapsed < _GEMINI_INTER_REQUEST_DELAY_S:
            time.sleep(_GEMINI_INTER_REQUEST_DELAY_S - elapsed)
        raw = _gemini_generate(prompt, m[7:], timeout)
        #Fix: Record the time of this call so the next call can compute the gap.
        _gemini_last_call_time = time.monotonic()
    elif m.startswith("ollama:"):
        m = m[7:]
        raw = _ollama_generate(prompt, m, timeout)
    else:
        raw = _ollama_generate(prompt, m, timeout)

    if dump:
        print(f"[Repair] model={display_model}", flush=True)
        print(f"[Repair] LLM Output:\n{raw}", flush=True)
        print(f"{'='*60}", flush=True)

    return raw

def _ollama_generate(prompt: str, model: str, timeout_s: int) -> str:
    payload = {"model": model, "prompt": prompt, "options": {"temperature": OLLAMA_TEMP, "top_p": OLLAMA_TOP_P, "num_predict": OLLAMA_NUM_PREDICT}, "stream": False}
    timeout = (10.0, max(30.0, float(timeout_s)))
    resp = _SESSION.post(f"{OLLAMA_HOST.rstrip('/')}/api/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    return _sanitize_llm_block(resp.json().get("response", "").strip())

def _hf_generate(prompt: str, model_id: str, timeout_s: int) -> str:
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_API_TOKEN is not set")
    payload = {"inputs": prompt, "parameters": {"temperature": OLLAMA_TEMP, "top_p": OLLAMA_TOP_P, "max_new_tokens": OLLAMA_NUM_PREDICT, "return_full_text": False}, "options": {"wait_for_model": True}}
    resp = _SESSION.post(f"https://api-inference.huggingface.co/models/{model_id}", headers={"Authorization": f"Bearer {token}"}, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data:
        result = data[0].get("generated_text", "")
    elif isinstance(data, dict):
        result = data.get("generated_text", "") or (data["choices"][0].get("text", "") if "choices" in data and data["choices"] else "")
    else:
        result = str(data)
    return _sanitize_llm_block(result.strip())

def _gemini_generate(prompt: str, model_id: str, timeout_s: int) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent?key={api_key}"
    )
    
    gemini_max_tokens = max(OLLAMA_NUM_PREDICT, 4096)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": gemini_max_tokens},
    }
    
    resp = _SESSION.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini API returned no candidates. Response: {data}")
        
    first_candidate = candidates[0]
    finish_reason = first_candidate.get("finishReason")
    
    # Check if Gemini blocked the proof generation due to safety/recitation flags
    if finish_reason and finish_reason not in ("STOP", "MAX_TOKENS"):
        raise RuntimeError(f"Gemini generation stopped unexpectedly. Reason: {finish_reason}. Data: {data}")

    parts = first_candidate.get("content", {}).get("parts", [])
    if not parts:
        # If there's no text part but it didn't explicitly throw an error
        raise RuntimeError(f"Gemini API returned an empty content part. Data: {data}")
        
    result = parts[0].get("text", "")
    result = result.strip()
    
    # Return raw text here. Let the specialized caller handle parsing/sanitization!
    return result

# ========== Repair Operations (Data Classes) ==========
@dataclass(frozen=True)
class InsertBeforeHole:
    line: str

@dataclass(frozen=True)
class ReplaceInSnippet:
    find: str
    replace: str

@dataclass(frozen=True)
class InsertHaveBlock:
    label: str
    statement: str
    after_line_matching: str
    body_hint: str

RepairOp = Tuple[str, object]

@dataclass
class _RepairMemory:
    rounds: int = 0
    # Keep full failed blocks (same block_type) we tried this session
    prev_blocks: List[str] = field(default_factory=list)
    # Fingerprints to dedup within a session
    prev_fps: Set[str] = field(default_factory=set)

# --- Prior-block store shared across repairs of the *same hole* ---------------
# Maps block_type -> list of failed blocks (latest first, length-capped)
_MAX_PREV_BLOCKS = int(os.getenv("REPAIR_MAX_PREV_BLOCKS", "4"))

# ========== Repair Operations (Parsing & Application) ==========

def _propose_block_repair(*, goal: str, errors: List[str], ce_hints: Dict[str, List[str]], 
                         proof_context: str,  block_type: str,
                         block_text: str, model: Optional[str], timeout_s: int,
                         why: str = "Previous attempt failed; propose a different block-level change.",
                         prior_failed_blocks: Optional[str] = None) -> str:
    ce = ce_hints.get("bindings", []) + ce_hints.get("def_hints", [])
    if block_type == "have-show":
        prompt = _LOCAL_SYSTEM + "\n\n" + _LOCAL_USER.format(
            goal=goal, errors="\n".join(f"- {e}" for e in errors) or "(none)",
            ce_hints="\n".join(ce) or "(none)", 
            proof_context=(proof_context or "").strip(),
            block_text=block_text.rstrip(), why=why,
            prior_failed_blocks=(prior_failed_blocks or "(none)")
        )
    else:
        prompt = _BLOCK_SYSTEM + "\n\n" + _BLOCK_USER.format(
            goal=goal, errors="\n".join(f"- {e}" for e in errors) or "(none)",
            ce_hints="\n".join(ce) or "(none)", 
            proof_context=(proof_context or "").strip(),
            block_text=block_text.rstrip(), why=why,
            prior_failed_blocks=(prior_failed_blocks or "(none)")
        )
    try:
        # Run generative call via your wrapper (_generate_simple calls _gemini_generate)
        raw_output = _generate_simple(prompt, model=model, timeout_s=timeout_s)
        # Sanitize the raw string exactly once right here
        santised_output =  _sanitize_llm_block(raw_output)
        # print(f"[repair] raw_output = {raw_output}", flush=True)
        # print(f"[repair] santized_output = {santised_output}", flush=True)
        return santised_output
    except Exception as e:
        # Un-commenting this is critical to diagnose why the repair returns ""
        print(f"[repair] LLM block proposal generation failed: {type(e).__name__}: {e}", flush=True)
        return ""

def propose_rule_based_repairs(goal_text: str, state_block: str, header: str, facts: List[str]) -> List[RepairOp]:
    """
    Declarative, data-driven fallback:
    - If REPAIR_RULES_JSON is set to a JSON file, load rules and emit ops that match.
    - Otherwise return [] (i.e., no ad-hoc heuristics).
    Rule schema (list):
      {
        "when": {
          "goal_contains_any": ["@", "map"],
          "goal_regex": "length\\s",
          "facts_contains_any": ["append_assoc"],
          "state_contains_any": ["Let "],
          "header_startswith": "proof (induction",
          "header_regex": "proof \\(induction.*\\)"
        },
        "op": { "insert_before_hole": "apply (simp add: append_assoc)" }
      }
      or
      {
        "when": { "header_startswith": "proof (induction", "not_header_contains": ["arbitrary:"] },
        "op": { "replace_in_snippet": { "find": "proof (induction xs)", "replace": "proof (induction xs arbitrary: ys)" } }
      }
    """
    path = _REPAIR_RULES_JSON
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            rules = json.load(f)
    except Exception:
        return []
    def _match(rule) -> Optional[RepairOp]:
        cond = rule.get("when", {}) or {}
        op   = rule.get("op", {}) or {}
        g, st, hd = goal_text or "", state_block or "", header or ""
        fs = facts or []
        import re as _re
        def contains_any(text, keys): return any(k in text for k in keys)
        def not_contains(text, keys): return not any(k in text for k in keys)
        # boolean guards (all must pass if present)
        checks = [
            ("goal_contains_any", lambda v: contains_any(g, v)),
            ("state_contains_any", lambda v: contains_any(st, v)),
            ("facts_contains_any", lambda v: any(x in fs for x in v)),
            ("goal_regex",        lambda v: bool(_re.search(v, g))),
            ("header_startswith", lambda v: hd.startswith(v)),
            ("header_regex",      lambda v: bool(_re.search(v, hd))),
            ("not_header_contains", lambda v: not_contains(hd, v)),
        ]
        for key, pred in checks:
            if key in cond:
                val = cond[key]
                if isinstance(val, list) and not val: 
                    continue
                if not pred(val):
                    return None
        # build op
        if "insert_before_hole" in op and isinstance(op["insert_before_hole"], str):
            return ("insert_before_hole", InsertBeforeHole(op["insert_before_hole"].strip()))
        if "replace_in_snippet" in op and isinstance(op["replace_in_snippet"], dict):
            fnd = (op["replace_in_snippet"].get("find") or "").strip()
            rep = (op["replace_in_snippet"].get("replace") or "").strip()
            if fnd and rep:
                return ("replace_in_snippet", ReplaceInSnippet(fnd, rep))
        if "insert_have_block" in op and isinstance(op["insert_have_block"], dict):
            v = op["insert_have_block"]; lab=v.get("label","H"); stmt=v.get("statement",""); aft=v.get("after_line_matching","then show ?thesis"); hint=v.get("body_hint","apply simp")
            if stmt.strip() and aft.strip():
                return ("insert_have_block", InsertHaveBlock(lab.strip(), stmt.strip(), aft.strip(), hint.strip()))
        return None
    out: List[RepairOp] = []
    for r in rules if isinstance(rules, list) else []:
        rop = _match(r)
        if rop: out.append(rop)
        if len(out) >= 3:
            break
    return out

# ========== Region Analysis ==========
def _enclosing_case_block(lines: List[str], hole_line: int) -> Tuple[int, int]:
    i = hole_line
    while i >= 0 and not _CASE_LINE_RE.match(lines[i]):
        i -= 1
    if i < 0:
        return (-1, -1)
    j = hole_line
    while j < len(lines) and not (_NEXT_OR_QED_RE.match(lines[j])):
        j += 1
    return (i, j)

def _enclosing_subproof(lines: List[str], hole_line: int) -> Tuple[int, int]:
    i = hole_line
    while i >= 0 and not _PROOF_RE.match(lines[i]):
        i -= 1
    if i < 0:
        return (-1, -1)
    depth, j = 1, i + 1
    while j < len(lines) and depth > 0:
        if _PROOF_RE.match(lines[j]):
            depth += 1
        elif _QED_RE.match(lines[j]):
            depth -= 1
        j += 1
    return (i, j if j > i else -1)

def _enclosing_have_show_block(lines: List[str], hole_line: int, trace) -> Tuple[int, int]:
    if not lines:
        return (-1, -1)

    i = _clamp_line_index(lines, hole_line)

    head_re  = re.compile(r"^\s*(have|show|obtain)\b")
    # IMPORTANT: do NOT include `proof` here — subproofs belong to the block.
    fence_re = re.compile(
        r"^\s*(?:have|show|obtain|thus|hence|then|also|moreover|ultimately|finally|case\b|next\b|qed\b)\b"
    )
    # Stop boundaries: climbing past these means we left the current local block context entirely
    boundary_re = re.compile(r"^\s*(?:lemma|theorem|proof|case\b|next\b|qed\b)\b")

    # # Accept calculation elements as starting heads too
    # head_re  = re.compile(r"^\s*(have|show|obtain|also|finally)\b")
    # # REMOVED also, moreover, ultimately, finally from the fence boundaries
    # fence_re = re.compile(
    #     r"^\s*(?:have|show|obtain|thus|hence|then|case\b|next\b|qed\b)\b"
    # )

    # climb to the enclosing have/show head, but stop if we hit a block fence
    while i >= 0 and not head_re.match(lines[i]):
        if boundary_re.match(lines[i]):
            return (-1, -1)
        i -= 1

    if i < 0 or not head_re.match(lines[i]):
        return (-1, -1)

    # if the head line itself has an inline "by …", keep only that line
    if _INLINE_BY_TAIL.search(lines[i] or ""):
        # If it's a one-liner but our target hole is below it, then the hole is NOT encapsulated
        if hole_line > i:
            return (-1, -1)
        return (i, i + 1)

    # Track nested subproofs correctly from the head line outward.
    depth = 0
    j = i + 1
    while j < len(lines):
        L = lines[j]

        # Base-level one-liner endings
        if depth == 0:
            # stop immediately after a base-level "sorry"
            if (L or "").strip() == "sorry":
                j = j + 1
                break
            # do not include any fence token at base depth
            if fence_re.match(L or ""):
                break

        # Subproof bookkeeping
        if _PROOF_RE.match(L or ""):
            depth += 1
        elif _QED_RE.match(L or ""):
            depth = max(0, depth - 1)

        j += 1

    # CRITICAL CHECK: Does the block actually encapsulate our focus line?
    # The block spans from index 'i' to 'j' (exclusive). If hole_line falls outside this range,
    # the target line is in a structural gap, not inside this specific block.
    if not (i <= hole_line < j):
        if trace:
            print(f"[repair] Found block ({i}, {j}) but it does not encapsulate focus line {hole_line}.")
        return (-1, -1)

    return (i, j if j > i else -1)

def _enclosing_whole_proof(lines: List[str]) -> Tuple[int, int]:
    last_qed = -1
    for i, line in enumerate(lines):
        if _QED_RE.match(line):
            last_qed = i
    if last_qed < 0:
        return (-1, -1)
    for i in range(last_qed, -1, -1):
        if _PROOF_RE.match(lines[i]):
            return (i, last_qed + 1)
    return (-1, -1)

# ========== Wrapper Stripping ==========
def _strip_wrapper_to_case_block(proposed: str, original_case_block: str) -> str:
    if not _WRAPPED_THEOREM_HEAD.match(proposed):
        return proposed
    case_name = None
    m = re.search(r"(?m)^\s*case\s*\((\w+)", original_case_block or "")
    if m:
        case_name = m.group(1)
    else:
        m = re.search(r"(?m)^\s*case\s+(\w+)", original_case_block or "")
        if m:
            case_name = m.group(1)
    lines = proposed.splitlines()
    start = None
    for i, L in enumerate(lines):
        if not _CASE_LINE_RE.match(L):
            continue
        if case_name is None or re.match(rf"^\s*case\s*\({re.escape(case_name)}\b", L) or re.match(rf"^\s*case\s+{re.escape(case_name)}\b", L):
            start = i
            break
    if start is None:
        return proposed
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _NEXT_OR_QED_RE.match(lines[j]):
            end = j
            break
    return "\n".join(lines[start:end]).rstrip()

def _strip_wrapper_to_have_show(proposed: str, original_block: str) -> str:
    # Keep only the single have/show/obtain micro-block, including any nested
    # `proof … qed` it contains, but STOP at the first base-level `sorry`
    # or base-level inline `by …`.
    lines = proposed.splitlines()
    if not lines:
        return proposed

    head_re  = re.compile(r"^\s*(have|show|obtain)\b")
    fence_re = re.compile(
        r"^\s*(?:have|show|obtain|thus|hence|then|also|moreover|ultimately|finally|case\b|next\b|qed\b)\b"
    )

    # find the first have/show/obtain head
    head_idx = next((i for i, L in enumerate(lines) if head_re.match(L)), -1)
    if head_idx == -1:
        return proposed

    out: List[str] = [lines[head_idx]]
    depth = 0

    for L in lines[head_idx + 1:]:
        # Base-level one-line endings
        if depth == 0:
            # Keep a base-level "sorry", but stop right after it
            if (L or "").strip() == "sorry":
                out.append(L)
                break
            # Do not include any new head/fence (then/also/moreover/…/case/next/qed)
            if fence_re.match(L or ""):
                break

        # Track nested subproofs (kept inside the micro-block)
        if _PROOF_RE.match(L or ""):
            depth += 1
        elif _QED_RE.match(L or ""):
            depth = max(0, depth - 1)

        out.append(L)

    # Trim any trailing whitespace lines we may have kept
    while out and out[-1].strip() == "":
        out.pop()

    # Final guard: if the last kept line is an inline 'by …' on the head, trim to that line only
    if len(out) == 1 and _INLINE_BY_TAIL.search(out[0] or ""):
        return out[0].rstrip()
    return "\n".join(out).rstrip()

def _strip_wrapper_to_subproof(proposed: str) -> str:
    if not _WRAPPED_THEOREM_HEAD.match(proposed):
        return proposed
    lines = proposed.splitlines()
    start = None
    for i, L in enumerate(lines):
        if _PROOF_RE.match(L):
            start = i
            break
    if start is None:
        return proposed
    depth, j = 1, start + 1
    while j < len(lines) and depth > 0:
        if _PROOF_RE.match(lines[j]):
            depth += 1
        elif _QED_RE.match(lines[j]):
            depth -= 1
        j += 1
    return "\n".join(lines[start:j if depth == 0 else len(lines)]).rstrip()

# ========== Safe Sorry Insertion ==========
def _find_enclosing_head(block_lines: List[str], from_idx: int) -> Optional[int]:
    for i in range(from_idx, -1, -1):
        if _HEAD_CMD_RE.match(block_lines[i] or ""):
            return i
    return None

def _apply_sequence_bounds(block_lines: List[str], idx: int) -> Tuple[int, int]:
    s = idx
    while s > 0 and _is_tactic_line(block_lines[s-1]):
        s -= 1
    e = idx + 1
    while e < len(block_lines) and _is_tactic_line(block_lines[e]):
        e += 1
    return s, e

def _replace_failing_tactics_with_sorry(block_text: str, *, full_text_lines: List[str], start_line: int, 
                                       end_line: int, isabelle, session: str, trace: bool = False) -> str:
    block_lines = block_text.splitlines()
    if not block_lines:
        return block_text    
    def build_doc(with_block_lines: List[str]) -> str:
        s0, e0 = max(0, start_line - 1), max(max(0, start_line - 1), min(end_line - 1, len(full_text_lines)))
        return "\n".join(full_text_lines[:s0] + with_block_lines + full_text_lines[e0:])
    
    while True:
        doc = build_doc(block_lines)
        _, errs = _quick_state_and_errors(isabelle, session, doc)
        err_in_block = sorted(set(l for l in _extract_error_lines(errs) if start_line <= l < end_line))
        thy = build_theory(doc.splitlines(), add_print_state=False, end_with=None)
        ok, _ = finished_ok(_run_theory_with_timeout(isabelle, session, thy, timeout_s=_ISA_VERIFY_TIMEOUT_S))
        
        if not err_in_block:
            break
        
        failing_idx = err_in_block[0] - start_line
        cand = None
        if 0 <= failing_idx < len(block_lines) and _is_tactic_line(block_lines[failing_idx]):
            cand = failing_idx
        else:
            for i in range(min(failing_idx, len(block_lines) - 1), -1, -1):
                if _is_tactic_line(block_lines[i]):
                    cand = i
                    break
            if cand is None:
                for i in range(max(0, failing_idx + 1), len(block_lines)):
                    if _is_tactic_line(block_lines[i]):
                        cand = i
                        break
        
        if cand is None:
            break

        # [FIX]
        # --- Diagnostics before modifying the block ---
        # Run Quickcheck/Nitpick on the exact failing tactic line, so we capture
        # a counterexample on the subgoal that is about to fail. A counterexample
        # here means the local obligation is FALSE — tactic repair cannot help,
        # and the structure above this line needs to change instead.
        try:
            diag = _run_nitpick_at_line(
                isabelle, session, full_text_lines,
                inject_before_1based=start_line + cand,
                timeout_s=6,
            )
            if diag.get("found_cex"):
                binds = ", ".join(diag.get("bindings", [])) or "(no parseable bindings)"
                _log("repair", "nitpick (pre-sorry) FALSE-subgoal",
                     f"counterexample at line {start_line + cand}: {binds}", trace=trace)
                _FALSE_SUBGOAL_LINES.add(start_line + cand)
            elif diag.get("raw"):
                _log("repair", "nitpick (pre-sorry)", diag["raw"], trace=trace)
        except Exception:
            pass

        indent = block_lines[cand][:len(block_lines[cand]) - len(block_lines[cand].lstrip())]
        if block_lines[cand].lstrip().startswith("apply"):
            head_idx = _find_enclosing_head(block_lines, cand)
            if head_idx is not None:
                head_indent = block_lines[head_idx][:len(block_lines[head_idx]) - len(block_lines[head_idx].lstrip())]
                seq_s, seq_e = _apply_sequence_bounds(block_lines, cand)
                block_lines[seq_s:seq_e] = [f"{head_indent}proof -", f"{head_indent}  sorry", f"{head_indent}qed"]
            else:
                break
        else:
            block_lines[cand] = f"{indent}sorry"
    
    return "\n".join(block_lines)

def try_cegis_repairs(*, full_text: str, hole_span: Tuple[int, int], goal_text: str, model: Optional[str], 
                     isabelle, session: str, repair_budget_s: float = 15.0, max_ops_to_try: int = 3, 
                     beam_k: int = 1, allow_whole_fallback: bool = False, trace: bool = False, 
                     resume_stage: int = 0) -> Tuple[str, bool, str]:
    from planner.skeleton import _quick_sketch_score    # imported here to prevent circular imports

    t0 = time.monotonic()
    left = lambda: max(0.0, repair_budget_s - (time.monotonic() - t0))
    current_text = full_text
    state0 = _print_state_before_hole(isabelle, session, current_text, hole_span, trace=trace)
    _log("repair", "State block", state0, trace=trace)
    
    if allow_whole_fallback and trace:
        print("[repair] (deprecated) allow_whole_fallback=True is ignored; driver handles regeneration.")        

    prior_store: Dict[str, List[str]] = {}

    # Track initial baseline sketch score for relative improvement checks
    initial_score = _quick_sketch_score(isabelle, session, current_text, timeout_s=min(10, left()), trace=trace)
    current_best_score = initial_score

    # Stage 1: have/show/obtain micro-block
    hole_line, _, lines = _hole_line_bounds(current_text, hole_span)
    anchor_line, anchor_reason = _earliest_failure_anchor(isabelle, session, current_text, default_line_0=hole_line)
    focus_line = _clamp_line_index(lines, anchor_line)
    #print(f"Focus line: {lines[focus_line]}", flush=True)
    if trace and anchor_line != hole_line:
        print(f"[repair] Retargeting from hole line {hole_line + 1} to earliest-failure line {anchor_line + 1} ({anchor_reason})", flush=True)

    if left() <= 1.0:
        if trace:
            print(f"[repair] Hard budget timeout in try_cegis_repairs ({left():.2f}s remaining). Aborting loop entirely.")
        return current_text, False, "timeout"
    
    hs_s, hs_e = _enclosing_have_show_block(lines, focus_line, trace)
    if resume_stage <= 1 and hs_s >= 0 and left() > 3.0:    # Fix: lower gate threshold from 5.0
        if trace:
            print("[repair] Trying have/show block repair…")
            print(f"Block: \n{lines[hs_s:hs_e]}")
            
        pre_repair_text = current_text
        current_text, verified = _repair_block(current_text, lines, hs_s, hs_e, goal_text, state0, 
                                     isabelle, session, model, left, trace, "have-show", 
                                     stage=1, prior_store=prior_store)

        if current_text != pre_repair_text:
            if verified or _verify_full_proof(isabelle, session, current_text, timeout_s=min(15, left())):
                return current_text, True, "stage=1 block:have-show"

            # If _repair_block updated the text, it has already verified internal score optimization.
            # Commit it immediately as progress rather than validating against a global threshold.
            if trace:
                print(f"[repair] Stage 1 partial progress accepted.")
            return current_text, False, "stage=1 block:have-show-partial"
        
        lines = current_text.splitlines()
        state0 = _print_state_before_hole(isabelle, session, current_text, hole_span, trace=trace)
    
    # Stage 2a: Case-block
    cs, ce = _enclosing_case_block(lines, focus_line)
    if resume_stage <= 2 and cs >= 0 and left() > 3.0:    # Fix: lower gate threshold from 5.0
        if trace:
            print("[repair] Trying case-block repair…")

        pre_repair_text = current_text
        current_text, verified = _repair_block(current_text, lines, cs, ce, goal_text, state0, isabelle, session, 
                                     model, left, trace, "case", stage=2, prior_store=prior_store)

        if current_text != pre_repair_text:
            if verified or _verify_full_proof(isabelle, session, current_text, timeout_s=min(15, left())):
                return current_text, True, "stage=2 block:case"

            # If _repair_block updated the text, it has already verified internal score optimization.
            # Commit it immediately as progress rather than validating against a global threshold.
            if trace:
                print(f"[repair] Stage 2a partial progress accepted.")
            return current_text, False, "stage=2 block:case-partial"
        
        lines = current_text.splitlines()
        state0 = _print_state_before_hole(isabelle, session, current_text, hole_span, trace=trace)


    # Stage 2b: Subproof
    ps, pe = _enclosing_subproof(lines, focus_line)
    if resume_stage <= 2 and ps >= 0 and left() > 2.0:    # Fix: lower gate threshold from 3.0
        if trace:
            print("[repair] Trying subproof repair…")

        pre_repair_text = current_text
        current_text, verified = _repair_block(current_text, lines, ps, pe, goal_text, state0, isabelle, session, 
                                     model, left, trace, "subproof", stage=2, prior_store=prior_store)

        if current_text != pre_repair_text:
            if verified or _verify_full_proof(isabelle, session, current_text, timeout_s=min(15, left())):
                return current_text, True, "stage=2 block:subproof"

            # If _repair_block updated the text, it has already verified internal score optimization.
            # Commit it immediately as progress rather than validating against a global threshold.
            if trace:
                print(f"[repair] Stage 2b partial progress accepted.")
            return current_text, False, "stage=2 block:subproof-partial"
    
        lines = current_text.splitlines()
        state0 = _print_state_before_hole(isabelle, session, current_text, hole_span, trace=trace)

    best_candidate = current_text if current_text != full_text else full_text
    if best_candidate != full_text:
        return best_candidate, False, f"stage={resume_stage} cegis-partial"
    return full_text, False, f"stage={resume_stage} cegis-nohelp"

def _repair_block(current_text: str, lines: List[str], start: int, end: int, goal_text: str, 
                 state0: str, isabelle, session: str, model: Optional[str], left, trace: bool, 
                 block_type: str, stage: int, *, prior_store: Optional[Dict[str, List[str]]] = None) -> Tuple[str, bool]:
    """Returns a Tuple[str, bool] representing (patched_text, verified_fully)."""
    from planner.skeleton import _quick_sketch_score    # imported here to prevent circular imports

    _, errs = _quick_state_and_errors(isabelle, session, current_text)
    err_texts = _normalize_error_texts(errs)
    ce = get_counterexample_hints_for_repair(isabelle, session, state0, timeout_s=10)
    block = "\n".join(lines[start:end])

    # Track baseline score internal to the mutation loops
    base_score = _quick_sketch_score(isabelle, session, current_text, timeout_s=10, trace=trace)
    best_local_score = base_score
    best_text_so_far = current_text
    is_fully_verified = False
    
    # Extract proof context instead of using state block
    proof_context = _extract_proof_context(current_text, start)
    
    _log("repair", f"{block_type}-block (input)", block, trace=trace)
    _log("repair", "proof_context (LLM input)", proof_context, trace=trace)
    _log("repair", "errors (LLM input)", "\n".join(err_texts) or "(none)", trace=trace)
    ce_list = ce.get("bindings", []) + ce.get("def_hints", []) if isinstance(ce, dict) else []  
    _log("repair", "counterexamples (LLM input)", "\n".join(ce_list) or "(none)", trace=trace)
    rounds = 3 if left() >= 12.0 else 2 if left() >= 6.0 else 1    # Fix: reduced from 18.0 - 10.0 to 12.0-6.0
    mem = _RepairMemory()

    # Build proposals in a few rounds; track failures and surface them to the LLM
    timed_out = False
    for rr in range(rounds):
        if left() <= 3.0:
            timed_out = True
            break
        mem.rounds = rr + 1
        why = f"Previous {block_type}-block attempt did not solve the goal; try a different strategy."

        # #Fix: Cap per-round LLM timeout so the total across all rounds stays
        # #Fix: within budget AND leaves headroom for the Gemini inter-request
        # #Fix: delay.  Previously the timeout could consume the entire remaining
        # #Fix: budget on round 0, leaving rounds 1-2 with nothing.
        remaining = left()
        timeout = int(min(45, max(20, remaining * 0.45 / max(1, rounds - rr))))
        
        # Build prior failed blocks text (trim + separators)
        prior_blocks_for_type = list(prior_store.get(block_type, [])) if isinstance(prior_store, dict) else []
        seed_list = [block] + mem.prev_blocks + prior_blocks_for_type
        
        # De-dup while preserving order (by fingerprint)
        seen: Set[str] = set()
        uniq: List[str] = []
        for b in seed_list:
            fpb = _fingerprint_block(b)
            if fpb and fpb not in seen:
                seen.add(fpb); uniq.append(b)
        seed_list = uniq
        
        if seed_list:
            fails_txt = ("\n---\n".join(_trim_block_for_prompt(b) for b in seed_list[:_MAX_PREV_BLOCKS])) or "(none)"
            _log("repair", "prior_block_failures (LLM input)", fails_txt, trace=trace)
        else:
            fails_txt = "(none)"        
        
        try:
            blk = _propose_block_repair(
                goal=goal_text, errors=err_texts, ce_hints=ce, 
                proof_context=proof_context, block_type=block_type,
                block_text=block, model=model, timeout_s=timeout, why=why,
                prior_failed_blocks=fails_txt
            )
            if trace:
                print(f"blk:\n{blk}", flush=True)

            # remove this
#             blk = """
# also have "... = rev (xs @ ys) @ [x]"
#       by simp
# """
#             blk = _sanitize_llm_block(blk)
#             print(f"blk:\n{blk}", flush=True)

        except Exception as e:
            # #Fix: On any exception (including a 429 that slipped past raise_for_status,
            # #Fix: or a network blip), wait one full delay period before trying the
            # #Fix: next round rather than immediately hammering the API again.
            if trace:
                print(f"[repair] in _repair_block hit exception when calling _propose_block_repair: {type(e).__name__}: {e}", flush=True)
            if model and model.startswith("gemini:"):
                err_str = str(e)
                if "429" in err_str:
                    wait = 15.0
                    print(f"[repair] 429 rate limit, sleeping {wait}s...", flush=True)
                    time.sleep(wait)
                elif "timeout" in err_str.lower():
                    time.sleep(2.0)
            blk = ""
        
        if not _is_effective_block(blk):
            continue
        
        # STRICT DEDUP: If this block matches ANY prior failure, skip it immediately
        fp_new = _fingerprint_block(blk)
        all_prior_fps = set([_fingerprint_block(b) for b in (mem.prev_blocks + prior_blocks_for_type)])
        
        if fp_new in all_prior_fps:
            if trace:
                print(f"[repair] Skipping duplicate block (fingerprint: {fp_new[:8]}...)", flush=True)
            continue  # Don't even try to verify, just skip
        
        before = blk
        if block_type == "case":
            blk = _strip_wrapper_to_case_block(blk, block)
        elif block_type == "have-show":
            blk = _strip_wrapper_to_have_show(blk, block)
        elif block_type == "subproof":
            blk = _strip_wrapper_to_subproof(blk)              
        if blk.strip() == block.strip():
            continue 
        
        # Build the patched text from the raw LLM block (no sorry substitution yet)
        new_block_lines_raw = blk.splitlines()
        patched_raw = "\n".join(lines[:start] + new_block_lines_raw + lines[end:])

        # Score in FULL context first — this is what matters
        candidate_score = _quick_sketch_score(isabelle, session, patched_raw, timeout_s=10, trace=trace)
        if trace:
            print(f"[cegis-eval] candidate_score={candidate_score} best_local_score={best_local_score} same_text={patched_raw == current_text}", flush=True)

        if candidate_score <= best_local_score and patched_raw != current_text:
            if trace:
                print(f"[cegis-eval] Block mutation improved score {best_local_score} → {candidate_score}", flush=True)

            # Only now sorry-ify to get a safe storable version
            blk_with_sorry = _replace_failing_tactics_with_sorry(
                blk, full_text_lines=lines, start_line=start + 1,
                end_line=end + 1, isabelle=isabelle, session=session, trace=trace
            )

            # Extract the EXACT leading indentation of the first line we are replacing
            original_indent = ""
            if start < len(lines):
                original_indent = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
            
            # Apply that exact indentation to the replacement lines
            new_block_lines = []
            for i, line in enumerate(blk_with_sorry.splitlines()):
                if line.strip() == "":
                    new_block_lines.append("")
                elif i == 0 or not line.startswith(original_indent):
                    # Strip any weak formatting the LLM guessed and prepend the TRUE indent
                    new_block_lines.append(original_indent + line.lstrip())
                else:
                    new_block_lines.append(line)

            patched = "\n".join(lines[:start] + new_block_lines + lines[end:])
            best_local_score = candidate_score
            best_text_so_far = patched

            end = start + len(new_block_lines)

            if trace:
                print(f"patched:\n{patched}\n", flush=True)
                print(f"patched_raw:\n{patched_raw}\n", flush=True)

            if _verify_full_proof(isabelle, session, patched_raw, timeout_s=10):
                return patched, True
        else:
            # No improvement — sorry-ify just to record as prior failure
            blk_with_sorry = _replace_failing_tactics_with_sorry(
                blk, full_text_lines=lines, start_line=start + 1,
                end_line=end + 1, isabelle=isabelle, session=session, trace=trace
            )

        _log("repair", f"{block_type}-block (output)", blk_with_sorry, trace=trace)  

        # Record this failed candidate into local and shared stores (so next round tries differ)
        fp = _fingerprint_block(blk_with_sorry)
        if fp and fp not in mem.prev_fps:
            mem.prev_fps.add(fp)
            mem.prev_blocks.insert(0, blk_with_sorry)
            mem.prev_blocks = mem.prev_blocks[:_MAX_PREV_BLOCKS]
            if isinstance(prior_store, dict):
                lst = prior_store.setdefault(block_type, [])
                # De-dup in shared store too
                if fp not in [_fingerprint_block(x) for x in lst]:
                    lst.insert(0, blk_with_sorry)
                    del lst[_MAX_PREV_BLOCKS:]   
        
        # === SYSTEMATIC LINE ALIGNMENT FIX ===
        # Capture the length before we update current_text
        old_num_lines = len(lines)
        
        current_text = best_text_so_far
        lines = current_text.splitlines()
        new_num_lines = len(lines)
        
        # If the file length shifted because an earlier round's patch was accepted,
        # adjust both tracking metrics by the precise layout delta.
        if new_num_lines != old_num_lines:
            line_delta = new_num_lines - old_num_lines
            end = end + line_delta
            
        proof_context = _extract_proof_context(current_text, start)
    
    # If we exited due to a timer expiration, let the caller know it was an invalid run
    if timed_out and best_text_so_far == current_text:
        print(f"Timed out")
        return current_text, False

    # Always return a tuple matching (patched_text, verified_fully)
    return best_text_so_far, is_fully_verified

# ---------- Public helper: whole-proof regeneration with prior-failure banlist ----------
def regenerate_whole_proof(*, full_text: str, goal_text: str, model: Optional[str],
                           isabelle, session: str, budget_s: float = 20.0,
                           trace: bool = False, prior_outline_text: Optional[str] = None
                          ) -> Tuple[str, bool, str]:
    """
    Re-generate the last proof..qed block (or from the lemma head to EOF if no qed yet),
    feeding decisive lines from `prior_outline_text` as a ban list so the LLM avoids
    repeating previously failed tactics. Only returns a patched text if it *verifies*.
    """
    lines = full_text.splitlines()
    ws, we = _enclosing_whole_proof(lines)
    if ws < 0 or we <= ws:
        # Fallback: from first lemma/theorem head to EOF
        start = None
        for i, L in enumerate(lines):
            if re.match(r"^\s*(?:lemma|theorem|corollary)\b", L):
                start = i
                break
        if start is None:
            return full_text, False, "whole:region-not-found"
        ws, we = start, len(lines)

    # Simple local timer for the block repair
    t0 = time.monotonic()
    left = lambda: max(0.0, budget_s - (time.monotonic() - t0))
    # Use empty/quick state — the block prompt already carries enough context
    state0 = ""
    # Seed prior failed blocks with the previous outline (so the first round won't repeat it)
    prior_store: Dict[str, List[str]] = {}
    if prior_outline_text:
        prior_store["whole"] = [prior_outline_text]

    # Unpack the updated tuple payload format cleanly here
    patched, verified = _repair_block(full_text, lines, ws, we, goal_text, state0, isabelle, session,
                            model, left, trace, "whole", stage=3, prior_store=prior_store)
    if verified:
        # _repair_block only returns a different text if it verified successfully
        return patched, True, "regen:whole-proof"
    
    return full_text, False, "regen:no-change"
