from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict, Iterable
import json
import os
import re
import time                          # Fix: needed for per-request sleep
from functools import lru_cache
from planner.prompts import SKELETON_PROMPT
import requests

# Pull defaults from your existing prover/config.py
from prover.config import (
    MODEL as DEFAULT_MODEL,
    OLLAMA_HOST,
    TIMEOUT_S as OLLAMA_TIMEOUT_S,
    OLLAMA_NUM_PREDICT,
    TEMP as OLLAMA_TEMP,
    TOP_P as OLLAMA_TOP_P,
)

# Isabelle helpers (for quick sketch check)
from prover.isabelle_api import build_theory, run_theory, last_print_state_block, _normalize_type, _get_field, _decode_body_to_dict
from prover.utils import parse_subgoals

# Reuse local-context miner from repair (defs/facts list)
from planner.repair import _facts_from_state as _facts_from_state

# One HTTP session (keep-alive)
_SESSION = requests.Session()

@dataclass(slots=True)
class Skeleton:
    text: str
    holes: List[Tuple[int, int]]  # (start_idx, end_idx) spans where 'sorry' occurs

SORRY_RE = re.compile(r"\bsorry\b")
PROOF_RE = re.compile(r"(?m)^\s*proof(?:\b|\s|\()", re.UNICODE)
QED_RE   = re.compile(r"(?m)^\s*qed\b", re.UNICODE)
BY_INLINE_RE = re.compile(r"\s+by\s+.*$")
# New regex helpers for sanitization
CASE_START_RE   = re.compile(r"(?m)^\s*case\b")
NEXT_OR_QED_RE  = re.compile(r"(?m)^\s*(next|qed)\b")
SHOW_THESIS_RE = re.compile(r"(?m)^\s*(?:then\s+)?show\s+\?thesis\b")
# Match the meta-variable immediately after 'show', capturing the 'show ' prefix so we can preserve any
# leading tokens like 'then', 'from ...', 'with ...', 'finally', etc. We only rewrite the '?thesis|?case' token.
SHOW_META_AT_SHOW = re.compile(r"(?m)(?P<prefix>\bshow\s+)\?(?P<meta>thesis|case)\b")
BARE_PROOF_RE  = re.compile(r"(?m)^\s*proof\s*$")
# General proof-mode detectors (for ?case/?thesis normalization)
_PROOF_OPEN_RE   = re.compile(r"(?m)^\s*proof(?:\s*\(([^)]+)\))?\s*$")
_QED_LINE_RE     = re.compile(r"(?m)^\s*qed\b")
_MODE_CASES_RE   = re.compile(r"^\s*cases\b")
_MODE_CASES_RULE = re.compile(r"^\s*cases\s+rule:")
_MODE_INDUCT_RE  = re.compile(r"^\s*(?:induction|induct|coinduction|coinduct)\b")
_HAS_TACTIC_NEXT = re.compile(r"(?m)^\s*(?:by\b|apply\b|proof\b|sorry\b|done\b)")
_HAVE_OR_SHOW    = re.compile(r"(?m)^\s*(have|show)\b")
_INLINE_BY       = re.compile(r"\s+by\s+.+$")
_CONTINUATION_HEAD = re.compile(r"(?m)^\s*(?:using|from|with|then|ultimately|finally|also|moreover)\b")
_STMT_OR_BOUNDARY = re.compile(r"(?m)^\s*(?:have|show|assume|case|next|qed)\b")

# =============================================================================
# Provider shims: Ollama (default), Hugging Face ("hf:"), Gemini ("gemini:")
# =============================================================================

def _ollama_generate_simple(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    num_predict: Optional[int] = None,
    timeout_s: Optional[int] = None,
) -> str:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    payload = {
        "model": model or DEFAULT_MODEL,
        "prompt": prompt,
        "options": {
            "temperature": OLLAMA_TEMP if temperature is None else temperature,
            "top_p": OLLAMA_TOP_P if top_p is None else top_p,
            "num_predict": OLLAMA_NUM_PREDICT if num_predict is None else num_predict,
        },
        "stream": False,
    }
    resp = _SESSION.post(url, json=payload, timeout=timeout_s or OLLAMA_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("response") or "").strip()

def _hf_generate_simple(
    prompt: str,
    model_id: str,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    timeout_s: Optional[int] = None,
) -> str:
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_API_TOKEN is not set")
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}
    payload: Dict[str, Any] = {
        "inputs": prompt,
        "parameters": {
            "temperature": OLLAMA_TEMP if temperature is None else temperature,
            "top_p": OLLAMA_TOP_P if top_p is None else top_p,
            "max_new_tokens": OLLAMA_NUM_PREDICT if max_new_tokens is None else max_new_tokens,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }
    resp = _SESSION.post(url, headers=headers, json=payload, timeout=timeout_s or OLLAMA_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data:
        return (data[0].get("generated_text") or "").strip()
    if isinstance(data, dict):
        if "generated_text" in data:
            return (data["generated_text"] or "").strip()
        choices = data.get("choices") or []
        if choices:
            t = choices[0].get("text") or choices[0].get("generated_text") or ""
            return str(t).strip()
    return str(data).strip()

@lru_cache(maxsize=1)
def _gemini_list_models_cached(api_key: str) -> List[str]:
    if not api_key:
        return []
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        resp = _SESSION.get(url, timeout=OLLAMA_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        out = []
        for m in data.get("models", []):
            name = m.get("name", "")
            short = name.split("/")[-1] if name else ""
            if short:
                out.append(short)
        return out
    except Exception:
        return []

def _gemini_resolve_model_id(model_id: str, *, timeout_s: Optional[int] = None) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return model_id
    models = _gemini_list_models_cached(api_key)
    if model_id in models:
        return model_id
    cands = [m for m in models if m.startswith(model_id)]
    if cands:
        stable = [m for m in cands if ("preview" not in m and "exp" not in m)]
        return (stable or cands)[0]
    return model_id

# #Fix: cache _gemini_cli_available() so shutil.which is not called on every
# #Fix: iteration of the temperature loop (was firing once per LLM call).
@lru_cache(maxsize=1)
def _gemini_cli_available() -> bool:
    from shutil import which
    return which("gemini") is not None

def _gemini_cli_generate_simple(prompt: str, model_id: str, *, timeout_s: Optional[int] = None) -> str:
    import subprocess
    cmd = ["gemini", "-m", model_id]
    proc = subprocess.run(
        cmd, input=prompt, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=timeout_s or OLLAMA_TIMEOUT_S, env=os.environ.copy()
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gemini CLI failed ({proc.returncode}): {(proc.stderr or proc.stdout).strip()}")
    return proc.stdout.strip()

def _gemini_rest_generate_simple(prompt: str, model_id: str, *, timeout_s: Optional[int] = None) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set (needed for Gemini REST)")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"

    # Fix: don't use OLLAMA_NUM_PREDICT for Gemini — it's far too small
    gemini_max_tokens = max(OLLAMA_NUM_PREDICT, 4096)
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": gemini_max_tokens}}
    resp = _SESSION.post(url, json=body, timeout=timeout_s or OLLAMA_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    try:
        cands = data.get("candidates") or []
        if cands:
            parts = ((cands[0].get("content") or {}).get("parts")) or []
            if parts:
                return (parts[0].get("text") or "").strip()
    except Exception:
        pass
    return str(data).strip()

def _gemini_generate_simple(prompt: str, model_id: str, *, timeout_s: Optional[int] = None) -> str:
    # #Fix: resolve the model ID exactly once, outside any retry/fallback branch.
    # #Fix: Previously this was called inside a loop context, contributing extra
    # #Fix: HTTP hits to /v1beta/models on every temperature iteration.
    resolved = _gemini_resolve_model_id(model_id, timeout_s=timeout_s)

    # #Fix: Try CLI first if available, then fall through to REST exactly once.
    # #Fix: The old code had a second fallback block that repeated both CLI and
    # #Fix: REST attempts with a hardcoded "gemini-2.5-pro" model, turning one
    # #Fix: logical request into up to 4 HTTP calls. Now we make at most 2 attempts
    # #Fix: (CLI → REST) with no further silent retry loops.
    if _gemini_cli_available():
        try:
            return _gemini_cli_generate_simple(prompt, resolved, timeout_s=timeout_s)
        except Exception:
            pass  # CLI failed; fall through to REST once only

    # #Fix: Single REST attempt — no secondary fallback loop after this.
    return _gemini_rest_generate_simple(prompt, resolved, timeout_s=timeout_s)

# #Fix: Add a per-call inter-request delay (seconds) for Gemini to avoid
# #Fix: exceeding the API's requests-per-minute quota when looping over
# #Fix: multiple temperatures in propose_isar_skeletons.
_GEMINI_INTER_REQUEST_DELAY_S: float = 1.0

def _generate_simple(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    num_predict: Optional[int] = None,
    timeout_s: Optional[int] = None,
) -> str:

    display_model = model or DEFAULT_MODEL
    dump = os.getenv("LLM_DUMP", "").strip().lower() in ("1", "true", "yes", "on")
    if dump:
        print(f"{'='*60}", flush=True)
        print(f"[Skeleton] LLM Prompt:\n{prompt.rstrip()}", flush=True)
        print(f"{'-'*60}", flush=True)

    if model:
        if model.startswith("hf:"):
            raw = _hf_generate_simple(
                prompt, model_id=model[len("hf:"):],
                temperature=temperature, top_p=top_p,
                max_new_tokens=num_predict, timeout_s=timeout_s
            )
        elif model.startswith("gemini:"):
            raw = _gemini_generate_simple(
                prompt, model_id=model[len("gemini:"):],
                timeout_s=timeout_s
            )
            # #Fix: Sleep after every Gemini REST/CLI call so back-to-back
            # #Fix: temperature iterations don't all fire within the same second
            # #Fix: and trip the 429 quota. Adjust _GEMINI_INTER_REQUEST_DELAY_S
            # #Fix: if your quota tier allows faster throughput.
            time.sleep(_GEMINI_INTER_REQUEST_DELAY_S)
        elif model.startswith("ollama:"):
            model = model[len("ollama:"):]
            raw = _ollama_generate_simple(
                prompt, model=model, temperature=temperature, top_p=top_p,
                num_predict=num_predict, timeout_s=timeout_s
            )
        else:
            raw = _ollama_generate_simple(
                prompt, model=model, temperature=temperature, top_p=top_p,
                num_predict=num_predict, timeout_s=timeout_s
            )
    else:
        raw = _ollama_generate_simple(
            prompt, model=model, temperature=temperature, top_p=top_p,
            num_predict=num_predict, timeout_s=timeout_s
        )

    if dump:
        print(f"[Skeleton] model={display_model}", flush=True)
        print(f"[Skeleton] LLM Output:\n{raw}", flush=True)
        print(f"{'='*60}", flush=True)

    return raw

# -----------------------------------------------------------------------------
# Utilities: sorry spans, sanitize, state block, facts, scoring
# -----------------------------------------------------------------------------

def find_sorry_spans(isar: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in SORRY_RE.finditer(isar)]

def _ensure_lemma_header(text: str, goal: str) -> str:
    body = text.lstrip()
    if not body.startswith("lemma"):
        return f'lemma "{goal}"\n{body}'
    return text

def _normalize_calculation_ellipsis(text: str) -> str:
    # Replace Unicode ellipsis and spaced PDF (. . .) with Isar token "..."
    text = text.replace("…", "...")
    text = re.sub(r"\.\s*\.\s*\.", "...", text)
    return text

def _crop_to_first_proof_block(text: str) -> str:
    """
    Keep only the first lemma..(proof..qed)* block; be nesting-aware so the cropped
    text ends at the *matching* 'qed' for the first 'proof', not at an inner one.
    """
    # Find the first lemma line
    m_lemma = re.search(r'(?m)^\s*lemma\s+"[^"]*"', text)
    if not m_lemma:
        return text
    tail = text[m_lemma.start():]

    # Find the first 'proof' after that lemma
    m_proof = PROOF_RE.search(tail)
    if not m_proof:
        # No proof — still return from lemma onward (sanitizers may add a skeleton later)
        return tail

    # Walk forward and balance nested 'proof'/'qed'
    depth = 1
    end_idx = None
    pos = m_proof.end()

    # We want earliest upcoming PROOF or QED at each step
    while True:
        m_next_proof = PROOF_RE.search(tail, pos)
        m_next_qed   = QED_RE.search(tail, pos)

        if not m_next_proof and not m_next_qed:
            # No closing 'qed' — return as-is; later passes may append a final 'qed'
            break

        # Choose whichever comes first
        choose_qed = False
        if m_next_qed and m_next_proof:
            choose_qed = (m_next_qed.start() <= m_next_proof.start())
        elif m_next_qed and not m_next_proof:
            choose_qed = True
        else:
            choose_qed = False

        if choose_qed:
            depth -= 1
            pos = m_next_qed.end()
            if depth == 0:
                end_idx = pos
                break
        else:
            depth += 1
            pos = m_next_proof.end()

    if end_idx is None:
        # Could not find the matching outer 'qed'; return tail as-is (other sanitizers add one)
        return tail

    return tail[:end_idx] + "\n"

def _normalize_show_kinds(text: str) -> str:
    """
    Flip ONLY the '?thesis'/'?case' meta after 'show', preserving any prefix (e.g., 'then', 'from', 'with', 'using', 'finally').
    Policy (line-local, nesting-aware):
      • Inside branches of (co)induction → 'show ?case'
      • Inside branches of 'proof (cases …)' (incl. 'cases rule: …') → 'show ?thesis'
      • Outside any explicit case branch → default to 'show ?thesis'
    """
    lines = text.splitlines()
    # Track current proof mode for the *innermost* open proof
    stack: list[str] = []  # values: "induct" | "cases" | "plain"
    in_case_branch = False
    for i, L in enumerate(lines):
        m_open = _PROOF_OPEN_RE.match(L)
        if m_open:
            mode = (m_open.group(1) or "").strip()
            if not mode:
                stack.append("plain")
            elif _MODE_INDUCT_RE.match(mode):
                stack.append("induct")
            elif _MODE_CASES_RULE.match(mode) or _MODE_CASES_RE.match(mode):
                stack.append("cases")
            else:
                stack.append("plain")
            continue
        if _QED_LINE_RE.match(L):
            if stack:
                stack.pop()
            continue
        if CASE_START_RE.match(L):
            in_case_branch = True
            continue
        if NEXT_OR_QED_RE.match(L):
            in_case_branch = False
            continue

        def _repl(m: "re.Match[str]") -> str:
            current = stack[-1] if stack else "plain"
            want = "case" if (in_case_branch and current == "induct") else "thesis"
            # If already correct, return unchanged
            if m.group("meta") == want:
                return m.group(0)
            return f'{m.group("prefix")}?{want}'

        # Rewrite at most once per line; this keeps everything except the meta token intact.
        lines[i] = SHOW_META_AT_SHOW.sub(_repl, L, count=1)
    return "\n".join(lines)

def _drop_redundant_sorry(text: str) -> str:
    """Remove a `sorry` that directly follows a finisher.

    Models frequently emit BOTH a real finisher and a trailing `sorry` for the
    same obligation, e.g.

        have f1: "..."
          using a1 by simp
          sorry            <-- illegal: the `have` is already closed by `by simp`

    The skeleton prompt's examples only ever model `sorry` at every leaf, which
    nudges the model toward appending `sorry` even when it also supplied a `by`.
    `_ensure_have_show_bodies` only *adds* missing bodies; it never removes this
    redundant `sorry`, so the malformed pair survives and aborts the proof before
    the fill/repair stages can run. This pass deletes the dangling `sorry` when the
    nearest preceding non-blank line already closed the goal.

    A line "closes the goal" if it is `done`, a bare `.`/`..`, a standalone
    `by <method>`, or ends in an inline ` by <method>` (e.g. `using a1 by simp`).
    We intentionally do NOT treat `proof`/`next`/`qed`/`case` as closers, so we
    never strip a `sorry` that is the legitimate body of a freshly opened goal.

    Additionally collapses *duplicate* sorries: a standalone `sorry` whose nearest
    preceding non-blank line already ends the obligation with `sorry` (either a
    standalone `sorry` or an inline `... sorry`, e.g. `have f4: "..." sorry`). The
    model sometimes emits both an inline and a trailing sorry for one `have`, which
    is malformed; we keep the first and drop the redundant follow-on.
    """
    lines = text.splitlines()
    out: List[str] = []
    # Index, in `out`, of the most recent non-blank line (for back-reference).
    last_nonblank = -1
    closer_by = re.compile(r"(?m)^\s*by\b")
    closer_done = re.compile(r"(?m)^\s*(?:done|\.\.?)\s*$")
    inline_by = re.compile(r"\s+by\s+\S")
    ends_in_sorry = re.compile(r"\bsorry\s*$")
    for L in lines:
        if SORRY_RE.search(L) and L.strip() == "sorry" and last_nonblank >= 0:
            prev = out[last_nonblank]
            # (a) redundant after a real finisher
            if closer_by.match(prev) or closer_done.match(prev) or inline_by.search(prev):
                continue
            # (b) duplicate sorry: the obligation is already terminated by a sorry
            #     on the previous non-blank line (standalone or inline).
            if ends_in_sorry.search(prev):
                continue
        out.append(L)
        if L.strip() != "":
            last_nonblank = len(out) - 1
    return "\n".join(out)

def _ensure_have_show_bodies(text: str) -> str:
    """
    Ensure every 'have …' / 'show …' has a body, but *preserve* local continuations:
    scan forward across lines starting with using/from/with/then/also/moreover/finally,
    and only insert 'sorry' at the *end* of that block if no tactic/proof body occurs
    before the next statement/boundary (have/show/assume/case/next/qed).
    """
    lines = text.splitlines()
    i, n = 0, len(lines)
    out: List[str] = []
    while i < n:
        L = lines[i]
        out.append(L)
        if _HAVE_OR_SHOW.match(L) and not _INLINE_BY.search(L):
            j = i + 1
            # skip blank lines
            while j < n and lines[j].strip() == "":
                out.append(lines[j]); j += 1
            # consume a local continuation block
            saw_body = False
            k = j
            while k < n:
                Nk = lines[k]
                if _HAS_TACTIC_NEXT.match(Nk):
                    saw_body = True
                    break
                if _STMT_OR_BOUNDARY.match(Nk):
                    break
                if _CONTINUATION_HEAD.match(Nk) or Nk.strip() == "":
                    out.append(Nk); k += 1
                    continue
                # unknown line: stop the block here
                break
            if not saw_body:
                indent = L[:len(L) - len(L.lstrip(" "))]
                out.append(f"{indent}  sorry")
            i = k
            continue
        i += 1
    return "\n".join(out)

def _maybe_proof_dash(text: str) -> str:
    """
    If there is a bare 'proof' at top-level and calculational cues present, prefer 'proof -'.
    """
    if not BARE_PROOF_RE.search(text):
        return text
    if re.search(r"(?m)^\s*(have|also|moreover|ultimately|finally|hence|thus)\b", text):
        return BARE_PROOF_RE.sub("proof -", text, count=1)
    return text

def _sanitize_outline(text: str, goal: str, *, force_outline: bool) -> str:
    text = _ensure_lemma_header(text, goal)
    # Normalize ellipsis first (avoid Unicode / spaced form)
    text = _normalize_calculation_ellipsis(text)

    # Fix 5: replace illegal 'by sorry' with just 'sorry'
    text = re.sub(r'\bby\s+sorry\b', 'sorry', text)
    # Fix 5: replace 'using ... sorry' inline patterns
    text = re.sub(r'\busing\s+\S+\s+sorry\b', 'sorry', text)

    # Keep content from *this* lemma onwards
    goal_header = f'lemma "{goal}"'
    idx = text.find(goal_header)
    if idx >= 0:
        text = text[idx:]
    else:
        first_lemma = text.find("lemma ")
        if first_lemma >= 0:
            text = text[first_lemma:]

    # Ensure proof/qed skeleton exists — UNLESS the lemma is already closed
    # by a one-line finisher (by ... / done). A one-liner is a complete,
    # valid proof without proof/qed, so adding them creates a dangling block.
    _ONELINE_FINISHER_RE = re.compile(
        r'(?ms)^\s*lemma\s+"[^"]*"\s*\n\s*(?:by\b[^\n]*|done)\s*\Z'
    )
    already_complete = bool(_ONELINE_FINISHER_RE.match(text.rstrip() + "\n"))

    if not already_complete:
        if not PROOF_RE.search(text):
            text = text.rstrip() + "\nproof\n  sorry\nqed\n"
        if not QED_RE.search(text):
            text = text.rstrip() + "\nqed\n"

    # Force an outline (remove inline 'by' if requested by caller)
    if force_outline:
        lines = text.splitlines()
        for i, L in enumerate(lines):
            if " by " in L:
                lines[i] = BY_INLINE_RE.sub(" sorry", L)
        text = "\n".join(lines)
        if "sorry" not in text:
            m_qed = QED_RE.search(text)
            if m_qed:
                insert_at = m_qed.start()
                text = text[:insert_at] + "  sorry\n" + text[insert_at:]

    # Light Isar fixups (order matters)
    #  1) Flip only the meta after 'show', preserving 'then/using/from/with/finally' etc.
    #  1b) Drop any 'sorry' that redundantly follows a 'by'/'done' finisher.
    #  2) Ensure every 'have/show' has a body; insert 'sorry' if missing to trigger fill/repair.
    #  3) Prefer 'proof -' when calculational cues are present.
    text = _normalize_show_kinds(text)
    text = _drop_redundant_sorry(text)
    text = _ensure_have_show_bodies(text)
    text = _maybe_proof_dash(text)

    # Trim to the first complete lemma..qed block to avoid trailing splices
    text = _crop_to_first_proof_block(text)

    if not text.endswith("\n"):
        text += "\n"
    return text

def _quick_sketch_score(isabelle, session_id: str, outline_text: str, *, timeout_s: int = 5, trace: bool = False) -> int:
    """
    Returns number of remaining subgoals, 0 if complete, or 9999 on error.
    Reads directly from the FINISHED response rather than print_state.
    """
    try:
        thy = build_theory(outline_text.splitlines(), add_print_state=False, end_with=None)
        if trace:
            print(f"[Skeleton] theory:\n{thy}", flush=True)

        resps = run_theory(isabelle, session_id, thy, timeout_s=timeout_s)
        if not resps:
            if trace:
                print(f"[Skeleton] no responses from run_theory", flush=True)
            return 9999
        if trace:
            print(f"[Skeleton] resps:", flush=True)
            for resp in resps:
                print(f"\t\t{resp}", flush=True)

        for r in reversed(resps):
            if _normalize_type(_get_field(r, ("response_type", "type", "kind", "tag", "name"))) != "FINISHED":
                continue
            obj = _decode_body_to_dict(_get_field(r, ("response_body", "body", "message", "payload")))
            if not isinstance(obj, dict):
                continue
            
            # Compute both failure and sorry counts before deciding score
            sorry_count = len(find_sorry_spans(outline_text))
            nodes = obj.get("nodes") or []
            failed_count = sum((n.get("status") or {}).get("failed", 0) for n in nodes)

            if obj.get("ok") is True and sorry_count == 0:
                if trace:
                    print(f"[Skeleton] proof complete (ok=true, no sorries) → score 0", flush=True)
                return 0

            for node in nodes:
                for msg in (node.get("messages") or []):
                    text = str(msg.get("message", ""))
                    if "goal (" in text or "subgoal" in text:
                        n = parse_subgoals(text)
                        if isinstance(n, int):
                            return n
                        
            score = failed_count + sorry_count
            if trace:
                print(f"[Sketch] ok=false → failures={failed_count} sorries={sorry_count} score={score}", flush=True)
            return score

        if trace:
            print(f"[Skeleton] no FINISHED response in {len(resps)} responses", flush=True)
        return 9999

    except Exception as e:
        if trace:
            print(f"[Skeleton] FAILED: {type(e).__name__}: {e}", flush=True)
        return 9999

def _state_block_for_goal(isabelle, session_id: str, goal: str) -> str:
    mini = f'lemma "{goal}"\nproof\n  sorry\nqed\n'
    try:
        thy = build_theory(mini.splitlines(), add_print_state=True, end_with="sorry")
        resps = run_theory(isabelle, session_id, thy)
        return last_print_state_block(resps) or ""
    except Exception:
        return ""

# --- Pattern detection / simple priors ---

_PAT_INDUCTION = re.compile(r"(?m)^\s*proof\s*\(induction\b", re.UNICODE)
_PAT_CASES     = re.compile(r"(?m)^\s*proof\s*\(cases\b", re.UNICODE)
_PAT_CASES_RULE= re.compile(r"(?m)^\s*proof\s*\(cases\s+rule:\s*([A-Za-z0-9_\.]+)", re.UNICODE)
_ID = r"[A-Za-z_][A-Za-z0-9_']*"

def _detect_pattern_key(outline: str) -> str:
    if _PAT_INDUCTION.search(outline):
        return "induction"
    m = _PAT_CASES_RULE.search(outline)
    if m:
        return f"cases_rule:{m.group(1)}"
    if _PAT_CASES.search(outline):
        return "cases"
    return "plain"

def _tokenize_goal(goal: str) -> set:
    toks = set(re.findall(_ID, goal))
    if "@" in goal: toks.add("@")
    if "⟹" in goal: toks.add("implies")
    return toks

def _load_priors(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Expect either {"rules":[...]} or a bare list of rules.
        if isinstance(data, dict) and isinstance(data.get("rules"), list):
            return data["rules"]
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []

def _pattern_penalty(goal: str, outline: str, rules: List[Dict[str, Any]]) -> float:
    """
    Lower is better. Simple heuristic + optional JSON rules.
    """
    key = _detect_pattern_key(outline)
    toks = _tokenize_goal(goal)
    pen = 0.0
    # Built-in gentle priors
    if ("@" in toks or "map" in toks) and key != "induction":
        pen += 0.4
    if ({"Suc", "0"} & toks) and key != "induction":
        pen += 0.3
    if ({"True", "False"} & toks) and not key.startswith("cases"):
        pen += 0.25
    if ({"Some", "None", "option"} & toks) and "cases_rule:option.exhaust" != key and key != "cases":
        pen += 0.25
    if ({"Inl", "Inr"} & toks) and "cases_rule:sum.exhaust" != key and key != "cases":
        pen += 0.25
    # Optional external rules
    for r in rules:
        cond = set(map(str, r.get("if_any_tokens", [])))
        prefer = set(map(str, r.get("prefer_patterns", [])))
        weight = float(r.get("weight", 0.3))
        if cond and (cond & toks) and key and prefer and key not in prefer:
            pen += weight
    return pen

def _hint_bonus_from_outline(outline: str, recommended: List[str]) -> int:
    if not recommended:
        return 0
    # Count how many recommended tokens appear in the outline text (rough proxy)
    text = outline
    c = 0
    for h in recommended[:10]:
        if h in text:
            c += 1
    return c

# -----------------------------------------------------------------------------
# NEW: Hint lexicon (micro-RAG) utilities
# -----------------------------------------------------------------------------

def _load_hintlex(path: Optional[str]) -> Dict[str, List[str]]:
    """
    Returns token -> [hint,...]. Accepts either {"token":[["hint",count],...]} or {"token":["hint",...]}.
    """
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    out: Dict[str, List[str]] = {}
    for tok, val in raw.items():
        if isinstance(val, list):
            if val and isinstance(val[0], list):
                out[tok] = [h for h, _c in val]
            else:
                out[tok] = [str(h) for h in val]
    return out

def _hints_from_hintlex(goal: str, hintlex: Dict[str, List[str]], top: int = 8) -> List[str]:
    toks = _tokenize_goal(goal)
    got: List[str] = []
    for t in toks:
        hs = hintlex.get(t)
        if not hs:
            continue
        for h in hs[:top]:
            got.append(h)
    # stable de-dup
    return list(dict.fromkeys(got))

# -----------------------------------------------------------------------------
# Outline generators
# -----------------------------------------------------------------------------

def propose_isar_skeleton(
    goal: str,
    model: Optional[str] = None,
    temp: float = 0.35,
    *,
    force_outline: bool = False,
    hints: Optional[List[str]] = None,
    trace: bool = False,
) -> Skeleton:
    # Inject tiny hint list when available (keeps default behavior if None/empty)
    prompt = SKELETON_PROMPT.format(goal=goal)
    if hints:
        prompt += "\nHINTS: Prefer using " + ", ".join(sorted(set(hints))) + " if applicable.\n"
    if trace:
        print(
            f"[Skeleton] skeleton candidate 1/1: temp={temp} timeout={OLLAMA_TIMEOUT_S}s",
            flush=True,
        )
    raw = _generate_simple(
        prompt=prompt,
        model=model or DEFAULT_MODEL,
        temperature=temp,
        timeout_s=OLLAMA_TIMEOUT_S,
    )
    cleaned = _sanitize_outline(raw, goal=goal, force_outline=force_outline)
    return Skeleton(text=cleaned, holes=find_sorry_spans(cleaned))

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import threading

def propose_isar_skeletons(
    goal: str,
    *,
    model: Optional[str] = None,
    temps: Iterable[float] = (0.3, 0.5, 0.8),
    k: Optional[int] = None,
    force_outline: bool = False,
    hints: Optional[List[str]] = None,
    timeout_s: Optional[int] = None,    # Fix 2
    trace: bool = False,
) -> List[Skeleton]:
    temps_list = list(temps)[:k] if k else list(temps)
    n_calls = len(temps_list)

    # For API models: parallel. For local: sequential.
    is_api_model = model and (
        model.startswith("gemini:") or model.startswith("hf:")
    )

    if not is_api_model or n_calls == 1:
        # Original sequential path — safe for Ollama
        deadline = (time.monotonic() + float(timeout_s)) if timeout_s else None
        seen, out = set(), []
        for i, t in enumerate(temps_list):
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 1.0:
                    break
                calls_left = max(1, n_calls - i)
                per_call_timeout = max(1, int(remaining / calls_left))
            else:
                per_call_timeout = OLLAMA_TIMEOUT_S
            if trace:
                print(
                    f"[Skeleton] skeleton candidate {i + 1}/{n_calls}: "
                    f"temp={t} timeout={per_call_timeout}s",
                    flush=True,
                )
            prompt = SKELETON_PROMPT.format(goal=goal)
            if hints:
                prompt += "\nHINTS: Prefer using " + \
                          ", ".join(sorted(set(hints))) + " if applicable.\n"
            try:
                raw = _generate_simple(prompt=prompt, model=model or DEFAULT_MODEL,
                                       temperature=float(t), timeout_s=per_call_timeout)
            except Exception as e:
                if trace:
                    print(f"[Skeleton] call at temp={t} failed: {type(e).__name__}: {e}", flush=True)
                continue
            if trace:
                print(
                    f"[Skeleton] candidate {i + 1}/{n_calls}: LLM returned {len(raw or '')} chars; sanitizing...",
                    flush=True,
                )
            # #Fix: The inter-request sleep for Gemini is already applied inside
            # #Fix: _generate_simple after every gemini: call, so no extra sleep needed here.
            cleaned = _sanitize_outline(raw, goal=goal,
                                                  force_outline=force_outline)
            sk = Skeleton(text=cleaned, holes=[])
            sk.holes = find_sorry_spans(sk.text)
            if sk.text.strip() not in seen:
                seen.add(sk.text.strip())
                out.append(sk)
                if trace:
                    print(
                        f"[Skeleton] candidate {i + 1}/{n_calls}: kept outline "
                        f"({len(sk.text)} chars, {len(sk.holes)} sorry holes)",
                        flush=True,
                    )
            elif trace:
                print(f"[Skeleton] candidate {i + 1}/{n_calls}: duplicate outline skipped", flush=True)
            if k is not None and len(out) >= int(k):
                break
        if out:
            if trace:
                print(f"[Skeleton] collected {len(out)} unique skeleton candidate(s)", flush=True)
            return out
        fallback = f'lemma "{goal}"\n  sorry\n'
        if trace:
            print("[Skeleton] no usable skeleton candidates; using minimal sorry outline", flush=True)
        return [Skeleton(text=fallback, holes=find_sorry_spans(fallback))]

    # API parallel path
    # Each worker gets the full timeout — they run concurrently so wall-clock ~= one call
    per_call_timeout = max(1, int((timeout_s or OLLAMA_TIMEOUT_S) * 0.85))
    results = {}
    errors = {}
    lock = threading.Lock()

    def _one_call(i, t):
        if trace:
            print(
                f"[Skeleton] skeleton candidate {i + 1}/{n_calls}: "
                f"temp={t} timeout={per_call_timeout}s",
                flush=True,
            )
        prompt = SKELETON_PROMPT.format(goal=goal)
        if hints:
            prompt += "\nHINTS: Prefer using " + \
                      ", ".join(sorted(set(hints))) + " if applicable.\n"
        try:
            raw = _generate_simple(prompt=prompt, model=model or DEFAULT_MODEL,
                                   temperature=float(t), timeout_s=per_call_timeout)
            sk = Skeleton(text=_sanitize_outline(raw, goal=goal,
                                                  force_outline=force_outline), holes=[])
            sk.holes = find_sorry_spans(sk.text)
            with lock:
                results[t] = sk
        except Exception as e:
            with lock:
                errors[t] = f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=n_calls) as ex:
        futures = [ex.submit(_one_call, i, t) for i, t in enumerate(temps_list)]
        # Wait with hard deadline — don't hang forever
        wait(futures, timeout=per_call_timeout + 3)
        # Futures still running after timeout are abandoned (can't kill, but
        # they'll timeout on their own HTTP call within per_call_timeout)

    # Log any failures explicitly — addresses the hidden failure concern
    for t, err in errors.items():
        print(f"[Skeleton] temp={t} failed: {err}", flush=True)

    # Restore deterministic order by temperature
    seen, out = set(), []
    for t in temps_list:
        if t in results:
            sk = results[t]
            if sk.text.strip() not in seen:
                seen.add(sk.text.strip())
                out.append(sk)

    if out:
        return out
    fallback = f'lemma "{goal}"\n  sorry\n'
    return [Skeleton(text=fallback, holes=find_sorry_spans(fallback))]


def _lib_templates_for_goal(goal: str) -> List[Skeleton]:
    toks = _tokenize_goal(goal)
    lib: List[str] = []
    if ("@" in toks or "map" in toks) and "xs" in toks:
        lib.append(
f'''lemma "{goal}"
proof (induction xs)
  case Nil
  show ?case by simp
next
  case (Cons x xs)
  show ?case
    sorry
qed
''')
    if ({"Suc","0"} & toks) and "n" in toks:
        lib.append(
f'''lemma "{goal}"
proof (induction n)
  case 0
  show ?case by simp
next
  case (Suc n)
  show ?case
    sorry
qed
''')
    if ({"True","False"} & toks) and "b" in toks:
        lib.append(
f'''lemma "{goal}"
proof (cases b)
  case True
  show ?thesis
    sorry
next
  case False
  show ?thesis
    sorry
qed
''')
    lib.append(
f'''lemma "{goal}"
proof -
  have f1: "(* fill a useful intermediate statement *)"
    sorry
  have f2: "(* another useful intermediate *)"
    using f1
    sorry
  show ?thesis
    using f1 f2
    sorry
qed
''')
    return [Skeleton(text=s if s.endswith("\n") else s+"\n", holes=find_sorry_spans(s)) for s in lib]

def propose_isar_skeleton_diverse_best(
    goal: str,
    *,
    isabelle,             # required for sketch check
    session_id: str,
    model: Optional[str] = None,
    temps: Iterable[float] = (0.35, 0.55, 0.85),
    k: int = 3,
    force_outline: bool = False,
    # knobs
    priors_path: Optional[str] = None,
    context_hints: bool = False,
    lib_templates: bool = False,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.2,
    # NEW: hint lexicon
    hintlex_path: Optional[str] = None,
    hintlex_top: int = 8,
    timeout_s: Optional[int] = None,    # Fix 2
    trace: bool = False,
) -> Tuple[Skeleton, Dict[str, Any]]:
    """
    Generate K outlines, optionally inject context & hintlex hints, run one-shot sketch checks,
    and return the best using composite score:
      score = alpha * subgoals + beta * pattern_penalty - gamma * hint_bonus
    """
    deadline = (time.monotonic() + float(timeout_s)) if timeout_s is not None else None

    # Optional context hints from Isabelle state + hint lexicon
    rec_hints: List[str] = []
    if context_hints:
        state_block = _state_block_for_goal(isabelle, session_id, goal)
        rec_hints += _facts_from_state(state_block)[:8]
    hintlex = _load_hintlex(hintlex_path)
    if hintlex:
        rec_hints += _hints_from_hintlex(goal, hintlex, top=hintlex_top)
    rec_hints = list(dict.fromkeys(rec_hints))[:12]  # stable de-dup + cap

    # Outline candidates (LLM) + optional library templates
    llm_timeout_s = timeout_s
    if deadline is not None:
        remaining = deadline - time.monotonic()
        llm_timeout_s = max(1, int(remaining)) if remaining > 1.0 else 0

    if llm_timeout_s == 0:
        fallback = f'lemma "{goal}"\n  sorry\n'
        cands = [Skeleton(text=fallback, holes=find_sorry_spans(fallback))]
    else:
        cands = propose_isar_skeletons(goal, model=model, temps=temps, k=k,
                                       force_outline=force_outline, hints=rec_hints,
                                       timeout_s=llm_timeout_s, trace=trace)    # Fix 2
    if lib_templates:
        cands = _lib_templates_for_goal(goal) + cands

    # Load optional priors/rules
    rules = _load_priors(priors_path)

    scored: List[Tuple[float, int, int]] = []  # (score, n_subgoals, idx)
    for i, sk in enumerate(cands):
        if deadline is not None:
            remaining = deadline - time.monotonic()
            candidates_left = max(1, len(cands) - i)
            score_timeout = max(1, int(remaining / candidates_left)) if remaining > 1.0 else 0
        else:
            score_timeout = 5

        if trace:
            if score_timeout > 0:
                print(
                    f"[Skeleton] scoring candidate {i + 1}/{len(cands)} "
                    f"({len(sk.text)} chars, {len(sk.holes)} sorry holes, timeout={score_timeout}s)...",
                    flush=True,
                )
            else:
                print(
                    f"[Skeleton] scoring candidate {i + 1}/{len(cands)} skipped: no skeleton budget left",
                    flush=True,
                )

        if score_timeout > 0:
            n = (_quick_sketch_score(isabelle, session_id, sk.text, timeout_s=score_timeout, trace=trace))
            if trace:
                print(f"[Skeleton] _quick_sketch_score returned: {n}")
        else:
            n = (9999)
            if trace:
                print(f"[Skeleton] skipped _quick_sketch_score because score_timeout>0; using default score")

        if trace:
            print(f"[Skeleton] scored candidate {i + 1}/{len(cands)}: subgoals={n}", flush=True)
        sorry_count = len(sk.holes)    # Fix: added sorry-count tiebreaker in scoring
        pat_pen = _pattern_penalty(goal, sk.text, rules)
        hint_b = _hint_bonus_from_outline(sk.text, rec_hints)
        score = alpha * float(n) + beta * float(pat_pen) - gamma * float(hint_b)
        scored.append((score, sorry_count, n, i))    # Fix: added sorry_count

    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]))    # Fix: added sorting by sorry_count
    best = cands[scored[0][3]]    # Fix: changed index from [2] to [3]
    if trace:
        print(f"[Skeleton] selected candidate {scored[0][3] + 1}/{len(cands)}", flush=True)
    diag = {
        "scores": scored,
        "num_candidates": len(cands),
        "used_hints": rec_hints[:12],
        "priors_rules": len(rules),
        "alpha_beta_gamma": (alpha, beta, gamma),
    }
    return best, diag
