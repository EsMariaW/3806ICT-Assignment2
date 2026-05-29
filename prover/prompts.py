# prover/prompts.py (optimized, compatible)
import re
from typing import List, Sequence

SYSTEM_STEPS = """You are an Isabelle/HOL proof expert generating intermediate tactics.

You will receive the lemma goal, the proof lines accepted so far, and the latest subgoal state. Your job is to propose 3–8 `apply`-style tactics that simplify or split the current subgoal — not finish it. Finishers go in a separate stage.

<output_rules>
- Output ONLY one tactic per line. No prose, no fences, no numbering, no bullets.
- Each line must start with `apply ` or `apply(`.
- Each tactic should plausibly REDUCE the subgoal count or simplify its shape. Do not include finishers (`by ...`, `done`).
- Use ONLY lemma names that appear in the "Helpful facts" section or in the accepted steps. Do NOT invent lemma names — a hallucinated `*_def` or library lemma fails the tactic.
- If you cannot think of a useful intermediate step, return fewer suggestions. Three good ones beat eight bad ones.
</output_rules>

<tactic_families>
Simplification: `apply simp`, `apply auto`, `apply clarsimp`
With hints: `apply (simp add: <facts>)`, `apply (simp only: <facts>)`
Case splits: `apply (cases xs)`, `apply (cases x rule: list.exhaust)`
Induction: `apply (induction xs)`, `apply (induction xs arbitrary: ys)`
Rewriting: `apply (subst <thm>)`, `apply (unfold <facts>)`
Logical structure: `apply (rule conjI)`, `apply (rule impI)`, `apply (intro impI)`, `apply (erule disjE)`
Splitters: `apply (simp split: option.splits if_splits)`
</tactic_families>

<examples>
For a goal with a conjunction in the conclusion:
apply (rule conjI)
apply simp
apply auto

For a goal with a case split on a list:
apply (cases xs)
apply simp
apply (cases xs rule: list.exhaust)

For an induction goal that hasn't started induction yet:
apply (induction xs)
apply simp
apply (induction xs arbitrary: ys)
</examples>
"""

SYSTEM_FINISH = """You are an Isabelle/HOL proof expert generating finisher tactics.

You will receive the lemma goal, the proof lines accepted so far, and the latest subgoal state. Your job is to propose 3–8 short tactics that could close the remaining subgoal in one step.

<output_rules>
- Output ONLY one tactic per line. No prose, no fences, no numbering, no bullets.
- Each line must start with `by ` or be the bare word `done`.
- Use `done` ONLY when the latest state shows no subgoals remaining.
- Order matters: put the cheapest tactic first. Verifier tries them in order.
- Use ONLY lemma names that appear in the "Helpful facts" section or in the accepted steps. Do NOT invent lemma names — a wrong fact reference fails the whole tactic.
- An omitted suggestion is better than a confidently wrong one. Returning 3 good tactics beats 8 with hallucinated facts.
- metis is expensive. Use it only when simpler tactics clearly fail. Cap metis calls at 2 facts unless absolutely necessary.
</output_rules>

<tactic_families>
Cheap first-pass: `done`, `by simp`, `by auto`, `by blast`
Equational with hints: `by (simp add: <facts>)`, `by (simp only: <facts>)`
Heavier search: `by fastforce`, `by force`, `by meson`
Targeted: `by (metis <facts>)`, `by (rule <thm>)`
Arithmetic: `by arith`, `by linarith`, `by presburger`
Case-split finishers: `by (cases xs, simp_all)`, `by (induct xs) auto`
Induction cases: `by (cases ...)`, `using Cons.IH by simp`, `using Cons.IH by auto`, `using Cons.IH Cons.prems by simp`
</tactic_families>

<examples>
For a typical equational goal:
by simp
by auto
by (simp add: append_assoc)

For a goal involving arithmetic:
by simp
by arith
by linarith

For a goal needing case analysis:
by (cases xs, simp_all)
by (induct xs) auto

For a goal with no subgoals left:
done
</examples>

For a goal inside an induction case named Cons:
using Cons.IH by simp
using Cons.IH by auto
by (cases xs)
"""

USER_TEMPLATE = """<goal>
{goal}
</goal>

<accepted_steps>
{steps}
</accepted_steps>

<latest_state>
{state_hint}
</latest_state>

<helpful_facts>
{facts}
</helpful_facts>

Output 3–8 candidate tactics, one per line.
"""

# Precompiled once (same patterns as before)
_LINE_RE   = re.compile(r"^\s*(?:[-*]\s*)?([a-zA-Z].*?)\s*$")
_FENCE_RE  = re.compile(r"```.*?```", re.DOTALL | re.MULTILINE)
_OLENUM_RE = re.compile(r"^\d+\.\s*")
_WS_RE     = re.compile(r"\s+")

def parse_ollama_lines(text: str, allowed_prefixes: Sequence[str], max_items: int) -> List[str]:
    """
    Extract LLM output lines that start with one of `allowed_prefixes`.
    - Dedents code blocks fenced by ```...```.
    - Strips list numbering like '1. ...' and bullets.
    - Collapses internal whitespace to single spaces.
    Behavior and return shape unchanged.
    """
    if text.startswith("__ERROR__"):
        return []
    if not text:
        return []

    text = _FENCE_RE.sub("", text)
    out: List[str] = []
    seen = set()
    prefixes = tuple(allowed_prefixes) if not isinstance(allowed_prefixes, tuple) else allowed_prefixes

    for ln in text.splitlines():
        m = _LINE_RE.match(ln)
        if not m:
            continue
        cand = _OLENUM_RE.sub("", m.group(1).strip())
        if not cand or len(cand) > 120 or "#" in cand:
            continue
        if not cand.startswith(prefixes):
            continue
        cand = _WS_RE.sub(" ", cand)
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
        if len(out) >= max_items:
            break
    return out
