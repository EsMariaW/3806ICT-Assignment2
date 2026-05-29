#!/usr/bin/env python3
"""
Extract lemma/theorem statements from Isabelle/HOL theories, with robust theory
header parsing (theory name + imports). Outputs a JSONL with per-goal metadata
and a plain TXT list of goals.

Usage:
  python datasets/hol_extract_goals_v4.py \
    --isabelle-hol /Applications/Isabelle2025.app/src/HOL \
    --out datasets \
    --only List Set Nat Algebra Analysis Complex_Analysis Number_Theory Binomial

Outputs:
  datasets/hol_goals.jsonl
  datasets/hol_goals.txt
"""
from __future__ import annotations
import argparse, json, random, re
from pathlib import Path
from typing import Optional, Iterable

# Strip (* ... *) comments (non-nested; good enough for headers)
COMMENT_RE = re.compile(r"\(\*.*?\*\)", re.DOTALL)

# Lemma/theorem starts
DECL_RE = re.compile(r'(?m)^\s*(lemma|theorem|corollary)\b')
DECL_OR_PROOF_RE = re.compile(
    r'(?m)^\s*(lemma|theorem|corollary|proof|qed|sorry|by|using|unfolding|end)\b'
)
LOCAL_CONTEXT_RE = re.compile(r'^\s*locale\b')
BEGIN_RE = re.compile(r'^\s*begin\b')
END_RE = re.compile(r'^\s*end\b')

# Theory header bits
THEORY_RE = re.compile(r'(?m)^\s*theory\s+(\S+)\b')
IMPORTS_BLOCK_RE = re.compile(r'(?s)\bimports\b(.*?)\bbegin\b')

_MIN_LEN, _MAX_LEN = 5, 2000
QUOTE_OR_CARTOUCHE_RE = re.compile(
    r'"(?P<dq>(?:\\.|[^"\\])*)"|‹(?P<ct>.*?)›',
    re.DOTALL,
)
STRUCTURED_KEYWORD_RE = re.compile(
    r'\b(fixes|assumes|defines|notes|obtains|shows)\b',
    re.IGNORECASE,
)
ASSUMES_BLOCK_RE = re.compile(
    r'\bassumes\b(?P<body>.*?)(?=\b(defines|notes|obtains|shows|proof|qed|sorry|by|using|unfolding|end|lemma|theorem|corollary)\b|\Z)',
    re.IGNORECASE | re.DOTALL,
)
SHOWS_BLOCK_RE = re.compile(
    r'\bshows\b(?P<body>.*?)(?=\b(proof|qed|sorry|by|using|unfolding|end|lemma|theorem|corollary)\b|\Z)',
    re.IGNORECASE | re.DOTALL,
)

def _one_line_goal(s: str) -> str:
    """Keep Isabelle syntax raw, but force each extracted proposition onto one line."""
    return re.sub(r"\s+", " ", s).strip()

def _unescape_isabelle_string(s: str) -> str:
    """Decode only standard quoted-string escapes; leave Isabelle escapes intact."""
    return s.replace(r'\"', '"').replace(r'\\', '\\')

def _quoted_props(s: str) -> list[str]:
    props: list[str] = []
    for m in QUOTE_OR_CARTOUCHE_RE.finditer(s):
        raw = m.group("dq") if m.group("dq") is not None else m.group("ct")
        if raw is None:
            continue
        if m.group("dq") is not None:
            raw = _unescape_isabelle_string(raw)
        prop = _one_line_goal(raw)
        if prop:
            props.append(prop)
    return props

def _looks_like_type_not_prop(s: str) -> bool:
    """Filter `fixes x :: "type"` payloads accidentally seen as propositions."""
    t = s.strip()
    if not t:
        return True
    if t.startswith("'") and any(tok in t for tok in (r"\<Rightarrow>", "=>", "⇒")):
        return True
    if "::" in t and any(tok in t for tok in (r"\<Rightarrow>", "=>", "⇒")):
        return True
    return False

def _valid_goal_text(s: str) -> bool:
    return _MIN_LEN <= len(s) <= _MAX_LEN and not _looks_like_type_not_prop(s)

def _mk_goal(premises: list[str], conclusion: str) -> str:
    parts = [f"({p})" for p in premises if p] + [conclusion]
    return _one_line_goal(r" \<Longrightarrow> ".join(parts))

def _statement_blocks(text: str) -> Iterable[str]:
    hidden_spans = _local_context_spans(text)
    for m in DECL_RE.finditer(text):
        if any(start <= m.start() < end for start, end in hidden_spans):
            continue
        nxt = DECL_OR_PROOF_RE.search(text, m.end())
        end = nxt.start() if nxt else len(text)
        yield text[m.start():end]

def _local_context_spans(text: str) -> list[tuple[int, int]]:
    """Return spans for locale blocks whose lemmas need hidden fixed constants."""
    spans: list[tuple[int, int]] = []
    pending_start: Optional[int] = None
    depth = 0
    span_start: Optional[int] = None

    for m in re.finditer(r".*(?:\n|$)", text):
        line = m.group(0)
        if not line:
            continue
        line_start, line_end = m.start(), m.end()
        if depth == 0:
            if pending_start is None and LOCAL_CONTEXT_RE.match(line):
                pending_start = line_start
            if pending_start is not None and BEGIN_RE.match(line):
                span_start = pending_start
                pending_start = None
                depth = 1
                continue
            if pending_start is not None and DECL_RE.match(line):
                pending_start = None
            continue

        if LOCAL_CONTEXT_RE.match(line):
            pending_start = line_start
        if pending_start is not None and BEGIN_RE.match(line):
            pending_start = None
            depth += 1
            continue
        if END_RE.match(line):
            depth -= 1
            if depth == 0 and span_start is not None:
                spans.append((span_start, line_end))
                span_start = None
                pending_start = None
    return spans

def _extract_goals_from_block(block: str) -> list[str]:
    # Structured Isabelle statement:
    #
    #   lemma foo:
    #     fixes x :: "..."
    #     assumes a: "A"
    #     shows "B"
    #
    # The old extractor grabbed the fixes type. Here the goal is A ==> B.
    shows_m = SHOWS_BLOCK_RE.search(block)
    if shows_m:
        conclusions = [p for p in _quoted_props(shows_m.group("body")) if not _looks_like_type_not_prop(p)]
        if not conclusions:
            return []
        assumes_m = ASSUMES_BLOCK_RE.search(block)
        premises = []
        if assumes_m:
            premises = [p for p in _quoted_props(assumes_m.group("body")) if not _looks_like_type_not_prop(p)]
        goals = [_mk_goal(premises, c) for c in conclusions]
        return [g for g in goals if _valid_goal_text(g)]

    # Simple statement:
    #
    #   lemma foo: "P"
    #   lemma "P"
    #
    # If this block has structured keywords but no `shows`, do not fall back to
    # arbitrary quotes; those are commonly type annotations.
    after_decl = re.sub(r'^\s*(lemma|theorem|corollary)\b', '', block, count=1, flags=re.IGNORECASE).strip()
    if after_decl.startswith("(in "):
        return []
    if STRUCTURED_KEYWORD_RE.search(after_decl):
        return []

    props = [p for p in _quoted_props(after_decl) if not _looks_like_type_not_prop(p)]
    if re.search(r'\bif\b', after_decl) and len(props) >= 2:
        goal = _mk_goal(props[1:], props[0])
        return [goal] if _valid_goal_text(goal) else []
    return [p for p in props[:1] if _valid_goal_text(p)]

def _strip_comments(s: str) -> str:
    return COMMENT_RE.sub("", s)

def _split_import_tokens(raw: str) -> list[str]:
    """
    Handle imports like:
      imports Main "HOL-Library.Multiset" "~~/src/HOL/Number_Theory/Primes"
    We normalize by removing quotes and common path punctuation, then split.
    """
    s = raw
    s = s.replace('"', ' ')
    s = s.replace('(', ' ').replace(')', ' ').replace('+', ' ')
    s = s.replace('~', ' ').replace('/', ' ').replace('\\', ' ')
    s = re.sub(r"\s+", " ", s.strip())
    if not s: return []
    toks = s.split(" ")
    # Keep reasonable atoms (theory-ish names)
    cleaned = [t for t in toks if re.match(r"^[A-Za-z0-9_.'-]+$", t)]
    return cleaned

def _parse_header(text: str) -> tuple[Optional[str], list[str]]:
    head = _strip_comments(text[: min(len(text), 20000)])
    m_thy = THEORY_RE.search(head)
    theory = m_thy.group(1) if m_thy else None
    imports: list[str] = []
    m_imp = IMPORTS_BLOCK_RE.search(head)
    if m_imp:
        imports = _split_import_tokens(m_imp.group(1))
    return theory, imports

def _theory_graph(hol: Path) -> dict[str, list[str]]:
    graph: dict[str, list[str]] = {}
    for thy in hol.rglob("*.thy"):
        try:
            text = thy.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        theory, imports = _parse_header(text)
        if theory:
            graph[theory] = [Path(imp).name for imp in imports]
    return graph

def _reachable_theories(hol: Path, root_theory: str) -> set[str]:
    graph = _theory_graph(hol)
    seen: set[str] = set()

    def visit(thy: str) -> None:
        if thy in seen:
            return
        seen.add(thy)
        for imp in graph.get(thy, []):
            if imp in graph:
                visit(imp)

    visit(root_theory)
    return seen

def _want_dir(sub: str, only: Optional[set[str]]) -> bool:
    if not only: return True
    parts = Path(sub).parts
    return any(p in only for p in parts)

def _extract_file(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    theory, imports = _parse_header(text)
    body = _strip_comments(text)
    for block in _statement_blocks(body):
        for stmt in _extract_goals_from_block(block):
            yield {"goal": stmt, "file": str(path), "theory": theory, "imports": imports}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isabelle-hol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--reachable-from", default=None,
                    help="Only keep goals whose source theory is in the import closure of this theory, e.g. Main.")
    ap.add_argument("--max-goals", type=int, default=None,
                    help="Write at most this many goals after filtering.")
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle before applying --max-goals.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outfile", default=None,
                    help="Plain goal output path. Default: OUT/hol_goals.txt")
    ap.add_argument("--jsonl-out", default=None,
                    help="Metadata output path. Default: OUT/hol_goals.jsonl, or OUTFILE with .jsonl suffix.")
    args = ap.parse_args()

    hol = Path(args.isabelle_hol).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    only = set(args.only) if args.only else None
    reachable = _reachable_theories(hol, args.reachable_from) if args.reachable_from else None
    thy_files = [p for p in hol.rglob("*.thy") if _want_dir(str(p.parent.relative_to(hol)), only)]
    print(f"Scanning {len(thy_files)} .thy files under {hol}")

    rows = []
    for thy in thy_files:
        try:
            rows.extend(_extract_file(thy))
        except Exception as e:
            print(f"[warn] {thy}: {e}")

    seen, uniq = set(), []
    for r in rows:
        if reachable is not None and r.get("theory") not in reachable:
            continue
        key = (r["goal"], r["file"])
        if key not in seen:
            seen.add(key); uniq.append(r)

    if args.shuffle:
        random.Random(args.seed).shuffle(uniq)
    if args.max_goals is not None:
        uniq = uniq[: max(0, args.max_goals)]

    txt_path = Path(args.outfile) if args.outfile else out / "hol_goals.txt"
    jsonl_path = Path(args.jsonl_out) if args.jsonl_out else (
        txt_path.with_suffix(".jsonl") if args.outfile else out / "hol_goals.jsonl"
    )
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    jsonl_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in uniq) + ("\n" if uniq else ""),
        encoding="utf-8")
    txt_path.write_text(
        "\n".join(r["goal"] for r in uniq) + ("\n" if uniq else ""),
        encoding="utf-8")

    print(f"Wrote {len(uniq)} items → {jsonl_path}")
    print(f"Also wrote plain list → {txt_path}")

if __name__ == "__main__":
    main()
