"""Stage 0b: a cheap lexical score and a compact sketch (path, language, top-level symbols, head, query hits) per file.
No model involved; this is what routes 100-200 candidates into the semantic stages."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from jqgrep.walker import FileEntry, read_text

TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]+|[0-9]{2,}|[぀-ヿ一-鿿]{2,}")
SYMBOL_PATTERNS = [
    re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)", re.M),                       # python
    re.compile(r"^\s*class\s+([A-Za-z_]\w*)", re.M),                                   # python / js / java
    re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)", re.M),  # rust
    re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|impl(?:<[^>]*>)?)\s+([A-Za-z_]\w*)", re.M),
    re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)", re.M),                   # go
    re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", re.M),  # js / ts
    re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", re.M),
    re.compile(r"^\s*(?:export\s+)?(?:interface|type|enum)\s+([A-Za-z_$][\w$]*)", re.M),
    re.compile(r"^\s*(?:public|private|protected|static|\s)*[\w<>\[\],\s]+\s+([A-Za-z_]\w*)\s*\([^;{]*\)\s*(?:throws[^{]*)?\{", re.M),  # java-ish
    re.compile(r"^\s*(?:#+\s+)(.+)$", re.M),                                            # markdown headings
]
IMPORT_PATTERN = re.compile(r"^\s*(?:import\s+[\w.]+|from\s+[\w.]+\s+import|use\s+[\w:]+|#include\s*[<\"][^>\"]+|require\(|import\s*\{)", re.M)
STOP = {"the", "a", "an", "of", "in", "on", "for", "to", "is", "are", "where", "which", "what", "how", "does", "do", "and", "or",
        "with", "that", "this", "it", "be", "by", "from", "as", "at", "into", "code", "file", "files", "function", "place", "places"}


def split_camel(tok: str) -> list[str]:
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", tok)
    return [p.lower() for p in parts if len(p) > 1]


def query_terms(query: str) -> list[str]:
    terms = []
    for t in TOKEN.findall(query):
        low = t.lower()
        if low in STOP or len(low) < 2:
            continue
        terms.append(low)
        for sub in split_camel(t):
            if sub != low and sub not in STOP and len(sub) > 2:
                terms.append(sub)
    # crude stemming: drop plural / gerund suffixes so "tokens" matches "token", "verifying" matches "verify"
    out = []
    for t in terms:
        out.append(t)
        for suf in ("ing", "es", "s", "ed", "ion"):
            if t.endswith(suf) and len(t) - len(suf) >= 4:
                out.append(t[: -len(suf)])
                break
    return list(dict.fromkeys(out))


@dataclass
class Sketch:
    entry: FileEntry
    lexical: float
    symbols: list[str]
    imports: list[str]
    head: str
    hits: list[tuple[int, str]]  # (1-based line, text)
    n_lines: int
    text: str = field(repr=False)

    def prose(self, head_chars: int = 800, max_hits: int = 4) -> str:
        parts = [f"Path: {self.entry.rel}", f"Language: {self.entry.language} ({self.n_lines} lines)"]
        if self.symbols:
            parts.append("Top-level symbols: " + ", ".join(self.symbols[:25]))
        if self.imports:
            parts.append("Imports: " + "; ".join(self.imports[:8]))
        parts.append("Head:\n" + self.head[:head_chars].rstrip())
        if self.hits:
            parts.append("Lines matching query words:\n" + "\n".join(f"  {ln}: {tx.strip()[:160]}" for ln, tx in self.hits[:max_hits]))
        return "\n".join(parts)


CODE_LANGS = {"python", "rust", "go", "javascript", "typescript", "java", "kotlin", "c", "cpp", "csharp", "ruby", "php", "swift", "scala",
              "shell", "sql", "elixir", "erlang", "haskell", "lua", "r", "julia", "dart", "objective-c"}
DATA_LANGS = {"json", "jsonl", "yaml", "toml", "ini", "csv", "tsv"}
DATA_DIRS = ("results", "data", "logs", "log", "output", "outputs", "artifacts", "cache", "fixtures", "snapshots")
DOC_DIRS = ("docs", "doc", "tasks", "notes", "wiki", "examples", "samples")


def file_prior(entry: FileEntry, code_only: bool = True) -> float:
    """jqgrep is a code search: source files count fully, docs less, data/result files much less; directories that hold
    generated data or documentation are discounted. With code_only=False every text file counts fully."""
    if not code_only:
        return 1.0
    prior = 1.0 if entry.language in CODE_LANGS else 0.5 if entry.language in ("markdown", "text", "html", "css", "docker", "make") else 0.35 if entry.language in DATA_LANGS else 0.6
    parts = entry.rel.lower().split("/")[:-1]
    if any(part in DATA_DIRS for part in parts):
        prior *= 0.3
    if any(part in DOC_DIRS for part in parts):
        prior *= 0.6
    if entry.size > 200_000:
        prior *= 0.5
    return prior


def sketch_file(entry: FileEntry, terms: list[str], text: str | None = None, code_only: bool = True) -> Sketch:
    text = read_text(entry.path) if text is None else text
    lines = text.splitlines()
    low = text.lower()
    symbols: list[str] = []
    for pat in SYMBOL_PATTERNS:
        symbols += [m.strip() for m in pat.findall(text)]
    symbols = list(dict.fromkeys(s for s in symbols if s))[:60]
    imports = [m.group(0).strip() for m in IMPORT_PATTERN.finditer(text)][:12]
    hits = []
    if terms:
        for i, ln in enumerate(lines):
            l = ln.lower()
            if any(t in l for t in terms):
                hits.append((i + 1, ln))
                if len(hits) >= 12:
                    break
    # lexical score: term frequency (log-damped) with bonuses for path and symbol matches
    tf = Counter()
    for t in terms:
        c = low.count(t)
        if c:
            tf[t] = c
    score = sum(1 + math.log(c) for c in tf.values())
    rel_low = entry.rel.lower()
    sym_low = " ".join(symbols).lower()
    for t in terms:
        if t in rel_low:
            score += 2.0
        if t in sym_low:
            score += 1.5
    coverage = sum(1 for t in terms if t in low) / max(1, len(terms))
    score *= 0.5 + coverage  # reward files that contain many different query words
    score *= file_prior(entry, code_only)
    return Sketch(entry=entry, lexical=score, symbols=symbols, imports=imports, head=text[:1200], hits=hits, n_lines=len(lines), text=text)


def stage0(entries: list[FileEntry], query: str, keep: int, code_only: bool = True) -> list[Sketch]:
    terms = query_terms(query)
    sketches = [sketch_file(e, terms, code_only=code_only) for e in entries]
    sketches.sort(key=lambda s: (-s.lexical, s.entry.rel))
    # keep lexical hits first, then fill with small files so that a query with no lexical overlap still gets candidates
    keep_list = [s for s in sketches if s.lexical > 0][:keep]
    if len(keep_list) < keep:
        rest = [s for s in sketches if s.lexical <= 0]
        rest.sort(key=lambda s: s.entry.size)
        keep_list += rest[: keep - len(keep_list)]
    return keep_list
