"""Text and JSON rendering of search results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Hit:
    path: str
    start: int
    end: int
    score: float            # range relevance probability
    file_score: float       # file-level relevance probability (stage 2)
    role: str
    role_p: float
    production: float
    ambiguous: float
    snippet: list[str] = field(default_factory=list)
    stage1: float | None = None


def render_text(hits: list[Hit], query: str, stats: dict, snippet_lines: int = 3) -> str:
    out = [f"query: {query}"]
    out.append(f"files walked {stats.get('walked', 0)} -> lexical {stats.get('stage0', 0)} -> jqv routed {stats.get('stage1', 0)} "
               f"-> verified {stats.get('stage2_files', 0)}; ranges scored {stats.get('ranges', 0)}; "
               f"model {stats.get('model', '?')} engine {stats.get('engine', '?')}; {stats.get('seconds', 0):.1f} s")
    out.append("")
    if not hits:
        out.append("(no ranges above the relative threshold)")
    for h in hits:
        flags = [h.role.replace("_", " ")]
        if h.ambiguous >= 0.5:
            flags.append("ambiguous")
        if h.production < 0.5:
            flags.append("non-production")
        out.append(f"{h.score:.2f}  {h.path}:{h.start}-{h.end}   [{', '.join(flags)}; file {h.file_score:.2f}]")
        for ln in h.snippet[:snippet_lines]:
            out.append(f"      {ln[:140]}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_json(hits: list[Hit], query: str, stats: dict) -> str:
    return json.dumps({"query": query, "stats": stats, "hits": [asdict(h) for h in hits]}, ensure_ascii=False, indent=2)
