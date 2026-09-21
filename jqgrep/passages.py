"""Stage 2 helpers: split a file into overlapping line ranges, build the shared-state questions, merge ranges."""

from __future__ import annotations

from dataclasses import dataclass

ROLE_LABELS = ["primary_implementation", "call_site_or_usage", "test_or_example", "config_or_docs"]
ROLE_TEXT = {
    "primary_implementation": "the primary implementation of what the query asks for",
    "call_site_or_usage": "a place that only calls or uses that functionality",
    "test_or_example": "a test, fixture or example",
    "config_or_docs": "configuration, data or documentation",
}


@dataclass
class Range:
    start: int  # 1-based inclusive
    end: int    # inclusive


def line_ranges(n_lines: int, size: int = 40, stride: int = 30, max_ranges: int = 32) -> list[Range]:
    """Overlapping windows of `size` lines every `stride` lines. If that would exceed max_ranges, the window grows
    so that the whole file is still covered by at most max_ranges windows."""
    if n_lines <= 0:
        return []
    if n_lines <= size:
        return [Range(1, n_lines)]
    n_windows = (n_lines - size + stride - 1) // stride + 1
    if n_windows > max_ranges:
        stride = max(1, (n_lines - size + max_ranges - 1) // (max_ranges - 1))
        size = max(size, stride + stride // 3)
        n_windows = (n_lines - size + stride - 1) // stride + 1
    out = []
    start = 1
    while start <= n_lines and len(out) < max_ranges:
        end = min(n_lines, start + size - 1)
        out.append(Range(start, end))
        if end >= n_lines:
            break
        start += stride
    if out and out[-1].end < n_lines:
        out[-1] = Range(out[-1].start, n_lines)
    return out


def windows(n_lines: int, max_lines: int, overlap: int = 40) -> list[Range]:
    """Split a long file into windows that fit the state budget (in lines); windows overlap slightly."""
    if n_lines <= max_lines:
        return [Range(1, n_lines)]
    out, start = [], 1
    while start <= n_lines:
        end = min(n_lines, start + max_lines - 1)
        out.append(Range(start, end))
        if end >= n_lines:
            break
        start = end - overlap + 1
    return out


def numbered(lines: list[str], first: int = 1) -> str:
    width = len(str(first + len(lines) - 1))
    return "\n".join(f"{i + first:>{width}} | {ln}" for i, ln in enumerate(lines))


def build_state(query: str, rel: str, language: str, lines: list[str], first_line: int) -> str:
    return (f"Search query: {query}\n\n"
            f"File: {rel} ({language}); lines {first_line}-{first_line + len(lines) - 1} are shown with line numbers.\n\n"
            f"{numbered(lines, first_line)}\n")


def range_question(r: Range) -> str:
    return f"Do lines {r.start}-{r.end} directly contain what the search query asks for (the relevant definition, logic or handling, not just a mention)?"


def extra_questions() -> list[tuple[str, str, list[str]]]:
    """(key, question, choices) asked once per file window alongside the range questions."""
    return [
        ("file_relevant", "Considering the whole file shown, is this file relevant to the search query?", ["yes", "no"]),
        ("role", "What role does this file play with respect to the search query?", [ROLE_TEXT[k] for k in ROLE_LABELS]),
        ("production", "Is this production code (as opposed to tests, examples or scratch files)?", ["yes", "no"]),
        ("ambiguous", "Is the query ambiguous with respect to this file, so that a human should double-check the match?", ["yes", "no"]),
    ]


def merge_ranges(scored: list[tuple[Range, float]], min_p: float, gap_lines: int = 0) -> list[tuple[Range, float]]:
    """Keep ranges with p >= min_p and merge overlapping / adjacent ones; the merged score is the maximum."""
    kept = sorted([(r, p) for r, p in scored if p >= min_p], key=lambda x: x[0].start)
    out: list[tuple[Range, float]] = []
    for r, p in kept:
        if out and r.start <= out[-1][0].end + 1 + gap_lines:
            pr, pp = out[-1]
            out[-1] = (Range(pr.start, max(pr.end, r.end)), max(pp, p))
        else:
            out.append((Range(r.start, r.end), p))
    return out


def trim_range(lines: list[str], r: Range, max_lines: int = 60) -> Range:
    """Shrink a merged range to its non-blank core, capped at max_lines around the middle."""
    start, end = r.start, r.end
    while start < end and not lines[start - 1].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    if end - start + 1 > max_lines:
        mid = (start + end) // 2
        start, end = max(r.start, mid - max_lines // 2), min(r.end, mid + max_lines // 2)
    return Range(start, end)
