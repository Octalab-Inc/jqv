"""Stage 0a: enumerate candidate files. Uses `git ls-files` when the root is inside a git repository (so .gitignore is
respected exactly), otherwise a filtered os.walk. Binary and oversized files are skipped."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", "dist",
             "build", "target", ".idea", ".vscode", ".tox", ".cache", ".next", "coverage"}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".pdf", ".zip", ".gz", ".tar", ".bz2", ".xz", ".7z", ".npz",
                 ".npy", ".pt", ".pth", ".safetensors", ".bin", ".so", ".dylib", ".dll", ".exe", ".o", ".a", ".class", ".jar",
                 ".woff", ".woff2", ".ttf", ".mp3", ".mp4", ".mov", ".lock", ".parquet", ".arrow", ".pkl", ".pickle"}
LANG_BY_SUFFIX = {
    ".py": "python", ".rs": "rust", ".go": "go", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".kt": "kotlin", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp", ".cs": "csharp", ".rb": "ruby",
    ".php": "php", ".swift": "swift", ".scala": "scala", ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".sql": "sql",
    ".md": "markdown", ".rst": "text", ".txt": "text", ".toml": "toml", ".yaml": "yaml", ".yml": "yaml", ".json": "json",
    ".jsonl": "jsonl", ".html": "html", ".css": "css", ".ini": "ini", ".cfg": "ini", ".dockerfile": "docker", ".m": "objective-c",
    ".ex": "elixir", ".exs": "elixir", ".erl": "erlang", ".hs": "haskell", ".lua": "lua", ".r": "r", ".jl": "julia", ".dart": "dart",
}


@dataclass
class FileEntry:
    path: Path      # absolute
    rel: str        # relative to the search root, forward slashes
    size: int
    language: str


def guess_language(path: Path) -> str:
    if path.name.lower() == "dockerfile":
        return "docker"
    if path.name.lower() == "makefile":
        return "make"
    return LANG_BY_SUFFIX.get(path.suffix.lower(), "text" if path.suffix == "" else path.suffix.lstrip(".").lower())


def is_binary(path: Path, probe: int = 8192) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(probe)
    except OSError:
        return True
    if b"\x00" in chunk:
        return True
    if not chunk:
        return False
    # mostly non-text bytes -> binary
    text_like = sum(1 for b in chunk if 9 <= b <= 13 or 32 <= b < 127 or b >= 128)
    return text_like / len(chunk) < 0.85


def _git_files(root: Path) -> list[Path] | None:
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                             capture_output=True, check=True, timeout=30).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return [root / p.decode("utf-8", "replace") for p in out.split(b"\0") if p]


def walk(root: str | Path, max_bytes: int = 1_000_000, max_files: int = 20000) -> list[FileEntry]:
    root = Path(root).resolve()
    if root.is_file():
        paths = [root]
        base = root.parent
    else:
        base = root
        paths = _git_files(root)
        if paths is None:
            paths = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
                for fn in filenames:
                    paths.append(Path(dirpath) / fn)
    out = []
    for p in paths:
        if not p.is_file() or p.suffix.lower() in SKIP_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(base).parts[:-1]):
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size == 0 or size > max_bytes or is_binary(p):
            continue
        out.append(FileEntry(path=p, rel=p.relative_to(base).as_posix(), size=size, language=guess_language(p)))
        if len(out) >= max_files:
            break
    return out


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")
