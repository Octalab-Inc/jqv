"""jqgrep without a model: walker, lexical sketches, range arithmetic, merging, and the whole cascade with a fake client."""

from __future__ import annotations

import json
from pathlib import Path

from jqgrep.output import render_json, render_text
from jqgrep.passages import Range, line_ranges, merge_ranges, trim_range, windows
from jqgrep.search import SearchConfig, search
from jqgrep.sketch import query_terms, sketch_file, split_camel, stage0
from jqgrep.walker import guess_language, is_binary, walk


def make_repo(tmp_path: Path) -> Path:
    (tmp_path / "src" / "auth").mkdir(parents=True)
    (tmp_path / "src" / "auth" / "jwt.py").write_text(
        "import hmac\n\n\ndef verify_signature(token: str, key: bytes) -> bool:\n    header, payload, sig = token.split('.')\n"
        "    expected = hmac.new(key, f'{header}.{payload}'.encode(), 'sha256').hexdigest()\n    return hmac.compare_digest(expected, sig)\n\n\n"
        "def decode(token):\n    return token.split('.')\n" + "\n".join(f"# filler line {i}" for i in range(80)) + "\n")
    (tmp_path / "src" / "api.py").write_text("from src.auth.jwt import verify_signature\n\n\ndef handle(req):\n    if not verify_signature(req.token, KEY):\n        return 401\n    return 200\n")
    (tmp_path / "README.md").write_text("# demo\n\nThis project verifies JWT tokens.\n")
    (tmp_path / "big.bin").write_bytes(b"\x00\x01" * 100)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("module.exports = 1\n")
    return tmp_path


def test_walker_filters(tmp_path):
    root = make_repo(tmp_path)
    entries = walk(root)
    rels = {e.rel for e in entries}
    assert "src/auth/jwt.py" in rels and "README.md" in rels
    assert "big.bin" not in rels and "node_modules/x.js" not in rels
    assert is_binary(root / "big.bin") and not is_binary(root / "README.md")
    assert guess_language(Path("a.rs")) == "rust" and guess_language(Path("Dockerfile")) == "docker"


def test_query_terms_and_lexical_sketch(tmp_path):
    assert "signature" in query_terms("where is the JWT signature verified") and "jwt" in query_terms("verify JWT tokens")
    assert split_camel("verifySignatureHMAC") == ["verify", "signature", "hmac"]
    root = make_repo(tmp_path)
    entries = walk(root)
    terms = query_terms("verify the JWT signature")
    jwt = next(e for e in entries if e.rel.endswith("jwt.py"))
    sk = sketch_file(jwt, terms)
    assert "verify_signature" in sk.symbols and sk.hits and sk.lexical > 0
    ranked = stage0(entries, "verify the JWT signature", keep=10)
    assert ranked[0].entry.rel == "src/auth/jwt.py"


def test_line_ranges_cover_and_cap():
    rs = line_ranges(100, size=40, stride=30, max_ranges=32)
    assert rs[0] == Range(1, 40) and rs[-1].end == 100 and all(r.end >= r.start for r in rs)
    rs = line_ranges(5000, size=40, stride=30, max_ranges=32)
    assert len(rs) <= 32 and rs[-1].end == 5000 and rs[0].start == 1
    assert line_ranges(10) == [Range(1, 10)] and line_ranges(0) == []
    ws = windows(1000, max_lines=400, overlap=40)
    assert ws[0] == Range(1, 400) and ws[-1].end == 1000 and ws[1].start == 361


def test_merge_and_trim():
    scored = [(Range(1, 40), 0.9), (Range(31, 70), 0.85), (Range(61, 100), 0.2), (Range(91, 130), 0.8)]
    merged = merge_ranges(scored, min_p=0.6)
    assert merged == [(Range(1, 70), 0.9), (Range(91, 130), 0.8)]
    lines = [""] * 3 + ["x"] * 10 + [""] * 3
    assert trim_range(lines, Range(1, 16)) == Range(4, 13)
    assert trim_range(["y"] * 200, Range(1, 200), max_lines=60).end - trim_range(["y"] * 200, Range(1, 200), max_lines=60).start <= 60


class FakeClient:
    """p(yes) is high when the question or state mentions the key phrase; deterministic, no model."""

    model_id, engine_name, temperature = "fake", "fake", None

    def n_tokens(self, text):
        return len(text) // 4

    def decide(self, state, questions):
        out = []
        for q, choices in questions:
            if len(choices) == 2:
                # stage 1: the sketch (in the question) shows the definition; stage 2: the state shows it and the range starts the file
                hot = "def verify_signature" in q or ("def verify_signature" in state and "lines" in q and "1-" in q)
                out.append([0.9, 0.1] if hot else [0.15, 0.85])
            else:
                out.append([0.7, 0.1, 0.1, 0.1])
        return out


def test_cascade_with_fake_client(tmp_path):
    root = make_repo(tmp_path)
    hits, stats = search("where is the JWT signature verified", str(root), SearchConfig(n0=10, n1=3, k=5, range_size=40, stride=30), FakeClient(), log=lambda m: None)
    assert hits and hits[0].path == "src/auth/jwt.py" and hits[0].start <= 4 <= hits[0].end
    assert hits[0].role == "primary_implementation" and stats["stage2_files"] >= 1
    txt = render_text(hits, "q", stats)
    assert "src/auth/jwt.py:" in txt
    js = json.loads(render_json(hits, "q", stats))
    assert js["hits"][0]["path"] == "src/auth/jwt.py"
