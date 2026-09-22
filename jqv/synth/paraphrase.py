"""Optional surface diversification: rewrite narrative paragraphs with a local Qwen, keeping every fact verbatim.

    uv run python -m jqv.synth.paraphrase --family long_policy --split train --model Qwen/Qwen3-14B --max-minutes 40

Only prose paragraphs are candidates (no headings, no bullet lists, no numbered clauses). The model is asked to
rewrite the wording only; the rewrite is accepted only if every number, date, identifier and capitalised name of the
original still appears and the length is within 0.6-1.6x. Rejected rewrites keep the original text. The state is
rewritten in place and `meta.paraphrased` lists the paragraph indices that changed. Deterministic given --seed.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

NUM = re.compile(r"\d[\d,.:/%-]*\d|\d")
ID_PAT = re.compile(r"\b[A-Z]{2,5}-\d{2,}(?:-\d+)?\b")
CAP = re.compile(r"\b[A-Z][a-z]{2,}\b")
MONTH = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b")

PROMPT = ("Rewrite the following paragraph from a business record in different words and with a different sentence structure. "
          "Keep every number, date, time, amount, identifier and name exactly as written, keep every comparison (more than, at least, "
          "before, after, within) and every negation exactly as they are, keep the same facts and the same meaning, and add nothing. "
          "Output only the rewritten paragraph.\n\nParagraph:\n{p}")


FACT_PREFIXES = ("Claim ", "Item ", "Order ", "Lot ", "Employee", "Account ", "Case ", "Contract ", "Adjuster's narrative", "Engineer's report",
                 "Claim narrative", "Household:", "Distances ", "Supplier certificate", "A unit drawn", "The container holds", "Repair ",
                 "Non-refundable", "Current term", "Policy EB", "Policy TR", "Policy HP",
                 "File ", "Operating licence", "Site permit", "Vendor accreditation", "Software subscription", "Fleet policy")  # temporal_v2 fact paragraphs
COMPARATORS = ("more than", "at least", "less than", "fewer than", "no later than", "not more than", "not later than", "on or after", "on or before",
               "before", "after", "within", "outside", "exceed", "at or above", "at or below", "above", "below", "until", "since",
               "not", "no ", "never", "without", "unless", "except", "only")


def candidate_paragraphs(state: str) -> list[int]:
    """Only item-specific fact paragraphs (claim files, reports, records). Rules, definitions, clauses, endorsements,
    amendments, headings and the distractor notes are never rewritten: a paraphrase could shift a threshold or a
    conclusion there without dropping any number, which the fact check would not catch."""
    out = []
    for i, p in enumerate(state.split("\n")):
        t = p.strip()
        if len(t) < 120 or len(t) > 1400 or t.isupper() or t.startswith(("=", "-", "|", "[")):
            continue
        if "note (" in t[:40].lower() or "note," in t[:40].lower():
            continue
        if not t.startswith(FACT_PREFIXES):
            continue
        out.append(i)
    return out


def guard_counts(text: str) -> dict[str, int]:
    low = " " + text.lower() + " "
    return {c: low.count(" " + c if not c.endswith(" ") else " " + c) for c in COMPARATORS}


STOP = {"The", "This", "That", "These", "Those", "After", "Before", "When", "Where", "While", "From", "With", "Without", "During",
        "Under", "Over", "Within", "Because", "However", "Although", "Note", "Notes", "Please", "Also", "Then", "There", "Their",
        "They", "She", "His", "Her", "Its", "Our", "Your", "You", "And", "But", "For", "Not", "Nor", "Yet", "Any", "All", "Both",
        "Each", "Some", "Such", "Only", "Once", "Since", "Until", "Unless", "Whether", "Which", "What", "Who", "Whose", "Given"}


def names_of(text: str) -> set[str]:
    """Capitalised words that are not sentence-initial and not function words: names, places, organisations."""
    out = set()
    for m in CAP.finditer(text):
        w = m.group(0)
        if w in STOP:
            continue
        i = m.start()
        before = text[max(0, i - 2):i]
        if i == 0 or before.endswith(("\n", ". ", "! ", "? ", ": ", "; ", "( ", "[ ")) or (len(before) >= 1 and before[-1] in "(['\""):
            continue
        out.add(w)
    return out


def facts_of(text: str) -> set[str]:
    return set(NUM.findall(text)) | set(ID_PAT.findall(text)) | names_of(text) | set(MONTH.findall(text))


def accept(orig: str, new: str) -> bool:
    new = new.strip()
    if not new or "\n\n" in new or new.lower().startswith(("here is", "rewritten", "sure")):
        return False
    r = len(new) / max(1, len(orig))
    if r < 0.6 or r > 1.6:
        return False
    missing = facts_of(orig) - facts_of(new)
    if missing:
        return False
    return guard_counts(orig) == guard_counts(new)  # comparators and negations must survive one-for-one


def load_model(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
    return tok, model, device


def rewrite_batch(tok, model, device, paragraphs: list[str], max_new: int) -> list[str]:
    import torch

    msgs = [[{"role": "user", "content": PROMPT.format(p=p)}] for p in paragraphs]
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs]
    enc = tok(texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.9, top_p=0.95, pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    return [tok.decode(g, skip_special_tokens=True).strip() for g in gen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--root", default="data/synth")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--max-minutes", type=float, default=40.0)
    ap.add_argument("--per-item", type=int, default=1, help="paragraphs to rewrite per item")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=1.0, help="share of items to attempt (items are shuffled first)")
    a = ap.parse_args()

    path = Path(a.root) / a.family / f"{a.split}.jsonl"
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rng = random.Random(a.seed)
    order = list(range(len(recs)))
    rng.shuffle(order)
    order = order[: int(len(order) * a.fraction)]
    todo = []  # (rec_idx, para_idx, text)
    for i in order:
        if recs[i].get("meta", {}).get("paraphrased"):
            continue
        cands = candidate_paragraphs(recs[i]["state"])
        for pi in rng.sample(cands, min(a.per_item, len(cands))):
            todo.append((i, pi, recs[i]["state"].split("\n")[pi]))
    print(f"{a.family}/{a.split}: {len(recs)} items, {len(todo)} candidate paragraphs, model {a.model}, time box {a.max_minutes} min", flush=True)
    import torch

    torch.manual_seed(a.seed)
    tok, model, device = load_model(a.model)
    t0 = time.time()
    done = accepted = 0
    changed: dict[int, list[int]] = {}
    for b in range(0, len(todo), a.batch):
        if (time.time() - t0) / 60 > a.max_minutes:
            print(f"time box reached after {done} paragraphs", flush=True)
            break
        batch = todo[b:b + a.batch]
        outs = rewrite_batch(tok, model, device, [t for _, _, t in batch], a.max_new)
        for (ri, pi, orig), new in zip(batch, outs):
            done += 1
            if accept(orig, new):
                lines = recs[ri]["state"].split("\n")
                lines[pi] = new
                recs[ri]["state"] = "\n".join(lines)
                changed.setdefault(ri, []).append(pi)
                accepted += 1
        el = time.time() - t0
        rate = done / el
        eta = (len(todo) - done) / rate / 60 if rate else float("nan")
        print(f"  {done}/{len(todo)} paragraphs, accepted {accepted} ({accepted / done:.0%}), {rate:.2f}/s, elapsed {el / 60:.1f} min, "
              f"ETA {min(eta, a.max_minutes - el / 60):.1f} min", flush=True)
    patch_path = path.with_suffix(".paraphrase.jsonl")
    with patch_path.open("a") as pf:
        for ri, pis in changed.items():
            recs[ri].setdefault("meta", {})["paraphrased"] = sorted(set(recs[ri]["meta"].get("paraphrased", []) + pis))
            recs[ri]["meta"]["paraphrase_model"] = a.model
            lines = recs[ri]["state"].split("\n")
            pf.write(json.dumps({"id": recs[ri]["id"], "paragraphs": {str(pi): lines[pi] for pi in pis}, "model": a.model}, ensure_ascii=False) + "\n")
    with path.open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {path}: {len(changed)} items changed ({accepted} paragraphs) in {(time.time() - t0) / 60:.1f} min; patch appended to {patch_path}", flush=True)


def apply_patch(split_path: Path) -> int:
    """Re-apply a saved paraphrase patch (train.paraphrase.jsonl) to a regenerated split. Returns the number of items changed."""
    patch_path = split_path.with_suffix(".paraphrase.jsonl")
    if not patch_path.exists():
        return 0
    patches: dict[str, dict] = {}
    for line in patch_path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            patches.setdefault(d["id"], {"paragraphs": {}, "model": d["model"]})["paragraphs"].update(d["paragraphs"])
    recs = [json.loads(l) for l in split_path.read_text().splitlines() if l.strip()]
    n = 0
    for r in recs:
        pt = patches.get(r["id"])
        if not pt:
            continue
        lines = r["state"].split("\n")
        ok = True
        for k, text in pt["paragraphs"].items():
            if int(k) < len(lines):
                lines[int(k)] = text
            else:
                ok = False
        if ok:
            r["state"] = "\n".join(lines)
            r.setdefault("meta", {})["paraphrased"] = sorted(int(k) for k in pt["paragraphs"])
            r["meta"]["paraphrase_model"] = pt["model"]
            n += 1
    with split_path.open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return n


if __name__ == "__main__":
    main()
