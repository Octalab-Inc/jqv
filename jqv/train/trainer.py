"""Train a decision head (+ optional LoRA) on MMLU with CE (+ lambda * Brier). Progress is printed every
`log_every` steps with an ETA, checkpoints are written every `ckpt_every` steps and `resume=True` continues
from the last checkpoint. Everything lives under results/train/<run_name>/."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from jqv.heads import load_head, make_head, save_head
from jqv.model import Runtime
from jqv.train.data import Example, collate, gather_features, load_split, make_example


@dataclass
class TrainConfig:
    run_name: str
    head: str = "pointer"  # pointer | slot
    lora_rank: int = 16  # 0 = frozen LLM, head only
    lora_alpha: int = 32
    lora_targets: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    steps: int = 600
    batch_size: int = 8
    lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 0.0
    warmup: int = 50
    brier_weight: float = 0.0
    max_len: int = 512
    seed: int = 0
    log_every: int = 10
    eval_every: int = 100
    ckpt_every: int = 100
    n_val: int = 256
    pointer_rank: int = 256
    slot_init_from_lm_head: bool = True
    shuffle_options: bool = True


def brier_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = logits.softmax(-1)
    onehot = torch.zeros_like(p).scatter_(1, y[:, None], 1.0)
    p = torch.where(torch.isfinite(logits), p, torch.zeros_like(p))
    return ((p - onehot) ** 2).sum(-1).mean()


class Trainer:
    def __init__(self, rt: Runtime, cfg: TrainConfig, out_root: Path):
        self.rt, self.cfg = rt, cfg
        self.dir = out_root / cfg.run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        self.device = rt.device
        self.pad_id = rt.tokenizer.pad_token_id or 0
        self.model = rt.model
        for p in self.model.parameters():
            p.requires_grad_(False)
        if cfg.lora_rank > 0:
            from peft import LoraConfig, get_peft_model

            lcfg = LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
                              target_modules=cfg.lora_targets, bias="none", task_type="CAUSAL_LM")
            self.peft_model = get_peft_model(self.model, lcfg)
        else:
            self.peft_model = None
        hidden = self.model.config.hidden_size
        kw = {"rank": cfg.pointer_rank} if cfg.head == "pointer" else {}
        self.head = make_head(cfg.head, hidden, **kw).to(self.device)
        if cfg.head == "slot" and cfg.slot_init_from_lm_head:
            self.head.init_from_lm_head(rt.lm_head, rt.prompt.choice_token_ids(26))
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        groups = [{"params": list(self.head.parameters()), "lr": cfg.head_lr}]
        if lora_params:
            groups.append({"params": lora_params, "lr": cfg.lr})
        self.opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
        self.step = 0
        self.best_val = None
        self.log_path = self.dir / "train_log.jsonl"
        n_trainable = sum(p.numel() for g in groups for p in g["params"])
        print(f"[{cfg.run_name}] head={cfg.head} lora_rank={cfg.lora_rank} trainable params={n_trainable / 1e6:.2f}M "
              f"steps={cfg.steps} batch={cfg.batch_size} brier_weight={cfg.brier_weight}", flush=True)

    # ----- data -----
    def prepare_data(self):
        rng = random.Random(self.cfg.seed)
        train_items = load_split("auxiliary_train")
        rng.shuffle(train_items)
        val_items = load_split("validation")
        random.Random(self.cfg.seed + 1).shuffle(val_items)
        self.val = [e for e in (make_example(self.rt.prompt, it, None, self.cfg.max_len) for it in val_items[: self.cfg.n_val * 2])
                    if e is not None][: self.cfg.n_val]
        self.train_items = train_items
        self.rng = rng
        self._cursor = 0

    def next_batch(self) -> list[Example]:
        out = []
        while len(out) < self.cfg.batch_size:
            it = self.train_items[self._cursor % len(self.train_items)]
            self._cursor += 1
            e = make_example(self.rt.prompt, it, self.rng if self.cfg.shuffle_options else None, self.cfg.max_len)
            if e is not None:
                out.append(e)
        return out

    # ----- model -----
    def logits_for(self, batch: dict) -> torch.Tensor:
        h = self.rt.backbone(input_ids=batch["ids"], attention_mask=batch["attn"], use_cache=False).last_hidden_state
        h_d, h_opts = gather_features(h, batch)
        return self.head(h_d, h_opts, batch["opt_mask"])

    def lr_scale(self) -> float:
        if self.step < self.cfg.warmup:
            return (self.step + 1) / self.cfg.warmup
        progress = (self.step - self.cfg.warmup) / max(1, self.cfg.steps - self.cfg.warmup)
        return max(0.05, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.141592653589793)).item()))

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        self.head.eval()
        correct, nll, brier, n = 0, 0.0, 0.0, 0
        for i in range(0, len(self.val), 16):
            batch = collate(self.val[i : i + 16], self.pad_id, self.device)
            z = self.logits_for(batch).float()
            logp = z.log_softmax(-1)
            correct += (z.argmax(-1) == batch["y"]).sum().item()
            nll += -logp[torch.arange(z.shape[0]), batch["y"]].sum().item()
            brier += brier_loss(z, batch["y"]).item() * z.shape[0]
            n += z.shape[0]
        self.model.train()
        self.head.train()
        return {"val_acc": correct / n, "val_nll": nll / n, "val_brier": brier / n, "n": n}

    # ----- checkpoints -----
    def save(self, tag: str = "last") -> None:
        d = self.dir / tag
        d.mkdir(parents=True, exist_ok=True)
        save_head(self.head, d / "head", {"step": self.step, "run_name": self.cfg.run_name, "model": self.rt.model_id,
                                          "prompt_hash": self.rt.prompt.hash, "lora": self.cfg.lora_rank > 0})
        if self.peft_model is not None:
            self.peft_model.save_pretrained(str(d / "adapter"))
        if tag == "last":
            torch.save({"opt": self.opt.state_dict(), "step": self.step, "cursor": self._cursor,
                        "rng": self.rng.getstate(), "best_val": self.best_val}, d / "state.pt")

    def resume(self) -> bool:
        d = self.dir / "last"
        if not (d / "state.pt").exists():
            return False
        st = torch.load(d / "state.pt", map_location="cpu", weights_only=False)
        head, _ = load_head(d / "head", self.device)
        self.head.load_state_dict(head.state_dict())
        self.head.train()
        if self.peft_model is not None:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            set_peft_model_state_dict(self.peft_model, load_file(str(d / "adapter" / "adapter_model.safetensors")))
        self.opt.load_state_dict(st["opt"])
        self.step, self._cursor, self.best_val = st["step"], st["cursor"], st["best_val"]
        self.rng.setstate(st["rng"])
        print(f"[{self.cfg.run_name}] resumed from step {self.step}", flush=True)
        return True

    # ----- loop -----
    def train(self, resume: bool = False) -> dict:
        self.prepare_data()
        if resume:
            self.resume()
        self.model.train()
        self.head.train()
        t0, done0 = time.time(), self.step
        run_loss, run_n = 0.0, 0
        while self.step < self.cfg.steps:
            batch = collate(self.next_batch(), self.pad_id, self.device)
            scale = self.lr_scale()
            for g, base in zip(self.opt.param_groups, [self.cfg.head_lr, self.cfg.lr]):
                g["lr"] = base * scale
            z = self.logits_for(batch).float()
            loss = nn.functional.cross_entropy(z, batch["y"])
            if self.cfg.brier_weight:
                loss = loss + self.cfg.brier_weight * brier_loss(z, batch["y"])
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in self.opt.param_groups for p in g["params"]], 1.0)
            self.opt.step()
            self.step += 1
            run_loss += loss.item()
            run_n += 1
            if self.step % self.cfg.log_every == 0 or self.step == self.cfg.steps:
                elapsed = time.time() - t0
                rate = (self.step - done0) / elapsed
                eta = (self.cfg.steps - self.step) / rate if rate > 0 else float("nan")
                rec = {"step": self.step, "loss": run_loss / run_n, "lr_scale": scale, "elapsed_s": elapsed, "eta_s": eta}
                print(f"[{self.cfg.run_name}] step {self.step}/{self.cfg.steps} loss {rec['loss']:.4f} "
                      f"{rate:.2f} step/s elapsed {elapsed / 60:.1f} min ETA {eta / 60:.1f} min", flush=True)
                with self.log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                run_loss, run_n = 0.0, 0
            if self.step % self.cfg.eval_every == 0 or self.step == self.cfg.steps:
                ev = self.evaluate()
                print(f"[{self.cfg.run_name}] step {self.step} val_acc {ev['val_acc']:.3f} val_nll {ev['val_nll']:.3f} "
                      f"val_brier {ev['val_brier']:.3f} (n={ev['n']})", flush=True)
                with self.log_path.open("a") as f:
                    f.write(json.dumps({"step": self.step, **ev}) + "\n")
                if self.best_val is None or ev["val_nll"] < self.best_val:
                    self.best_val = ev["val_nll"]
                    self.save("best")
            if self.step % self.cfg.ckpt_every == 0 or self.step == self.cfg.steps:
                self.save("last")
        final = self.evaluate()
        (self.dir / "final.json").write_text(json.dumps({"step": self.step, **final}, indent=2))
        print(f"[{self.cfg.run_name}] done: {final}", flush=True)
        return final
