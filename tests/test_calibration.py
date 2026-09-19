import json

import pytest
import torch

from jqv.calibration import CalibrationMismatch, TemperatureScaler
from jqv.prompt import PromptStyle, prompt_hash


def test_save_load_roundtrip_with_provenance(tmp_path):
    ts = TemperatureScaler(3.5)
    ts.save(tmp_path / "t.json", model="Qwen/Qwen3-1.7B", engine="packed", prompt_hash="abc", dataset="mmlu",
            n_val=400, choice_counts=[4])
    d = json.loads((tmp_path / "t.json").read_text())
    assert d["temperature"] == 3.5 and d["model"] == "Qwen/Qwen3-1.7B" and d["dataset"] == "mmlu"
    assert d["fitted_at"]
    back = TemperatureScaler.load(tmp_path / "t.json")
    assert back.temperature == 3.5 and back.meta["n_val"] == 400 and back.info()["dataset"] == "mmlu"


def test_legacy_file_loads_but_is_not_verified(tmp_path):
    (tmp_path / "old.json").write_text(json.dumps({"temperature": 2.0}))
    ts = TemperatureScaler.load(tmp_path / "old.json")
    assert ts.temperature == 2.0 and ts.meta == {}
    with pytest.raises(CalibrationMismatch):
        ts.check_compatible("Qwen/Qwen3-1.7B", "abc", allow_mismatch=False)
    with pytest.warns(UserWarning):
        assert ts.check_compatible("Qwen/Qwen3-1.7B", "abc", allow_mismatch=True)


def test_model_or_prompt_mismatch_is_rejected():
    ts = TemperatureScaler(2.0, {"model": "Qwen/Qwen3-1.7B", "prompt_hash": "abc"})
    assert ts.check_compatible("Qwen/Qwen3-1.7B", "abc", allow_mismatch=False) == []
    with pytest.raises(CalibrationMismatch, match="model"):
        ts.check_compatible("Qwen/Qwen3-4B", "abc", allow_mismatch=False)
    with pytest.raises(CalibrationMismatch, match="prompt_hash"):
        ts.check_compatible("Qwen/Qwen3-1.7B", "zzz", allow_mismatch=False)


def test_prompt_hash_changes_with_style():
    assert prompt_hash(PromptStyle()) == prompt_hash(PromptStyle())
    assert prompt_hash(PromptStyle()) != prompt_hash(PromptStyle(chat=False))
    assert prompt_hash(PromptStyle()) != prompt_hash(PromptStyle(system="other"))


def test_fit_recovers_scale():
    torch.manual_seed(0)
    z = torch.randn(2000, 4) * 3
    y = z.argmax(-1)
    # labels drawn from softmax(z / 4): the fitted T should be near 4
    y = torch.distributions.Categorical(logits=z / 4).sample()
    ts = TemperatureScaler().fit(z, y)
    assert 3.0 < ts.temperature < 5.5
