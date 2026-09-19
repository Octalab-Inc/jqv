import torch

from jqv.heads import PointerHead, SlotHead, load_head, save_head


def test_pointer_head_is_permutation_equivariant():
    torch.manual_seed(0)
    head = PointerHead(hidden=32, rank=8)
    h_d = torch.randn(2, 32)
    h_opts = torch.randn(2, 4, 32)
    mask = torch.ones(2, 4, dtype=torch.bool)
    z = head(h_d, h_opts, mask)
    perm = torch.tensor([2, 0, 3, 1])
    z_perm = head(h_d, h_opts[:, perm], mask)
    assert torch.allclose(z[:, perm], z_perm, atol=1e-6)


def test_slot_head_masks_unused_slots_and_roundtrips(tmp_path):
    head = SlotHead(hidden=16, max_choices=26)
    z = head(torch.randn(3, 16), torch.zeros(3, 3, 16), torch.tensor([[True, True, False]] * 3))
    assert z.shape == (3, 3) and torch.isinf(z[:, 2]).all()
    save_head(head, tmp_path / "h", {"step": 1})
    back, cfg = load_head(tmp_path / "h", "cpu")
    assert cfg["kind"] == "slot" and cfg["step"] == 1
    assert torch.equal(back.proj.weight, head.proj.weight)
