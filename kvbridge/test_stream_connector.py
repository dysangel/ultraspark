"""Unit tests for StreamConnector model-agnostic geometry (issue #23)."""

import pytest

from stream_connector import blob_from_dump, derive_layer_geometry


def names_4b():
    # qwen35-4b: 32 layers, attn every 4th starting at 3 (8 attn)
    attn = [f"model.layers.{i}.self_attn.attn" for i in range(3, 32, 4)]
    lin = [f"model.layers.{i}.linear_attn" for i in range(32)
           if i % 4 != 3]
    return attn + lin


def names_27b():
    # qwen38-27b: 64 layers, attn every 4th starting at 3 (16 attn)
    attn = [f"model.layers.{i}.self_attn.attn" for i in range(3, 64, 4)]
    lin = [f"model.layers.{i}.linear_attn" for i in range(64)
           if i % 4 != 3]
    return attn + lin


def test_geometry_4b():
    assert derive_layer_geometry(names_4b()) == (31, 32)


def test_geometry_27b():
    assert derive_layer_geometry(names_27b()) == (63, 64)


def test_geometry_no_attn_fails_loud():
    with pytest.raises(RuntimeError, match="could not derive"):
        derive_layer_geometry(["model.layers.0.linear_attn"])


def test_geometry_implausible_interval_fails_loud():
    with pytest.raises(RuntimeError, match="implausible"):
        # 64 layers but only 3 attn layers -> not a fixed interval
        derive_layer_geometry(
            [f"model.layers.{i}.self_attn.attn" for i in (3, 31, 63)]
            + [f"model.layers.{i}.linear_attn" for i in range(64)
               if i not in (3, 31, 63)])


def test_blob_from_dump_requires_n_layer_with_tail(tmp_path, monkeypatch):
    # rec_states given without n_layer must fail loudly, not default to 4B
    import stream_connector as sc
    import synthesize as S
    import torch

    attn = [3, 7]
    canon = {L: torch.zeros(4, 4, 2, 256, dtype=torch.float16) for L in attn}

    class FakeDump:
        pass

    monkeypatch.setattr(S, "canonical_from_dump_pair",
                        lambda d, a, dtype=torch.float16: canon)
    rec = {5: {"conv": torch.zeros(3, 8), "ssm": torch.zeros(4, 8, 8)}}
    with pytest.raises(ValueError, match="n_layer is required"):
        sc.blob_from_dump("ignored", attn, rec_states=rec)
