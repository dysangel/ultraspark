"""Issue #26: Spark-side connector tests — id-array/short-prompt capture.

Root cause regression tests: ExampleConnector floors ReqMeta token_ids to a
whole number of attention blocks (page size 784 on this hybrid), so any
prompt shorter than ~785 tokens was captured as an EMPTY tensor — dumps
landed in the hash-of-nothing dir (d41d8cd9…) and, once that dir existed,
every later short prompt false-hit "External Cache Hit" and never dumped.

Run on the Spark (needs vllm-spike + torch):
  PATH=$HOME/vllm-spike/bin:$PATH python -m pytest test_hma_dump_connector.py
"""
import json
import os

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ReqMeta,
)

import hma_dump_connector as hma
import stream_connector as sc

BS = 784  # attention page size on the 27B hybrid (see serve_kv_27b.log)


def test_upstream_make_meta_empties_short_prompts():
    # documents the root cause (issue #26): floor alignment to block size
    meta = ReqMeta.make_meta(list(range(300)), [7], BS,
                             is_store=True, mm_hashes=[])
    assert meta.token_ids.shape[0] == 0
    assert meta.slot_mapping.shape[0] == 0


def test_make_full_meta_keeps_short_prompt_ids():
    ids = [151644, 872, 198, 9707, 11]
    meta = hma.make_full_meta(ids, [7], BS, is_store=True, mm_hashes=[])
    assert meta.token_ids.tolist() == ids
    assert meta.slot_mapping.tolist() == [
        7 * BS + i for i in range(len(ids))]


def test_make_full_meta_long_prompt_full_coverage():
    ids = list(range(2000))  # 2 full 784-blocks + a 432-token partial tail
    meta = hma.make_full_meta(ids, [3, 4, 5], BS, is_store=True, mm_hashes=[])
    # full prompt incl. the final partial block — no floor truncation
    assert meta.token_ids.shape[0] == 2000
    assert meta.slot_mapping.shape[0] == 2000
    assert meta.slot_mapping.tolist() == [s for b in (3, 4, 5)
                                          for s in range(b * BS, (b + 1) * BS)][:2000]


def test_make_full_meta_empty_ids_fails_loud():
    with pytest.raises(RuntimeError, match="empty prompt token ids"):
        hma.make_full_meta([], [1], BS, is_store=True, mm_hashes=[])


def test_dump_key_unique_and_matches_upstream_naming():
    from vllm.utils.hashing import safe_hash
    a = hma.dump_key_for([1, 2, 3])
    b = hma.dump_key_for([1, 2, 4])
    assert a != b
    assert len(a) == 32  # safe_hash digest width on this vLLM
    assert a == safe_hash(torch.tensor([1, 2, 3]).numpy().tobytes(),
                          usedforsecurity=False).hexdigest()


def test_get_num_new_matched_tokens_never_external_hits():
    # issue #26 symptom 2: with a poison d41d8cd9 dir on disk, the upstream
    # matcher reports a hit for EVERY short prompt. Our override must never
    # claim external tokens (the Spark only dumps; the Mac restores).
    class _Any:
        prompt_token_ids = [1, 2, 3]
        mm_features = []

    got = hma.HMADumpConnector.get_num_new_matched_tokens(
        object(), _Any(), 0)
    assert got == (0, False)


class _FakeReq:
    def __init__(self, ids, blocks):
        self.req_id = "r1"
        self.prompt_token_ids = ids
        self.mm_features = []
        self.block_ids = blocks


class _FakeSchedOut:
    def __init__(self, new_reqs):
        self.scheduled_new_reqs = new_reqs


def _connector(tmp_path):
    class _C(hma.HMADumpConnector):
        def request_finished_all_groups(self, request, block_ids):
            return False, None

    c = object.__new__(_C)
    c._attn_group_idx = 0
    c._block_size = BS
    c._storage_path = str(tmp_path)
    return c


def test_build_connector_meta_short_id_array_prompt_stored(tmp_path):
    c = _connector(tmp_path)
    ids = [9707, 11, 279, 4062, 7586, 1614, 382, 13]  # id-array style, short
    meta = c.build_connector_meta(_FakeSchedOut([_FakeReq(ids, ([1],))]))
    assert len(meta.requests) == 1
    r = meta.requests[0]
    assert r.is_store
    assert r.token_ids.tolist() == ids  # FULL ids, not truncated to empty


def test_build_connector_meta_skips_existing_dump(tmp_path):
    c = _connector(tmp_path)
    ids = [5, 6, 7]
    key = hma.dump_key_for(ids)
    os.makedirs(os.path.join(str(tmp_path), key))
    meta = c.build_connector_meta(_FakeSchedOut([_FakeReq(ids, ([1],))]))
    assert meta.requests == []  # idempotent: already dumped this exact prompt


# ---- StreamConnector round/sidecar handling (issue #26) --------------------

def _stream_connector(tmp_path, monkeypatch, requests):
    class _SC(sc.StreamConnector):
        def request_finished_all_groups(self, request, block_ids):
            return False, None

    inst = object.__new__(_SC)
    inst._storage_path = str(tmp_path)
    inst._prompt_ids = {}
    inst._pending_round_keys = set()
    inst._served = set()
    inst._rec_states = {5: {"conv": torch.zeros(3, 8),
                            "ssm": torch.zeros(4, 8, 8)}}
    inst._n_layer = 8
    inst._last_attn_layer = 7
    inst._port = 0
    inst._accept_timeout = 1.0
    inst._server = None
    inst._model_id = "test-model"

    class _CM:
        pass

    cm = _CM()
    cm.requests = requests
    cm.linear_slot = None
    inst._get_connector_metadata = lambda: cm
    # parent (HMADumpConnector) save: no-op; the real one writes layer files
    monkeypatch.setattr(hma.HMADumpConnector, "save_kv_layer",
                        lambda self, *a, **k: None)
    return inst


class _StoreReq:
    is_store = True
    mm_hashes = []

    def __init__(self, ids):
        self.token_ids = torch.tensor(ids)


def test_save_kv_layer_records_and_writes_sidecar(tmp_path, monkeypatch):
    inst = _stream_connector(tmp_path, monkeypatch, [_StoreReq([1, 2, 3])])
    inst.save_kv_layer("language_model.model.layers.5.linear_attn.proj",
                       None, None)
    key = sc.safe_hash_key([1, 2, 3])
    assert inst._prompt_ids[key] == [1, 2, 3]
    assert inst._pending_round_keys == {key}
    sidecar = os.path.join(str(tmp_path), key, hma.PROMPT_IDS_SIDECAR)
    assert json.load(open(sidecar))["prompt_token_ids"] == [1, 2, 3]


def test_save_kv_layer_empty_capture_fails_loud(tmp_path, monkeypatch):
    inst = _stream_connector(tmp_path, monkeypatch, [_StoreReq([])])
    with pytest.raises(RuntimeError, match="empty token_ids"):
        inst.save_kv_layer(
            "language_model.model.layers.5.linear_attn.proj", None, None)


def test_wait_for_save_serves_only_current_round(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "blob_from_dump", lambda *a, **k: b"BLOB")
    inst = _stream_connector(tmp_path, monkeypatch, [])
    # legacy + poison dirs on disk must be ignored entirely
    os.makedirs(os.path.join(str(tmp_path),
                             "d41d8cd98f00b204e9800998ecf8427e"))
    os.makedirs(os.path.join(str(tmp_path), "legacykey"))
    inst.wait_for_save()
    assert inst._server is None  # nothing served, no crash

    # a current-round dump gets served with the round's rec states
    key = sc.safe_hash_key([1, 2, 3])
    inst._prompt_ids[key] = [1, 2, 3]
    inst._pending_round_keys.add(key)
    os.makedirs(os.path.join(str(tmp_path), key))
    with open(os.path.join(str(tmp_path), key,
              "language_model.model.layers.7.self_attn.attn.safetensors"),
            "w") as f:
        f.write("x")
    inst.wait_for_save()
    assert inst._server is not None
    with inst._server._lock:
        assert key in inst._server._blobs
        assert inst._server._meta[key]["prompt_token_ids"] == [1, 2, 3]
        # poison dir never served
        assert "d41d8cd98f00b204e9800998ecf8427e" not in inst._server._blobs
    inst._server.stop()
