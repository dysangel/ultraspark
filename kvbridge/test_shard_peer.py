"""Tests for the TP2 shard peer transport (issue #37).

Round trip: rank-1 connector packs its dump dir + recurrent states into one
payload, serves it via BlobServer; rank-0 fetches, merges, and synthesizes a
blob whose K rows are full-width (4 heads x 256 = 1024).

Run: python3 kvbridge/test_shard_peer.py
"""
import io
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(__file__))
from framing import send_frame, recv_frame  # noqa: E402
from stream_connector import BlobServer, pack_shard_payload, fetch_shard_payload  # noqa: E402
from tp2_merge import merge_recurrent_shards  # noqa: E402

LAYER = "language_model.model.layers.11.self_attn.attn.safetensors"


def _mkdump(root, seed):
    import safetensors.torch as st
    d = os.path.join(root, f"d{seed}")
    os.makedirs(d)
    st.save_file({"kv_cache": torch.randn(10, 2, 512).bfloat16()}, os.path.join(d, LAYER))
    return d


def test_pack_fetch_roundtrip():
    with tempfile.TemporaryDirectory() as root:
        d = _mkdump(root, 1)
        rec = {3: {"conv": torch.randn(2, 96), "ssm": torch.randn(2, 128, 64)}}
        payload = pack_shard_payload(d, rec)
        srv = BlobServer(53901, accept_timeout=1.0)
        srv.add("k1", payload, {})
        srv.start()
        try:
            got = fetch_shard_payload("127.0.0.1", 53901, "k1")
            assert set(got["files"]) == {LAYER}, got["files"].keys()
            assert set(got["rec_states"]) == {3}
            assert got["rec_states"][3]["conv"].shape == (2, 96)
        finally:
            srv.stop()


def test_merged_payload_full_width_blob():
    # end-to-end shape: two rank payloads -> merged dump -> canonical K width 1024
    from convert import to_canonical
    with tempfile.TemporaryDirectory() as root:
        d0, d1 = _mkdump(root, 0), _mkdump(root, 1)
        import safetensors.torch as st
        from tp2_merge import merged_dump_dir
        out = merged_dump_dir(d0, [d1])
        import glob, re
        layers = to_canonical(
            {int(re.search(r"layers\.(\d+)\.", os.path.basename(f)).group(1)):
             st.load_file(f)["kv_cache"] for f in glob.glob(out + "/*.safetensors")},
            n_layers=64)
        k = layers[11]
        assert k.shape == (10, 4, 2, 256), k.shape


def test_rec_merge_matches_tp2_merge():
    # issue #39 semantics: conv channel-concat, ssm head-concat
    a = (torch.randn(3, 96), torch.randn(2, 128, 64))
    b = (torch.randn(3, 96), torch.randn(2, 128, 64))
    R, S = merge_recurrent_shards(a, b)
    assert R.shape == (3, 192) and S.shape == (4, 128, 64)


def test_bind_failure_raises_on_start():
    # issue #37 flake: bind used to happen inside the accept-loop daemon
    # thread — a port held by a previous engine generation killed the thread
    # silently while add() kept succeeding. start() must raise instead.
    import socket as _socket
    blocker = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    blocker.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", 53902))
    blocker.listen(1)
    try:
        srv = BlobServer(53902, host="127.0.0.1", accept_timeout=1.0)
        srv.add("k", b"bytes")
        try:
            srv.start()
            raise AssertionError("start() on an occupied port must raise")
        except OSError:
            pass  # loud failure on the caller's thread — exactly what we want
    finally:
        blocker.close()


def test_silent_connection_does_not_wedge_server():
    # issue #37 flake #2: a connection that sends nothing used to block the
    # serial accept loop forever, so later clients connected (kernel backlog)
    # but were never served. Connections are now served per-thread.
    import socket as _socket
    with tempfile.TemporaryDirectory() as empty:
        payload = pack_shard_payload(empty, {})
    srv = BlobServer(53903, host="127.0.0.1", accept_timeout=1.0)
    srv.add("k1", payload)
    srv.start()
    silent = _socket.create_connection(("127.0.0.1", 53903), timeout=5)
    try:
        # real client must be served while the silent conn sits open
        got = fetch_shard_payload("127.0.0.1", 53903, "k1")
        assert got["files"] == {} and got["rec_states"] == {}
    finally:
        silent.close()
        srv.stop()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests green")
