"""Unit tests for kvbridge framing + StreamConnector serve/pull loop.

Run: python3 -m pytest kvbridge/test_framing.py -q
"""

import json
import os
import socket
import threading

import pytest

import framing
import pull_blob
import stream_connector


class TestFraming:
    def test_roundtrip(self):
        a, b = socket.socketpair()
        with a, b:
            # send from a thread: 100 KiB won't fit the socketpair buffer,
            # so a same-thread sendall before recv would deadlock
            frames = [b"hello", b"", b"x" * 100000]
            t = threading.Thread(
                target=lambda: [framing.send_frame(a, f) for f in frames])
            t.start()
            assert framing.recv_frame(b) == b"hello"
            assert framing.recv_frame(b) == b""
            assert framing.recv_frame(b) == b"x" * 100000
            t.join()

    def test_no_overread(self):
        """recv_frame must not consume bytes of the next frame."""
        a, b = socket.socketpair()
        with a, b:
            a.sendall(framing.U64.pack(3) + b"abc" + framing.U64.pack(2) + b"zz")
            assert framing.recv_frame(b) == b"abc"
            assert framing.recv_frame(b) == b"zz"

    def test_partial_reads(self):
        """Server dribbles a frame one byte at a time; recv reassembles."""
        a, b = socket.socketpair()
        with a, b:
            payload = b"0123456789" * 500
            wire = framing.U64.pack(len(payload)) + payload

            def dribble():
                for i in range(0, len(wire), 1):
                    a.sendall(wire[i:i + 1])
            t = threading.Thread(target=dribble)
            t.start()
            assert framing.recv_frame(b) == payload
            t.join()

    def test_incomplete_raises_with_partial(self):
        a, b = socket.socketpair()
        with a, b:
            framing.send_frame(a, b"complete frame")
            assert framing.recv_frame(b) == b"complete frame"
            # now a truncated frame: header says 10, only 3 bytes arrive
            a.sendall(framing.U64.pack(10) + b"abc")
            a.close()
            with pytest.raises(framing.IncompleteFrameError) as ei:
                framing.recv_frame(b)
            assert ei.value.partial == b"abc"
            assert ei.value.length == 10

    def test_oversize_rejected(self):
        with pytest.raises(ValueError):
            framing.send_frame(None, b"x" * (framing.MAX_FRAME + 1))

    def test_iter_frames_clean_eof(self):
        a, b = socket.socketpair()
        with a, b:
            framing.send_frame(a, b"one")
            framing.send_frame(a, b"two")
            a.shutdown(socket.SHUT_WR)
            assert list(framing.iter_frames(b)) == [b"one", b"two"]


class TestServePull:
    def test_serve_pull_end_to_end(self, tmp_path):
        blob = bytes(range(256)) * 50000  # 12.8 MiB, spans >1 chunk
        port = 53901

        def run_server():
            stream_connector.serve_blob(blob, port, host="127.0.0.1")

        t = threading.Thread(target=run_server, daemon=True)
        t.start()

        # tiny retry loop: server may not be listening yet
        import time
        for _ in range(100):
            try:
                hdr = pull_blob.pull("127.0.0.1", port, str(tmp_path / "out.bin"))
                break
            except (ConnectionRefusedError, OSError):
                time.sleep(0.05)
        else:
            raise AssertionError("server never came up")
        t.join(timeout=10)

        import hashlib
        assert hdr["blob_size"] == len(blob)
        assert hashlib.sha256((tmp_path / "out.bin").read_bytes()).hexdigest() == hdr["sha256"]
        assert (tmp_path / "out.bin").read_bytes() == blob


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))


class TestAcceptLoopMultiBlob:
    """Issue #20: accept-loop server, multi-blob store, rich header."""

    def _start_server(self, tmp_path, blobs, meta=None, port=53911):
        srv = stream_connector.BlobServer(port, host="127.0.0.1")
        for i, (key, blob) in enumerate(blobs.items()):
            srv.add(key, blob, (meta or {}).get(key, {
                "prompt_token_ids": list(range(100 + i, 110 + i)),
                "model_id": "test-model",
            }))
        srv.start()
        import time
        for _ in range(100):
            try:
                stream_connector.list_blobs("127.0.0.1", port)
                return srv
            except OSError:
                time.sleep(0.05)
        raise AssertionError("server never came up")

    def test_two_connections_same_blob(self, tmp_path):
        """Accept-loop: a second connection after the first closed works."""
        blob = bytes(range(256)) * 3000
        srv = self._start_server(tmp_path, {"k1": blob})
        try:
            for i in range(2):
                hdr = pull_blob.pull("127.0.0.1", 53911, str(tmp_path / f"o{i}.bin"), key="k1")
                assert hdr["blob_size"] == len(blob)
                assert (tmp_path / f"o{i}.bin").read_bytes() == blob
        finally:
            srv.stop()

    def test_multi_blob_pull_by_key_and_list(self, tmp_path):
        blobs = {"aaaa": b"alpha" * 1000, "bbbb": b"beta" * 777}
        srv = self._start_server(tmp_path, blobs)
        try:
            entries = stream_connector.list_blobs("127.0.0.1", 53911)
            assert {e["key"] for e in entries} == set(blobs)
            for key, blob in blobs.items():
                hdr = pull_blob.pull("127.0.0.1", 53911, str(tmp_path / f"{key}.bin"), key=key)
                assert hdr["key"] == key
                assert (tmp_path / f"{key}.bin").read_bytes() == blob
        finally:
            srv.stop()

    def test_no_key_with_multiple_blobs_errors(self, tmp_path):
        srv = self._start_server(tmp_path, {"a": b"x" * 10, "b": b"y" * 10})
        try:
            with pytest.raises(RuntimeError, match="key"):
                pull_blob.pull("127.0.0.1", 53911, str(tmp_path / "o.bin"))
        finally:
            srv.stop()

    def test_unknown_key_errors(self, tmp_path):
        srv = self._start_server(tmp_path, {"a": b"x" * 10})
        try:
            with pytest.raises(RuntimeError, match="unknown"):
                pull_blob.pull("127.0.0.1", 53911, str(tmp_path / "o.bin"), key="nope")
        finally:
            srv.stop()

    def test_header_carries_identity(self, tmp_path):
        blob = b"ident" * 512
        ids = [5, 6, 7, 8]
        srv = self._start_server(tmp_path, {"k": blob},
                                 meta={"k": {"prompt_token_ids": ids, "model_id": "qwen35-4b-fp8"}})
        try:
            hdr = pull_blob.pull("127.0.0.1", 53911, str(tmp_path / "o.bin"), key="k")
            assert hdr["prompt_token_ids"] == ids
            assert hdr["model_id"] == "qwen35-4b-fp8"
        finally:
            srv.stop()

    def test_resume_with_key_across_reconnect(self, tmp_path):
        """Resume lands on the right blob even with others registered."""
        blob = os.urandom(3 * 1024 * 1024)  # >1 chunk, not a chunk multiple
        srv = self._start_server(tmp_path, {"aaa": b"noise", "kkk": blob})
        try:
            out = tmp_path / "o.bin"
            # first attempt: read the header + first data frame, then hang up
            with socket.create_connection(("127.0.0.1", 53911), timeout=10) as s:
                framing.send_frame(s, json.dumps({"op": "get", "key": "kkk", "offset": 0}).encode())
                framing.recv_frame(s)          # header
                first = framing.recv_frame(s)  # first data frame
            out.write_bytes(first)
            hdr = pull_blob.pull("127.0.0.1", 53911, str(out), key="kkk",
                                 resume_from=len(first))
            assert out.read_bytes() == blob
            assert hdr["blob_size"] == len(blob)
        finally:
            srv.stop()


class TestFailLoudCoverage:
    """Issue #20: refuse attention-only / partial-coverage blobs."""

    def test_no_recurrent_states_raises(self):
        with pytest.raises(RuntimeError, match="recurrent"):
            stream_connector.validate_coverage({})

    def test_partial_rec_states_raise(self):
        # 3 linear layers expected (inferred from ctx), only 2 captured
        rec = {2: {"conv": object(), "ssm": object()},
               6: {"conv": object(), "ssm": object()}}
        with pytest.raises(RuntimeError, match="coverage"):
            stream_connector.validate_coverage(rec, expected_linear_layers={2, 6, 10})

    def test_full_coverage_passes(self):
        rec = {L: {"conv": object(), "ssm": object()} for L in (2, 6, 10)}
        stream_connector.validate_coverage(rec, expected_linear_layers={2, 6, 10})


class TestRecurrentTail:
    def _real_tail_layout(self, blob, end):
        import struct as st
        o = end
        cc, = st.unpack_from("<I", blob, o); o += 4
        pos, nsid = st.unpack_from("<iI", blob, o); o += 8 + 4 * nsid
        s_trans, n_layer = st.unpack_from("<II", blob, o); o += 8
        rows = {}
        for table, count in (("conv", 24), ("ssm", 24)):  # 24 linear layers
            for _ in range(count):
                t, = st.unpack_from("<i", blob, o); o += 4
                rs, = st.unpack_from("<Q", blob, o); o += 8
                o += cc * rs
                rows.setdefault(table, []).append((t, rs))
        return cc, pos, n_layer, rows, o

    def test_tail_structure_matches_real_blob(self):
        import torch
        import synthesize as S
        blob = open("/tmp/lc_state.bin", "rb").read()
        import sys; sys.path.insert(0, "kvbridge")
        from verify import parse_attn
        _, _, end = parse_attn(blob)
        cc, pos, n_layer, rows, consumed = self._real_tail_layout(blob, end)
        assert cc == 1 and pos == 2000 and n_layer == 32
        assert consumed == len(blob)
        conv_layers = [r for r in rows["conv"]]
        assert len(conv_layers) == 24 and all(t == 0 and rs == 98304 for t, rs in conv_layers)
        ssm_layers = rows["ssm"]
        assert len(ssm_layers) == 24 and all(t == 0 and rs == 2097152 for t, rs in ssm_layers)

        conv = {L: torch.zeros(3, 8192) for L in range(32) if L % 4 != 3}
        ssm = {L: torch.zeros(32, 128, 128) for L in conv}
        tail = S.recurrent_tail(conv, ssm, n_layer=32, last_pos=2000)
        assert len(tail) == len(blob) - end, (len(tail), len(blob) - end)


class TestSidecar:
    """Issue #19: pull writes a <stem>.json sidecar with the prompt token ids
    so the Mac-side bridge hook (USPK_BRIDGE_DIR) can verify prompt identity."""

    def _server(self, ids):
        srv = stream_connector.BlobServer(53913, host="127.0.0.1")
        srv.add("kkk", b"blobbytes" * 1000,
                {"prompt_token_ids": ids, "model_id": "m"})
        srv.start()
        import time
        for _ in range(100):
            try:
                stream_connector.list_blobs("127.0.0.1", 53913)
                return srv
            except OSError:
                time.sleep(0.05)
        raise AssertionError("server never came up")

    def test_sidecar_written_next_to_blob(self, tmp_path):
        ids = [2001, 1234, 567]
        srv = self._server(ids)
        try:
            out = tmp_path / "req-abc.blob"
            hdr = pull_blob.pull("127.0.0.1", 53913, str(out), key="kkk")
            sidecar = tmp_path / "req-abc.json"
            assert sidecar.exists()
            sc = json.loads(sidecar.read_text())
            assert sc["token_ids"] == ids
            assert sc["blob_size"] == hdr["blob_size"]
            assert sc["sha256"] == hdr["sha256"]
            assert sc["key"] == "kkk"
        finally:
            srv.stop()

    def test_no_ids_no_sidecar(self, tmp_path):
        srv = stream_connector.BlobServer(53913, host="127.0.0.1")
        srv.add("kkk", b"x" * 100)  # no meta
        srv.start()
        import time
        for _ in range(100):
            try:
                stream_connector.list_blobs("127.0.0.1", 53913)
                break
            except OSError:
                time.sleep(0.05)
        try:
            out = tmp_path / "req-abc.blob"
            pull_blob.pull("127.0.0.1", 53913, str(out), key="kkk")
            assert not (tmp_path / "req-abc.json").exists()
        finally:
            srv.stop()
