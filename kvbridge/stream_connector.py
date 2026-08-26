"""StreamConnector: serve the synthesized llama.cpp KV blob over TCP (issue #11).

Subclass of HMADumpConnector. The parent (via ExampleConnector) still writes
whole-request safetensors per layer into the shared storage path; on
wait_for_save() we synthesize the FULL hybrid seq-state blob — attention
section from the dump pair files (kvbridge/synthesize.py, do NOT reimplement)
plus the recurrent R/S tail captured from the linear-attention layers — and
serve it on a simple TCP socket: single accept, then length-prefixed pull
requests (see framing.py / WIRE.md).

Recurrent tail capture (issue #16 finding: attention-only blobs are not
restorable on this hybrid; llama.cpp never rebuilds delta/recurrent states):
at the LAST attention layer's save hook (layer 31 runs last in the forward),
every linear layer's `kv_cache` in the forward context already holds the
post-batch conv/ssm state for its mamba slot:
    conv_state [slots, d_conv-1, conv_channels] bf16
    ssm_state  [slots, n_v_heads, v_dim, k_dim] f32
The request's linear-group block id is the slot. We snapshot both per layer
and serialize them in llama-memory-recurrent.cpp state_write order.

NOTE on "no Python in the KV byte path": that rule is about the Mac's runtime
decode path. This code is control-plane / serving-side (Spark, vLLM host) —
permitted. The Mac-side runtime consumer (llama.cpp restore) never goes
through Python.

Deploy: this file + synthesize.py + framing.py live in the repo under
kvbridge/ and are scp'd to ~/kvbridge on the Spark, where
PYTHONPATH=$HOME/kvbridge lets the connector import them.
"""

import glob
import hashlib
import io
import json
import os
import re
import socket
import threading

import torch

try:
    from hma_dump_connector import HMADumpConnector
except ImportError:  # Mac-side: connector only exists on the Spark (vLLM host)
    HMADumpConnector = object

import framing
import synthesize as synth_mod


def derive_layer_geometry(group_names: list[str]) -> tuple[int, int]:
    """All kv-cache group layer names -> (last_attn_layer, n_layer).

    Pure function (unit-tested): regex-derived from names like
    `model.layers.N.self_attn...`. Fails loudly on underivable geometry
    rather than falling back to any hardcoded per-model value (issue #23).
    """
    attn_ids = sorted(int(m.group(1)) for n in group_names
                      if (m := re.search(r"layers\.(\d+)\.self_attn", n)))
    any_ids = [int(m.group(1)) for n in group_names
               if (m := re.search(r"layers\.(\d+)\.", n))]
    if not attn_ids or not any_ids:
        raise RuntimeError(
            "STREAMC: could not derive layer geometry from kv_cache_groups "
            f"(attn={attn_ids[:5]}...): refusing to run with wrong "
            "hardcoded layer indices (issue #23)")
    n_layer = max(any_ids) + 1
    # sanity: attn layers must tile the stack at a fixed interval (every
    # 4th on both qwen35-4b and qwen38-27b); catches geometry regressions
    if n_layer % len(attn_ids) or n_layer // len(attn_ids) < 2:
        raise RuntimeError(
            f"STREAMC: implausible layer geometry: {len(attn_ids)} attn "
            f"layers in {n_layer} total (expected a fixed interval)")
    return attn_ids[-1], n_layer


def list_blobs(host: str, port: int) -> list[dict]:
    """Client-side convenience: ask a BlobServer which blobs it holds."""
    import pull_blob
    return pull_blob.list_blobs(host, port)

DEFAULT_PORT = 52901
DEFAULT_ACCEPT_TIMEOUT = 300.0
CHUNK = 4 * 1024 * 1024


def safe_hash_key(token_ids: list[int]) -> str:
    """Prompt hash exactly as vLLM's ExampleConnector names dump dirs
    (hash of the token-id bytes), so blob keys == dump dir basenames."""
    from vllm.utils.hashing import safe_hash
    return safe_hash(torch.tensor(token_ids).numpy().tobytes(),
                     usedforsecurity=False).hexdigest()


def attn_layers_from_dump(dump_dir: str) -> list[int]:
    """[3,7,...,31] from the per-layer safetensors filenames in the dump."""
    layers = []
    for path in glob.glob(os.path.join(dump_dir, "*.safetensors")):
        m = re.search(r"layers\.(\d+)\.", os.path.basename(path))
        if m:
            layers.append(int(m.group(1)))
    return sorted(layers)


def pack_shard_payload(dump_dir: str, rec_states: dict) -> bytes:
    """Pack a rank's dump dir + recurrent states into one payload (issue #37).

    Rank >= 1 connectors serve this via BlobServer on stream_port+1; rank 0
    fetches it and merges. torch.save keeps tensors together with metadata.
    """
    files = {}
    for name in sorted(os.listdir(dump_dir)):
        if name.endswith(".safetensors"):
            with open(os.path.join(dump_dir, name), "rb") as f:
                files[name] = f.read()
    buf = io.BytesIO()
    torch.save({"files": files, "rec_states": rec_states}, buf)
    return buf.getvalue()


def fetch_shard_payload(host: str, port: int, key: str) -> dict:
    """Fetch and unpack a peer rank's shard payload from its BlobServer.

    The peer publishes after packing its whole dump dir (torch.save of
    ~200 MB at 600 tokens, more for longer prompts). Profiling (issue
    #41) showed rank 1 packs 100 MiB in ~0.15 s — the fetch was slow only
    because the FIRST attempt lands ~50 ms before publish and the old 2 s
    retry sleep turned that into 2.3 s per request. Poll fast instead:
    300 x 0.1 s = 30 s budget, which still covers the largest prompts.
    """
    import time
    for attempt in range(300):  # 30 s budget, 100 ms polls (issue #41)
        try:
            return _fetch_shard_payload_once(host, port, key)
        except (ConnectionRefusedError, TimeoutError, OSError) as e:
            last = e
        except RuntimeError as e:  # "unknown blob key" until the peer publishes
            if "unknown blob key" not in str(e):
                raise
            last = e
        time.sleep(0.1)
    raise last


def _fetch_shard_payload_once(host: str, port: int, key: str) -> dict:
    with socket.create_connection((host, port), timeout=30) as s:
        framing.send_frame(s, json.dumps({"op": "get", "key": key, "offset": 0}).encode())
        header = json.loads(framing.recv_frame(s))
        if "error" in header:
            raise RuntimeError(f"peer {host}:{port}: {header['error']}")
        buf = io.BytesIO()
        while True:
            frame = framing.recv_frame(s)
            if not frame:
                break
            buf.write(frame)
    buf.seek(0)
    return torch.load(buf, weights_only=False)


def blob_from_dump(dump_dir: str, attn_layers: list[int],
                   rec_states: dict | None = None,
                   n_layer: int = None) -> bytes:
    """Full hybrid blob: attention section + (optional) recurrent tail.

    n_layer (total model layers, written into the recurrent section header)
    is deliberately required with rec_states — a silent default would bake
    one model's geometry into another model's tail (review, PR #24).
    """
    canonical = synth_mod.canonical_from_dump_pair(dump_dir, attn_layers)
    blob = synth_mod.synthesize(canonical, attn_layers)
    if rec_states:
        if n_layer is None:
            raise ValueError("n_layer is required when rec_states is given")
        t = canonical[attn_layers[0]].shape[0]
        blob += synth_mod.recurrent_tail(
            {L: v["conv"] for L, v in rec_states.items()},
            {L: v["ssm"] for L, v in rec_states.items()},
            n_layer=n_layer, last_pos=t - 1)
    return blob


def validate_coverage(rec_states: dict, expected_linear_layers=None) -> None:
    """Fail-loud partial-coverage gate (issue #20: replace warn-once).

    An attention-only blob (no recurrent tail) is NOT restorable on this
    hybrid (issue #16), and a tail missing some linear layers produces a
    blob llama.cpp accepts but that diverges — both are hard errors now.
    """
    if not rec_states:
        raise RuntimeError(
            "no recurrent states captured; refusing to serve an "
            "attention-only blob (issue #16: not restorable)")
    have = set(rec_states)
    want = set(expected_linear_layers) if expected_linear_layers is not None else have
    missing = want - have
    if missing:
        raise RuntimeError(
            f"partial recurrent coverage: missing linear layers {sorted(missing)} "
            f"(captured {sorted(have)}, expected {sorted(want)})")
    for L, v in rec_states.items():
        if "conv" not in v or "ssm" not in v:
            raise RuntimeError(f"linear layer {L}: incomplete state (need conv+ssm)")


class BlobServer:
    """Accept-loop TCP server over a key -> blob store (issue #20).

    v0 served exactly one blob on exactly one connection; this keeps a
    listening socket open for the process lifetime, accepts connections in
    a loop, and answers `list` / `get` (resumable) requests. Blobs are
    keyed by prompt hash so each successive generate stays pullable.
    """

    def __init__(self, port: int, host: str = "0.0.0.0",
                 accept_timeout: float = DEFAULT_ACCEPT_TIMEOUT):
        self._port = port
        self._host = host
        self._accept_timeout = accept_timeout
        self._blobs: dict[str, bytes] = {}
        self._meta: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._srv_socket: socket.socket | None = None

    def add(self, key: str, blob: bytes, meta: dict | None = None) -> None:
        with self._lock:
            self._blobs[key] = blob
            self._meta[key] = dict(meta or {})

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            # Bind on the CALLER's thread (issue #37 flake): binding inside
            # the accept-loop daemon meant a failed bind (port held by a
            # previous engine generation) killed the thread silently while
            # add() kept succeeding — the peer "published" into a void.
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                srv.bind((self._host, self._port))
            except OSError as e:
                srv.close()
                print(f"STREAMC BlobServer bind {self._host}:{self._port} "
                      f"FAILED: {e!r} — blobs added now will NOT be "
                      f"reachable (issue #37)", flush=True)
                raise
            srv.listen(8)
            srv.settimeout(1.0)
            self._srv_socket = srv
            self._stop.clear()
            self._thread = threading.Thread(target=self._accept_loop,
                                            daemon=True, name="BlobServer")
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=5)
        self._thread = None
        if self._srv_socket is not None:
            self._srv_socket.close()
            self._srv_socket = None

    def _accept_loop(self) -> None:
        srv = self._srv_socket  # pre-bound in start() (issue #37)
        if srv is None:
            return
        try:
            while not self._stop.is_set():
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                # issue #37 flake #2: serve each connection in its own thread.
                # A serial loop wedges forever on a silent connection (a port
                # probe that connects and sends nothing blocks recv_frame with
                # no timeout) — real clients then complete the TCP handshake
                # via the backlog but never get served (connect succeeds,
                # reply never comes — exactly the observed rank-0 timeout).
                conn.settimeout(60.0)
                t = threading.Thread(target=self._serve_conn_threaded,
                                     args=(conn, addr), daemon=True)
                t.start()
        finally:
            srv.close()

    def _serve_conn_threaded(self, conn, addr) -> None:
        print(f"STREAMC BlobServer conn from {addr[0]}:{addr[1]}", flush=True)
        try:
            with conn:
                self._serve_conn(conn)
        except Exception as e:
            # never let a client kill the server; log loudly instead
            print(f"STREAMC BlobServer error serving {addr}: {e!r}", flush=True)

    def _serve_conn(self, conn) -> None:
        while True:
            try:
                req = json.loads(framing.recv_frame(conn))
            except framing.IncompleteFrameError:
                return
            except OSError:
                return  # RST after client closed with unread data (resume case)
            op = req.get("op")
            if op == "list":
                with self._lock:
                    entries = [{
                        "key": k,
                        "blob_size": len(self._blobs[k]),
                        **self._meta.get(k, {}),
                    } for k in self._blobs]
                framing.send_frame(conn, json.dumps(entries).encode())
            elif op == "get":
                self._serve_get(conn, req)
            else:
                framing.send_frame(conn, json.dumps(
                    {"error": f"bad op: {req!r}"}).encode())

    def _serve_get(self, conn, req) -> None:
        with self._lock:
            key = req.get("key")
            if key is None and len(self._blobs) == 1:
                key = next(iter(self._blobs))
            blob = self._blobs.get(key)
            meta = dict(self._meta.get(key, {}))
        if blob is None:
            if key is None and len(self._blobs) > 1:
                framing.send_frame(conn, json.dumps({"error":
                    "multiple blobs served; specify 'key' (see op 'list')"}).encode())
            else:
                # honest miss even when other blobs exist (issue #37: this
                # used to misreport a keyed miss as "multiple blobs")
                framing.send_frame(conn, json.dumps(
                    {"error": f"unknown blob key: {key!r}"}).encode())
            return
        off = int(req.get("offset", 0))
        if not 0 <= off <= len(blob):
            framing.send_frame(conn, json.dumps(
                {"error": f"offset {off} out of range 0..{len(blob)}"}).encode())
            return
        framing.send_frame(conn, json.dumps({
            "key": key,
            "blob_size": len(blob),
            "offset": off,
            "sha256": hashlib.sha256(blob).hexdigest(),
            # issue #20: identity fields so the Mac can verify before restore;
            # old clients ignore unknown JSON fields
            "prompt_token_ids": meta.get("prompt_token_ids"),
            "model_id": meta.get("model_id"),
        }).encode())
        try:
            for start in range(off, len(blob), CHUNK):
                framing.send_frame(conn, blob[start:start + CHUNK])
            framing.send_frame(conn, b"")  # end-of-stream sentinel frame
        except OSError:
            # client hung up mid-stream; it resumes by key + offset
            return


def serve_blob(blob: bytes, port: int, host: str = "0.0.0.0",
               accept_timeout: float = DEFAULT_ACCEPT_TIMEOUT) -> "BlobServer":
    """Back-compat single-blob helper: start a BlobServer with one blob."""
    srv = BlobServer(port, host=host, accept_timeout=accept_timeout)
    srv.add("default", blob)
    srv.start()
    return srv



class StreamConnector(HMADumpConnector):
    # HMA support comes from the parent (HMADumpConnector declares SupportsHMA
    # with a non-owning observer; required for TP2, ops#2 / issue #37).
    def request_finished_all_groups(self, request, block_ids):
        return False, None

    """Dump to files like the parent, then serve the synthesized blob."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._port = int(self._kv_transfer_config.get_from_extra_config(
            "stream_port", DEFAULT_PORT))
        self._accept_timeout = float(self._kv_transfer_config.get_from_extra_config(
            "stream_accept_timeout", DEFAULT_ACCEPT_TIMEOUT))
        self._linear_group_idx = None
        # model-agnostic layer geometry (issue #23: 27B has 64 layers, last
        # attn layer 63; 4B had 32/31 — previously hardcoded)
        all_names = [n for g in self._kv_cache_config.kv_cache_groups
                     for n in g.layer_names]
        self._last_attn_layer, self._n_layer = derive_layer_geometry(all_names)
        for i, g in enumerate(self._kv_cache_config.kv_cache_groups):
            # block_ids[0] in the parent's store path assumes group 0 is
            # the ATTENTION group — print the real order so a geometry
            # change (Ornith MoE hybrid) can't break it silently.
            print(f"STREAMC group[{i}] sample={g.layer_names[:2]}", flush=True)
            if not any("self_attn" in n for n in g.layer_names):
                self._linear_group_idx = i
        # layer index -> {"conv": tensor[slot], "ssm": tensor[slot]} (cpu)
        self._rec_states: dict[int, dict] = {}
        self._logged_decode_skip = False
        self._served: set[str] = set()
        # prompt hash (dump dir basename) -> prompt token ids (issue #20)
        self._prompt_ids: dict[str, list[int]] = {}
        # keys dumped since the last wait_for_save() (issue #26): only the
        # current round's dumps may be served — self._rec_states belongs to
        # this round, so pairing them with an older dump dir (e.g. left on
        # disk by a previous engine run) would serve a mismatched blob
        self._pending_round_keys: set[str] = set()
        self._server: BlobServer | None = None
        # issue #37 (TP2): comma list of peer shard servers (host:port), set
        # ONLY in TP launches; absent => solo behavior, byte-for-byte unchanged.
        peers = self._kv_transfer_config.get_from_extra_config("shard_peers", "")
        self._shard_peers = [p.strip() for p in peers.split(",") if p.strip()]
        # issue #37: rank is NOT reliable here (torch.distributed initializes
        # after connector construction) — re-evaluated in wait_for_save.
        self._rank = 0
        self._model_id = getattr(
            getattr(self, "_vllm_config", None), "model_config", None)
        self._model_id = getattr(self._model_id, "model", None) or "unknown"

    # ---- capture -------------------------------------------------------

    def build_connector_meta(self, scheduler_output):
        # runs on the SCHEDULER-role connector; the metadata object is
        # shipped to the WORKER-role connector, so we piggyback the mamba
        # slot on it (ReqMeta itself has no field for it)
        meta = super().build_connector_meta(scheduler_output)
        if self._linear_group_idx is not None:
            for new_req in scheduler_output.scheduled_new_reqs:
                ids = new_req.block_ids[self._linear_group_idx]
                if ids:
                    meta.linear_slot = ids[-1]  # mamba page size == 1
        return meta

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        import time as _time
        self._t_last_hook = _time.perf_counter()  # issue #41 dumpw boundary
        # record prompt hash -> token ids for every stored request (issue #20).
        # Since the issue #26 fix the ReqMeta carries the FULL prompt ids
        # (no block-size truncation), so this is also where we catch a
        # regression back to empty captures — fail loud rather than serve an
        # unverifiable blob.
        captured_ids = {}
        try:
            cm = self._get_connector_metadata()
            for request in getattr(cm, "requests", []):
                if not getattr(request, "is_store", False):
                    continue
                ids = [int(t) for t in request.token_ids.tolist()]
                if not ids:
                    raise RuntimeError(
                        "STREAMC: store request with empty token_ids — "
                        "block-size truncation regression (issue #26)")
                self._prompt_ids[safe_hash_key(ids)] = ids
                captured_ids[safe_hash_key(ids)] = ids
                self._pending_round_keys.add(safe_hash_key(ids))
        except Exception:
            import traceback; print("STREAMC prompt-id capture FAILED",
                                    traceback.format_exc(), flush=True)
            raise
        super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
        # persist ids next to the dump (issue #26): the dump dir now exists
        # (parent created it while saving this layer), and the sidecar lets
        # a restarted engine re-serve old dumps instead of crashing on them
        try:
            from hma_dump_connector import PROMPT_IDS_SIDECAR
            for key, ids in captured_ids.items():
                dump_dir = os.path.join(self._storage_path, key)
                sidecar = os.path.join(dump_dir, PROMPT_IDS_SIDECAR)
                if not os.path.exists(sidecar):
                    os.makedirs(dump_dir, exist_ok=True)
                    with open(sidecar, "w") as f:
                        json.dump({"prompt_token_ids": ids}, f)
        except Exception:
            import traceback; print("STREAMC sidecar write FAILED",
                                    traceback.format_exc(), flush=True)
            raise
        m = re.search(r"layers\.(\d+)\.self_attn", layer_name)
        if not m or int(m.group(1)) != self._last_attn_layer:
            return  # last attn layer runs last: linear states are final
        # mamba slot is per-request; re-derive from this round's metadata,
        # never cache across rounds (a stale slot would silently capture the
        # wrong request's recurrent states)
        meta = self._get_connector_metadata()
        slot = getattr(meta, "linear_slot", None)
        if slot is None:
            if not any(getattr(r, "is_store", False)
                       for r in getattr(meta, "requests", [])):
                # decode step: no store request in this round, nothing to
                # capture. Log once — a regression that misclassifies prefill
                # rounds as decode would otherwise serve stale tails.
                if not self._logged_decode_skip:
                    print("STREAMC: no store request this round - skipping "
                          "recurrent capture (decode step)", flush=True)
                    self._logged_decode_skip = True
                return
            raise RuntimeError(
                "STREAMC: no mamba slot on connector metadata for a store "
                "request; cannot capture recurrent states (fail-loud, "
                "issue #20)")
        # fail-loud: no more warn-and-continue on capture errors
        from vllm.forward_context import get_forward_context
        ctx = get_forward_context()
        md_all = getattr(ctx, "attn_metadata", None)
        captured = {}
        expected = set()
        for name, layer in ctx.no_compile_layers.items():
            lm = re.search(r"layers\.(\d+)\.linear_attn", name)
            kc = getattr(layer, "kv_cache", None)
            if not lm or not isinstance(kc, (list, tuple)):
                continue
            expected.add(int(lm.group(1)))
            # issue #39: every linear layer owns its OWN mamba slot
            # (prefill_state_indices run 1,2,3,... per layer). Resolve this
            # layer's slot from its GDN metadata; the old single
            # block_ids[-1] slot read 47/48 layers' states wrong.
            lslot = None
            if isinstance(md_all, dict):
                md = md_all.get(getattr(layer, "prefix", None))
                if md is not None:
                    for attr in ("prefill_state_indices",
                                 "non_spec_state_indices_tensor"):
                        idx = getattr(md, attr, None)
                        if idx is not None and len(idx):
                            lslot = int(idx[0].item() if hasattr(idx[0], "item")
                                        else idx[0])
                            break
            if lslot is None:
                raise RuntimeError(
                    f"STREAMC: no per-layer mamba slot for {name} — "
                    "cannot capture recurrent state (issue #39)")
            conv_t = kc[0][lslot].detach().float().cpu().clone()
            ssm_t = kc[1][lslot].detach().float().cpu().clone()
            # issue #39: states stored RAW here. The GQA v-head permutation
            # is applied in wait_for_save AFTER rank shards are merged —
            # under TP2 llama-order heads interleave across ranks, so a
            # per-rank permutation cannot be concatenated into global order.
            captured[int(lm.group(1))] = {"conv": conv_t, "ssm": ssm_t}
            # issue #39 debug: dump ALL slots of the FIRST linear layer's
            # pools so the Mac can locate which slot holds the real state
            if int(lm.group(1)) == 0 and os.environ.get("STREAMC_DEBUG_POOLS"):
                torch.save({"conv_pool": kc[0].detach().float().cpu(),
                            "ssm_pool": kc[1].detach().float().cpu(),
                            "slot": int(slot)},
                           "/tmp/streamc_pools.pt")
                print(f"STREAMC pool dump: conv {tuple(kc[0].shape)} "
                      f"ssm {tuple(kc[1].shape)} slot={slot} -> /tmp/streamc_pools.pt",
                      flush=True)
        validate_coverage(captured, expected_linear_layers=expected)
        self._rec_states = captured

    # ---- serve ---------------------------------------------------------

    def wait_for_save(self):
        """Called after each save round; files + rec states are ready now.

        Registers each new dump as a blob keyed by prompt hash on the
        persistent BlobServer (accept-loop, issue #20) — successive
        generates are each pullable for the process lifetime.
        """
        try:
            # issue #26: serve ONLY this round's dumps (keyed by the hash of
            # the FULL prompt ids). Dirs left on disk by earlier rounds or a
            # previous engine run are ignored — _rec_states belongs to the
            # current round only, and pre-fix poison dirs (d41d8cd9…) must
            # never be served. Previously this loop globbed the whole
            # storage path, which (a) crashed the engine on restart over old
            # dirs with no recorded ids and (b) risked pairing an old dump
            # with the wrong round's recurrent states.
            keys = sorted(self._pending_round_keys)
            self._pending_round_keys.clear()
            for key in keys:
                dump_dir = os.path.join(self._storage_path, key)
                attn_layers = attn_layers_from_dump(dump_dir)
                if not attn_layers:
                    raise RuntimeError(
                        f"round dump {key[:12]} has no layer files")
                if key in self._served:
                    continue
                self._served.add(key)
                validate_coverage(self._rec_states)  # fail-loud, issue #20
                token_ids = self._prompt_ids.get(key)
                if not token_ids:
                    # unreachable given the save_kv_layer guard, but keep the
                    # issue #20 invariant: never serve an unverifiable blob
                    raise RuntimeError(
                        f"no prompt token ids recorded for blob {key}; "
                        "refusing to serve unverifiable blob (issue #20)")
                import torch.distributed as _dist
                self._rank = _dist.get_rank() if _dist.is_initialized() else 0
                # issue #41 phase timing: breakdown of the bridge overhead.
                # dumpw = last save hook -> wait_for_save (parent's
                # safetensors writes happen during hooks)
                import time as _time
                T0 = _time.perf_counter()
                T_dumpw = T0 - getattr(self, "_t_last_hook", T0)
                print(f"STREAMC rank={self._rank} shard_peers={self._shard_peers} "
                      f"serving key={key[:12]}", flush=True)
                if self._rank > 0:
                    # issue #37: publish this rank's shard for rank 0 to merge
                    if self._server is None:
                        self._server = BlobServer(
                            self._port + 1, accept_timeout=self._accept_timeout)
                        self._server.start()
                    T1 = _time.perf_counter()
                    payload = pack_shard_payload(dump_dir, self._rec_states)
                    T2 = _time.perf_counter()
                    self._server.add(key, payload, {})
                    print(f"STREAMC rank{self._rank} published shard key={key[:12]} "
                          f"on {self._port + 1}", flush=True)
                    print(f"STREAMC-TIMING rank={self._rank} tokens={len(token_ids)} "
                          f"dumpw={T_dumpw:.2f} pack={T2-T1:.2f} "
                          f"payload={len(payload)//1048576}MiB", flush=True)
                    continue
                dump_for_blob, rec_for_blob = dump_dir, self._rec_states
                T_fetch = T_peerw = 0.0
                if self._shard_peers:
                    from tp2_merge import merged_dump_dir, merge_recurrent_shards
                    peer_dirs = []
                    recs = [self._rec_states]
                    for peer in self._shard_peers:
                        host, p = peer.rsplit(":", 1)
                        T1 = _time.perf_counter()
                        pl = fetch_shard_payload(host, int(p), key)
                        T2 = _time.perf_counter()
                        pdir = dump_dir.rstrip("/") + ".peer_" + host
                        os.makedirs(pdir, exist_ok=True)
                        for name, blobf in pl["files"].items():
                            with open(os.path.join(pdir, name), "wb") as f:
                                f.write(blobf)
                        T3 = _time.perf_counter()
                        T_fetch += T2 - T1
                        T_peerw += T3 - T2
                        peer_dirs.append(pdir)
                        recs.append(pl["rec_states"])
                    T1 = _time.perf_counter()
                    dump_for_blob = merged_dump_dir(dump_dir, peer_dirs)
                    # issue #39: conv shards are per-rank [q_r|k_r|v_r]
                    # segments — merge_rank_states interleaves segments and
                    # concats ssm heads in rank order (global vLLM order).
                    from gdn_layout import merge_rank_states
                    rec_for_blob = merge_rank_states(recs)
                    T2 = _time.perf_counter()
                else:
                    T1 = T2 = _time.perf_counter()
                # issue #39: llama.cpp v-head ordering applied to the final
                # (rank-merged or solo) states, with full-model geometry.
                from gdn_layout import finalize_states
                rec_for_blob = finalize_states(rec_for_blob)
                T3 = _time.perf_counter()
                blob = blob_from_dump(dump_for_blob, attn_layers, rec_for_blob,
                                      n_layer=self._n_layer)
                T4 = _time.perf_counter()
                print(f"STREAMC-TIMING rank=0 tokens={len(token_ids)} "
                      f"dumpw={T_dumpw:.2f} fetch={T_fetch:.2f} "
                      f"peerw={T_peerw:.2f} merge={T2-T1:.2f} "
                      f"final={T3-T2:.2f} synth={T4-T3:.2f} "
                      f"total={T4-T0:.2f}", flush=True)
                print(f"STREAMC serving {len(blob)} bytes from {dump_dir} "
                      f"key={key[:12]} tokens={len(token_ids)} port {self._port} "
                      f"rec_layers={len(self._rec_states)}",
                      flush=True)
                if self._server is None:
                    self._server = BlobServer(
                        self._port, accept_timeout=self._accept_timeout)
                self._server.add(key, blob, {
                    "prompt_token_ids": token_ids,
                    "model_id": self._model_id,
                })
                self._server.start()
        except Exception:
            import traceback; print("STREAMC serve FAILED", traceback.format_exc(), flush=True)
            raise
