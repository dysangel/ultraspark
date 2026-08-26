"""Pull-mode client: fetch the synthesized KV blob from the Spark (issue #11).

Usage: python3 kvbridge/pull_blob.py <host> <port> <out.bin>
Resumes automatically if the connection drops mid-transfer: it re-issues a
`get` with the offset of the last completed frame and appends. Validates the
sha256 the server advertises in the header frame.
"""

import hashlib
import json
import socket
import sys

import framing


def list_blobs(host: str, port: int) -> list[dict]:
    """Ask the server which blobs it holds (op `list`, issue #20)."""
    with socket.create_connection((host, port), timeout=30) as sock:
        framing.send_frame(sock, json.dumps({"op": "list"}).encode())
        reply = json.loads(framing.recv_frame(sock))
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(f"server error: {reply['error']}")
        return reply


def write_sidecar(out_path: str, header: dict) -> str | None:
    """Issue #19: persist the header's prompt identity next to the blob.

    <foo>.blob -> <foo>.json with the prompt token ids the blob covers, so the
    Mac-side USPK_BRIDGE_DIR hook can verify the request prefix before restore.
    No sidecar when the server didn't advertise ids (legacy v0 server).
    """
    ids = header.get("prompt_token_ids")
    if not ids:
        return None
    import os
    stem, ext = os.path.splitext(out_path)
    path = (stem if ext == ".blob" else out_path) + ".json"
    with open(path, "w") as f:
        json.dump({
            "token_ids": ids,
            "blob_size": header.get("blob_size"),
            "sha256": header.get("sha256"),
            "key": header.get("key"),
            "model_id": header.get("model_id"),
        }, f)
    return path


def pull(host: str, port: int, out_path: str, max_retries: int = 5,
         key: str | None = None, resume_from: int = 0) -> dict:
    """Fetch blob `key` (prompt hash) to out_path, resumable.

    With key=None and exactly one blob served, that blob is fetched; with
    several, the server errors and you should pick a key via list_blobs().
    """
    done = resume_from
    header = None
    h = hashlib.sha256()
    if done:
        # seed the running digest with already-written bytes
        with open(out_path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        import os
        assert os.path.getsize(out_path) == done
    attempts = 0
    while True:
        try:
            with socket.create_connection((host, port), timeout=30) as sock:
                framing.send_frame(sock, json.dumps(
                    {"op": "get", "key": key, "offset": done}).encode())
                header = json.loads(framing.recv_frame(sock))
                if "error" in header:
                    raise RuntimeError(f"server error: {header['error']}")
                if header["offset"] != done:
                    raise RuntimeError(f"server resumed at {header['offset']}, "
                                       f"expected {done}")
                while True:
                    frame = framing.recv_frame(sock)
                    if frame == b"":            # end-of-stream sentinel
                        with open(out_path, "ab") as f:
                            pass
                        got = h.hexdigest()
                        if got != header["sha256"]:
                            raise RuntimeError(
                                f"sha256 mismatch: got {got} want {header['sha256']}")
                        write_sidecar(out_path, header)
                        return header
                    if done + len(frame) > header["blob_size"]:
                        raise RuntimeError("server sent past blob_size")
                    with open(out_path, "ab") as f:
                        f.write(frame)
                    h.update(frame)
                    done += len(frame)
        except framing.IncompleteFrameError:
            attempts += 1
            if attempts > max_retries:
                raise
            # frames are self-delimiting; only completed frames were written


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    open(sys.argv[3], "wb").close()  # truncate output
    key = sys.argv[4] if len(sys.argv) == 5 else None
    if key == "-l":
        for e in list_blobs(sys.argv[1], int(sys.argv[2])):
            print(e["key"], e["blob_size"], e.get("model_id"))
        sys.exit(0)
    hdr = pull(sys.argv[1], int(sys.argv[2]), sys.argv[3], key=key)
    print(f"PULLED {hdr['blob_size']} bytes sha256={hdr['sha256'][:16]}… -> {sys.argv[3]}")
    print(f"  key={hdr.get('key')} model={hdr.get('model_id')} "
          f"prompt_tokens={len(hdr.get('prompt_token_ids') or [])}")
