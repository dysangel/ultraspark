"""Length-prefixed, resumable frame framing for the USPK KV wire (issue #11).

Frame = u64 big-endian payload length, then exactly that many payload bytes.

Resumability: recv_exact accumulates partial reads; if the connection drops
mid-frame it raises IncompleteFrameError carrying the partial payload and the
bytes still owed, so a caller can resume from a known-good offset (offset +
len(partial payload of completed frames)) by re-issuing a `get` with that
offset on a new connection. Frames are self-delimiting, so any byte offset
that falls on a frame boundary is a valid resume point; the server only
sends blob bytes on frame boundaries, and the client tracks them.
"""

import socket
import struct

U64 = struct.Struct(">Q")
MAX_FRAME = 16 * 1024 * 1024  # sanity cap; larger frames are refused


class IncompleteFrameError(ConnectionError):
    """Connection dropped inside a frame.

    Attributes:
        partial: bytes received so far of the incomplete payload.
        length: the frame's declared total payload length.
    """

    def __init__(self, partial: bytes, length: int):
        self.partial = partial
        self.length = length
        super().__init__(f"got {len(partial)}/{length} bytes before close")


def send_frame(sock: socket.socket, payload: bytes) -> None:
    """Send one length-prefixed frame."""
    if len(payload) > MAX_FRAME:
        raise ValueError(f"frame too large: {len(payload)} > {MAX_FRAME}")
    sock.sendall(U64.pack(len(payload)) + payload)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes; raise IncompleteFrameError(b'', n) on early EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise IncompleteFrameError(bytes(buf), n)
        buf += chunk
    return bytes(buf)


def recv_frame(sock: socket.socket) -> bytes:
    """Read one length-prefixed frame (resumable-safe: never over-reads)."""
    (length,) = U64.unpack(recv_exact(sock, U64.size))
    if length > MAX_FRAME:
        raise ValueError(f"frame too large: {length} > {MAX_FRAME}")
    return recv_exact(sock, length)


def iter_frames(sock: socket.socket):
    """Yield frames until the peer closes cleanly between frames.

    A close is "clean" iff it happened while reading the 8-byte length
    header (partial empty, owed == U64.size). A close inside a payload or
    mid-header-after-partial is re-raised as IncompleteFrameError.
    """
    while True:
        try:
            yield recv_frame(sock)
        except IncompleteFrameError as e:
            if not e.partial and e.length == U64.size:
                return  # clean EOF exactly at a frame boundary
            raise
