"""ExampleConnector + SupportsHMA, dump-only, attention-group aware.

The parent indexes new_req.block_ids[0], which under the hybrid memory
allocator is the FIRST cache group (the linear/delta group on Qwen3.5) —
wrong slots for the attention layers it then reads. We pick the group
that owns the full-attention layers instead.

Issue #26: vLLM's ExampleConnector truncates ReqMeta token_ids to a whole
number of attention blocks (align_to_block_size, floor). On this hybrid the
attention page size is 784 tokens, so ANY prompt shorter than ~785 tokens
produced an EMPTY token tensor: the dump dir got named by the hash of zero
bytes (d41d8cd9...), every short prompt collided into it, and — once that
poison dir existed — ExampleConnector._found_match_for_prompt() (which
hashes the same block-aligned prefix slice, i.e. also zero bytes for short
prompts) reported a bogus "External Cache Hit" for every subsequent short
prompt, sending it down the load path so it never dumped again (symptom 2:
"text prompts stopped producing dumps mid-session").

Fix here, without touching vLLM:
  * build_connector_meta builds ReqMeta WITHOUT the block-size truncation
    (full token ids + full slot mapping), so dump dirs are keyed by the
    hash of the complete prompt and capture always works — including
    pre-tokenized ("prompt": [ids]) and short prompts.
  * get_num_new_matched_tokens is overridden to (0, False): the Spark never
    restores its own dumps (the Mac restores via the pulled blob), so the
    ExampleConnector external-prefix-cache path is pure liability here —
    it is what produced the false hits against d41d8cd9.
"""
import os

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
    ExampleConnectorMetadata,
    ReqMeta,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

PROMPT_IDS_SIDECAR = "prompt_ids.json"


def make_full_meta(token_ids: list[int], block_ids: list[int],
                   block_size: int, is_store: bool,
                   mm_hashes: list[str]) -> ReqMeta:
    """ReqMeta.make_meta without the block-size floor truncation (issue #26).

    ExampleConnector.make_meta does
        valid = align_to_block_size(len(token_ids), block_size)  # floor
        token_ids = token_ids[:valid]
    which yields an EMPTY tensor whenever len(token_ids) <= block_size (784
    on this hybrid) — the root cause of the d41d8cd9 empty-key collisions.
    We keep every prompt token (and a slot mapping of the same length, as
    ReqMeta documents), so dump keys are unique per prompt and the saved
    tensors cover the full prompt including the final partial block.

    slot_mapping is built exactly like the upstream version, just sliced to
    the FULL token count instead of the block-aligned count.
    """
    if not token_ids:
        raise RuntimeError(
            "HMA: store request with empty prompt token ids — refusing to "
            "build a dump keyed by the hash of nothing (issue #26)")
    n = min(len(token_ids), len(block_ids) * block_size)
    block_ids_tensor = torch.tensor(block_ids)
    num_blocks = block_ids_tensor.shape[0]
    block_offsets = torch.arange(0, block_size)
    slot_mapping = (
        block_offsets.reshape((1, block_size))
        + block_ids_tensor.reshape((num_blocks, 1)) * block_size
    ).flatten()[:n]
    return ReqMeta(
        token_ids=torch.tensor(token_ids)[:n],
        slot_mapping=slot_mapping,
        is_store=is_store,
        mm_hashes=mm_hashes,
    )


def dump_key_for(token_ids: list[int]) -> str:
    """Hash of the FULL token-id bytes — what dump dirs are named with when
    built via make_full_meta (request.token_ids is hashed verbatim by
    ExampleConnector._generate_filename_debug)."""
    from vllm.utils.hashing import safe_hash
    return safe_hash(torch.tensor(token_ids).numpy().tobytes(),
                     usedforsecurity=False).hexdigest()


class HMADumpConnector(ExampleConnector, SupportsHMA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._attn_group_idx = None
        groups = self._kv_cache_config.kv_cache_groups
        for i, g in enumerate(groups):
            names = list(g.layer_names)
            if any("self_attn" in n for n in names):
                self._attn_group_idx = i
                logger.info("HMA: attention group idx=%d layers=%d e.g. %s",
                            i, len(names), names[:3])

    def request_finished_all_groups(self, request, block_ids):
        return False, None

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int):
        # Never load from our own dumps (issue #26): the ExampleConnector
        # match hashes a block-FLOOR prefix of the prompt — the empty slice
        # for any prompt shorter than the 784-token attention page — which
        # false-matched the d41d8cd9 poison dir and silently disabled
        # dumping for every short prompt after the first one. The Spark's
        # role is dump+serve only; restore happens on the Mac from the
        # pulled blob.
        return 0, False

    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        meta = ExampleConnectorMetadata()
        idx = self._attn_group_idx if self._attn_group_idx is not None else 0
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            mm_hashes = [f.identifier for f in new_req.mm_features]
            key = dump_key_for(token_ids) if token_ids else None
            if key and os.path.isdir(os.path.join(self._storage_path, key)):
                # already dumped this exact prompt this process/disk —
                # idempotence the load path used to (accidentally) provide
                logger.info("HMA: skip store tokens=%d (dump %s exists)",
                            len(token_ids), key[:12])
                continue
            logger.info("HMA: store tokens=%d groups=%s using group %d",
                        len(token_ids), [len(b) for b in new_req.block_ids], idx)
            meta.requests.append(make_full_meta(
                token_ids, new_req.block_ids[idx], self._block_size,
                is_store=True, mm_hashes=mm_hashes,
            ))
        return meta
