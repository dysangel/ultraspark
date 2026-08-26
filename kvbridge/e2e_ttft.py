"""TTFT + greedy-continuation probe for the e2e demo (issue #10).

Usage: python3 kvbridge/e2e_ttft.py <base_url> <prompt_file> [--nprobs 5]

Sends a greedy (temp 0) streaming completion of `max_tokens` tokens against a
llama-server /completion endpoint and reports:
  * ttft_ms      — wall time from request start to first generated token byte
  * total_ms     — wall time to stream end
  * text         — the greedy continuation (stdout, JSON on last line)
  * top5         — top-5 next-token distribution of the FIRST generated token
                   (via nprobs), for cross-path sanity comparison

TTFT here includes prompt processing: that is exactly what the bridge is
supposed to eliminate.
"""

import argparse
import json
import sys
import time
import urllib.request


def one_request(base_url, prompt, max_tokens, want_nprobs, timeout=120):
    body = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "stream": True,
    }
    if want_nprobs:
        # nprobs is ignored for the streamed continuation tokens but llama.cpp
        # still reports the prompt-processing distribution in the final
        # "timings" pass; we take top-5 from the first non-stream probe below
        # if needed.
        body["nprobs"] = want_nprobs
    req = urllib.request.Request(
        base_url.rstrip("/") + "/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    ttft = None
    text = []
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if r.status != 200:
            raise RuntimeError(f"client got HTTP {r.status}, not 200")
        for line in r:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            tok = ev.get("content", "")
            if tok:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000.0
                text.append(tok)
    total = (time.perf_counter() - t0) * 1000.0
    if ttft is None:
        raise RuntimeError(
            f"client received no content tokens in {total/1000:.1f}s "
            f"(status was 200 — server-side completion with a dead client "
            f"path? issue #19)")
    return {
        "ttft_ms": ttft,
        "total_ms": total,
        "text": "".join(text),
    }


def first_token_top5(base_url, prompt, k=5):
    """Greedy 1-token request via OAI /v1/completions logprobs -> top-k list.

    (The native /completion endpoint ignores `nprobs` server-side.)
    """
    import math
    body = {
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": k,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        obj = json.load(r)
    try:
        entry = obj["choices"][0]["logprobs"]["content"][0]
    except (KeyError, IndexError):
        return None
    out = [{"tok": e["token"], "p": round(math.exp(e["logprob"]), 6)}
           for e in entry["top_logprobs"][:k]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("prompt_file")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--nprobs", type=int, default=5)
    args = ap.parse_args()

    with open(args.prompt_file) as f:
        prompt = f.read()

    res = one_request(args.base_url, prompt, args.max_tokens, want_nprobs=False)
    res["top5"] = first_token_top5(args.base_url, prompt, args.nprobs)
    json.dump(res, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
