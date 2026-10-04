#!/usr/bin/env python3
"""Deterministic benchmark for the local Qwen3.8-Flash-Next llama.cpp server.

Sends fixed-size prompts (unique filler per run so the prompt cache never
hits), measures prefill and decode from the server's `timings` block, and
prints a median summary. Same script runs before and after a llama.cpp
rebuild so results are directly comparable.

Usage: bench-qwen38-flash-next.py [--url URL] [--runs N] [--prompt-tokens N]
                                  [--gen-tokens N] [--label LABEL] [--out FILE]
                                  [--code-context DIR] [--corpus FILE]
                                  [--save-corpus FILE]

--code-context DIR assembles a prompt from the largest real source files in a
tree (workload-realistic alternative to filler text). --save-corpus snapshots
the assembled text so stock and rebuilt servers are benched on byte-identical
prompts; later runs pass --corpus FILE to reuse it.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.request

FILLER = (
    "The engineer reviewed the telemetry log for anomalies in the sensor "
    "readings. Temperature margins held steady through the test window. "
)

CODE_EXTS = (".kt", ".java", ".cpp", ".cc", ".c", ".h", ".md")
# Vendored upstream libraries and their machine-generated tables make poor
# agent-context stand-ins; skip them when walking the tree.
EXCLUDE_DIRS = {
    ".git", "build", "build-ffmpeg", "build-mosh", "build-proot", "node_modules",
    "scratch", "releases", "metadata", "sysroot", "glib", "libxml2",
    "virglrenderer", "freetype", "cairo", "pcre2", "libinput", "valgrind",
    "openssl", "zlib",
}
MAX_FILE_CHARS = 250_000


def is_generated(path: str) -> bool:
    name = os.path.basename(path).lower()
    return "generated" in name or name.endswith("_tables.h")


def build_corpus(root: str, min_chars: int) -> tuple[str, list[str]]:
    paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if fn.endswith(CODE_EXTS) and not is_generated(fn):
                p = os.path.join(dirpath, fn)
                try:
                    size = os.path.getsize(p)
                except OSError:
                    continue
                if size >= 2000:
                    paths.append((size, p))
    paths.sort(reverse=True)
    parts, used, total = [], [], 0
    for size, p in paths:
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                text = f.read(MAX_FILE_CHARS)
        except OSError:
            continue
        rel = os.path.relpath(p, root)
        parts.append(f"// ---- {rel} ----\n" + text)
        used.append(rel)
        total += len(text)
        if total >= min_chars:
            break
    return "\n\n".join(parts), used


def build_code_prompt(target_tokens: int, nonce: int, corpus: str) -> str:
    # Human-written code tokenises at roughly 3.5-4 chars/token; the server
    # reports the exact prompt_n, so the target only needs to be approximate.
    return (
        f"Bench request nonce {nonce}-{time.time_ns()}. "
        "The following is real source code from the Haven Android project, "
        "the way an agent would see it in context.\n"
        + corpus[: int(target_tokens * 3.8)]
        + "\n\nContinue the last file in the same style: write the next "
        "function. Output code only, no explanations."
    )


def build_prompt(target_tokens: int, nonce: int) -> str:
    # ~12 tokens per FILLER repetition; the nonce makes every prompt unique.
    reps = max(1, target_tokens // 12)
    return (
        f"Bench request nonce {nonce}-{time.time_ns()}. Ignore the following "
        "background text; it exists only to exercise prompt processing.\n"
        + FILLER * reps
        + "\nHow many repetitions of the word 'telemetry' appeared above? "
        "Reply with one short sentence."
    )


def run_once(url: str, prompt: str, gen_tokens: int) -> dict:
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen_tokens,
        "temperature": 0,
        "seed": 42,
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        payload = json.loads(resp.read())
    t = payload["timings"]
    return {
        "prompt_n": t["prompt_n"],
        "prompt_per_second": t["prompt_per_second"],
        "predicted_n": t["predicted_n"],
        "predicted_per_second": t["predicted_per_second"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--label", default="unlabelled")
    ap.add_argument("--out", default=None)
    ap.add_argument("--code-context", default=None,
                    help="source tree to assemble a code-context corpus from")
    ap.add_argument("--save-corpus", default=None,
                    help="save the assembled corpus text to this file")
    ap.add_argument("--corpus", default=None,
                    help="reuse a previously saved corpus file")
    args = ap.parse_args()

    corpus = None
    used_files = []
    if args.corpus:
        with open(args.corpus, encoding="utf-8") as f:
            corpus = f.read()
    elif args.code_context:
        # 40% headroom so the corpus still fills the prompt after the
        # per-token truncation.
        corpus, used_files = build_corpus(
            args.code_context, int(args.prompt_tokens * 3.8 * 1.4))
        print(f"corpus: {len(corpus)} chars from {len(used_files)} files", file=sys.stderr)
        for rel in used_files:
            print(f"  {rel}", file=sys.stderr)
        if args.save_corpus:
            with open(args.save_corpus, "w", encoding="utf-8") as f:
                f.write(corpus)
            print(f"saved corpus: {args.save_corpus}", file=sys.stderr)

    runs = []
    for i in range(args.runs):
        prompt = (build_code_prompt(args.prompt_tokens, i, corpus) if corpus
                  else build_prompt(args.prompt_tokens, i))
        t0 = time.time()
        r = run_once(args.url, prompt, args.gen_tokens)
        r["wall_s"] = round(time.time() - t0, 2)
        runs.append(r)
        print(f"run {i}: prefill {r['prompt_n']} tok @ {r['prompt_per_second']:.1f} tok/s | "
              f"decode {r['predicted_n']} tok @ {r['predicted_per_second']:.1f} tok/s | "
              f"wall {r['wall_s']}s", file=sys.stderr)

    summary = {
        "label": args.label,
        "server": args.url,
        "runs": args.runs,
        "prompt_tokens_target": args.prompt_tokens,
        "gen_tokens": args.gen_tokens,
        "prompt_kind": "code" if corpus else "filler",
        "corpus_sha256": hashlib.sha256(
            corpus[: int(args.prompt_tokens * 3.8)].encode()).hexdigest() if corpus else None,
        "corpus_files": len(used_files),
        "prefill_tok_s_median": round(statistics.median(r["prompt_per_second"] for r in runs), 2),
        "decode_tok_s_median": round(statistics.median(r["predicted_per_second"] for r in runs), 2),
        "prompt_n": runs[0]["prompt_n"],
        "predicted_n": [r["predicted_n"] for r in runs],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detail": runs,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"saved: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())