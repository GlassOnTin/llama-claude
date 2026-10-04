#!/usr/bin/env python3
"""tty-chat - minimal streaming TTY chat for OpenAI-compatible endpoints, with image attach.

Usage:
  tty-chat.py [first prompt]     interactive REPL; attach an image with @/path/to/img.png
  CHAT_BASE=http://host:port/v1 CHAT_MODEL=name tty-chat.py

Reasoning tokens (reasoning_content deltas) go to stderr so stdout stays clean.
"""
import base64
import json
import mimetypes
import os
import sys
import urllib.request

BASE = os.environ.get("CHAT_BASE", "http://localhost:8765/v1")
MODEL = os.environ.get("CHAT_MODEL", "qwen3.8-flash-next")


def content_parts(text):
    """Split @path tokens out of the message into base64 image_url parts."""
    parts = []
    for tok in text.split():
        path = tok[1:]
        if tok.startswith("@") and os.path.isfile(path):
            mime = mimetypes.guess_type(path)[0] or "image/png"
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:{mime};base64,{b64}"}})
            text = text.replace(tok, f"[image: {os.path.basename(path)}]")
    parts.insert(0, {"type": "text", "text": text})
    return parts


def stream_chat(messages):
    body = json.dumps({"model": MODEL, "messages": messages, "stream": True}).encode()
    req = urllib.request.Request(f"{BASE}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    out = ""
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            delta = json.loads(payload)["choices"][0].get("delta", {})
            if think := delta.get("reasoning_content"):
                sys.stderr.write("\x1b[2m" + think + "\x1b[0m")
                sys.stderr.flush()
            if c := delta.get("content"):
                sys.stdout.write(c)
                sys.stdout.flush()
                out += c
    print()
    return out


def main():
    history = []
    pending = sys.argv[1:]
    while True:
        if pending:
            msg, pending = pending[0], pending[1:]
        else:
            try:
                msg = input("you> ")
            except (EOFError, KeyboardInterrupt):
                break
        if msg.strip() in ("/q", "/quit", "/exit"):
            break
        if msg.strip() == "/reset":
            history.clear()
            continue
        if not msg.strip():
            continue
        history.append({"role": "user", "content": content_parts(msg)})
        sys.stdout.write("model> ")
        reply = stream_chat(history)
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
