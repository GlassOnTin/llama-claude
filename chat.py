#!/usr/bin/env python3
"""
Interactive Streaming Chat CLI for local llama.cpp server (Triple RTX 5090 Qwen3.8-Flash-Next)
Connects directly to http://127.0.0.1:8090
"""

import sys
import json
import time
import readline
import urllib.request
import urllib.error

ENDPOINT = "http://127.0.0.1:8090/v1/chat/completions"

CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[92m"
MAGENTA = "\033[95m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def is_server_busy():
    try:
        req = urllib.request.Request("http://127.0.0.1:8090/slots", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            slots = json.loads(resp.read().decode("utf-8"))
            if slots and slots[0].get("is_processing", False):
                return True
    except Exception:
        pass
    return False

def stream_chat(messages, system_prompt=None):
    payload = {
        "model": "qwen3.8-flash-next",
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
        "top_p": 0.90,
        "min_p": 0.05,
        "max_tokens": 4096
    }
    
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    in_think = False
    header_printed = False
    assistant_content = ""
    reasoning_content = ""
    
    # Check if server is currently occupied
    if is_server_busy():
        sys.stdout.write(f"{DIM}{YELLOW}[Waiting for active GPU slot to free...]{RESET}\r")
        sys.stdout.flush()
    
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                
                chunk = json.loads(data_str)
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                
                # First token arrived - print header and clear any wait messages
                if not header_printed:
                    sys.stdout.write("\r\033[K")  # Clear waiting line
                    header_printed = True
                
                # Reasoning / thinking delta
                r_delta = delta.get("reasoning_content")
                if r_delta:
                    if not in_think:
                        sys.stdout.write(f"{DIM}{CYAN}Thought: {RESET}{DIM}")
                        in_think = True
                    sys.stdout.write(r_delta)
                    sys.stdout.flush()
                    reasoning_content += r_delta
                
                # Visible content delta
                c_delta = delta.get("content")
                if c_delta:
                    if in_think:
                        sys.stdout.write(f"{RESET}\n\n{BOLD}Qwen > {RESET}")
                        in_think = False
                    elif not in_think and not assistant_content:
                        sys.stdout.write(f"\n{BOLD}Qwen > {RESET}")
                    sys.stdout.write(c_delta)
                    sys.stdout.flush()
                    assistant_content += c_delta

        if in_think:
            sys.stdout.write(f"{RESET}\n")
        sys.stdout.write("\n")
        sys.stdout.flush()
        return assistant_content
    except urllib.error.URLError as e:
        sys.stdout.write("\r\033[K")
        print(f"\nError connecting to llama-server on :8090: {e}")
        return None

def main():
    print(f"{BOLD}=== Qwen3.8-Flash-Next Direct Terminal Chat ==={RESET}")
    print(f"Connected to {CYAN}{ENDPOINT}{RESET}")
    print(f"{DIM}Commands: '/clear' (reset memory), 'exit' or Ctrl+D (quit){RESET}\n")
    
    messages = []
    
    # Check if arguments provided for one-shot query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        messages.append({"role": "user", "content": query})
        stream_chat(messages)
        return

    while True:
        try:
            user_input = input(f"{BOLD}{GREEN}You > {RESET}").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", ":q"):
                break
            if user_input == "/clear":
                messages = []
                print(f"{DIM}[Conversation context cleared]{RESET}\n")
                continue
            
            messages.append({"role": "user", "content": user_input})
            reply = stream_chat(messages)
            if reply:
                messages.append({"role": "assistant", "content": reply})
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
