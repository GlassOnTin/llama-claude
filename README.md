# llama-claude: Dual RTX 5090 Local AI Engine for Claude Code

High-performance local AI inference engine powering **Anthropic Claude Code** using **Qwen3.8-27B (Vision + MTP)** distributed across dual 32GB NVIDIA GeForce RTX 5090 GPUs (64 GB total VRAM).

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Host["Ubuntu Linux Workstation (64GB RAM)"]
        CC["Anthropic Claude Code CLI\n(lclaude)"]
        LS["llama-server (:8090)\nFlashAttention-3 + MTP + Jinja ChatML"]
    end

    subgraph GPU0["GPU 0: Internal RTX 5090 (32GB VRAM)"]
        L0["Layers 0 – 29\n(~13.3 GB)"]
        K0["KV Cache Partition\n(~2.35 GB @ 262k Q8)"]
        UI["Desktop / Xorg\n(~1.5 GB)"]
    end

    subgraph GPU1["GPU 1: AORUS AI-BOX RTX 5090 (32GB VRAM)"]
        L1["Layers 30 – 63\n(~13.3 GB)"]
        K1["KV Cache Partition\n(~2.35 GB @ 262k Q8)"]
        MTP["MTP Speculative Head\n(~2.95 GB)"]
        VIS["Vision mmproj\n(~0.87 GB)"]
    end

    CC -->|Anthropic Messages API\n/v1/messages| LS
    LS -->|PCIe Gen 5| GPU0
    LS -->|Thunderbolt 4 / USB4 (40 Gb/s)\nPipelined Layer Handoff| GPU1
```

---

## Key Highlights

| Feature | Implementation | Benefit |
| :--- | :--- | :--- |
| **Model** | Qwen3.8-27B (Q8_0, 26.9B params) | Bit-exact quality to FP16 (<0.001 ppl loss) |
| **Multimodal Vision** | `mmproj-Qwen3.8-27B-BF16.gguf` | Image understanding & screenshot analysis in Claude Code |
| **Context Window** | **262,144 tokens (256k)** | Full repository awareness |
| **KV Cache Footprint** | `Q8_0` (~4.7 GB total at 262k) | Only 16 of 64 layers carry KV cache due to Hybrid Attention |
| **Speculative Decoding** | Multi-Token Prediction (MTP 3-draft) | **1.6× – 2.4× generation speedup** (75–100+ tok/s) |
| **Multi-GPU Parallelism**| `--split-mode layer --tensor-split 30,34` | Eliminates TB4 PCIe bus bottlenecks; balances desktop VRAM |
| **Reasoning Stream** | `--reasoning on --reasoning-format deepseek` | Converts `<think>` tags into native Anthropic Thinking blocks |
| **Compaction Headroom** | `CLAUDE_CODE_MAX_CONTEXT_TOKENS=160000` | 60k token buffer guaranteeing clean auto/manual `/compact` |

---

## 1. Multi-GPU Engineering on Thunderbolt 4

### The Thunderbolt Bandwidth Challenge
The second RTX 5090 is housed in an external **GIGABYTE AORUS RTX5090 AI-BOX** connected via Thunderbolt 4 / USB4 (40 Gb/s ≈ 3.2 GB/s usable).
* **Row Splitting (`-sm row` / Tensor Parallelism)**: Requires all-reduce synchronization across every attention head in all 64 layers (64 syncs per token). Over a 40 Gb/s TB4 cable, this saturates the bus and causes severe latency.
* **Layer Splitting (`-sm layer` / Pipeline Parallelism)**: Layers 0–29 run on GPU 0; layers 30–63 run on GPU 1. Cross-GPU transmission occurs **only once per token** (a single ~200 KB hidden-state tensor between layers 29 and 30). This keeps TB4 PCIe utilization under 2%.

### Asymmetric VRAM Balancing (`--tensor-split 30,34`)
* **GPU 0 (Internal)**: Drives the desktop display server and background apps (~3.2 GB allocated).
* **GPU 1 (External)**: 100% dedicated compute.
* Setting `--tensor-split 30,34` assigns 30 layers to GPU 0 and 34 layers to GPU 1, perfectly balancing free memory on both cards to ~28.5 GB headroom each.

---

## 2. Host System Optimization (Ubuntu Linux)

Thunderbolt eGPUs on Linux require specific kernel and PCIe settings to prevent GSP firmware timeouts (`Xid 175` / `Sysmembar` drops) caused by Active State Power Management (ASPM):

### Automated Host Setup
Run the included host configuration script:
```bash
cd host-config
sudo ./setup-host.sh
```

### Manual Configuration
1. **PCIe ASPM Performance Policy**: Prevents the USB4 PCIe controller from dropping into L1/L1.2 sleep:
   ```bash
   echo performance | sudo tee /sys/module/pcie_aspm/parameters/policy
   ```
2. **GRUB Kernel Arguments (`/etc/default/grub.d/99-nvidia-egpu.cfg`)**:
   ```bash
   GRUB_CMDLINE_LINUX_DEFAULT="$GRUB_CMDLINE_LINUX_DEFAULT pcie_aspm.policy=performance iommu=pt"
   ```
   * `iommu=pt`: Bypasses DMA remapping overhead for PCIe devices.
3. **Driver Downshift Lock (`/etc/modprobe.d/nvidia-egpu.conf`)**:
   ```bash
   options nvidia NVreg_RegistryDwords="PowerMizerEnable=0x1;PerfLevelSrc=0x2222;RMDisablePcieGenSpeedChange=1"
   ```
4. **Hotplug udev Automation (`/etc/udev/rules.d/99-nvidia-hotplug.rules`)**:
   Ensures `/dev/nvidia1` is created and `nvidia-persistenced` is notified whenever the AI-BOX is attached:
   ```udev
   ACTION=="add|bind", SUBSYSTEM=="pci", DRIVERS=="nvidia", ATTR{vendor}=="0x10de", RUN+="/sbin/ub-device-create"
   ACTION=="add|bind", SUBSYSTEM=="pci", DRIVERS=="nvidia", ATTR{vendor}=="0x10de", RUN+="/bin/systemctl try-restart nvidia-persistenced.service"
   ```

---

## 3. Qwen3.8 Hybrid Attention & 262k Context Math

Qwen3.8 utilizes a hybrid architecture pairing **Gated DeltaNet linear attention** with standard full self-attention:
* **Only 16 of the 64 transformer layers maintain a standard KV cache**.
* **KV Cache VRAM at 262k Tokens**:
  * **FP16 KV Cache**: ~9.4 GB total (~4.7 GB/GPU)
  * **Q8_0 KV Cache** (`--cache-type-k q8_0 --cache-type-v q8_0`): **~4.7 GB total (~2.35 GB/GPU)**
  * **Q4_0 KV Cache**: ~2.4 GB total (~1.2 GB/GPU)

With 27.1 GB total model weights and a 4.7 GB Q8_0 KV cache, the entire 256k context model consumes only **~32 GB of your 64 GB total VRAM**, leaving >30 GB free.

---

## 4. Multi-Token Prediction (MTP) Speculative Decoding

Qwen3.8 incorporates native Multi-Token Prediction draft layers. In this setup:
* Base Model: `Qwen3.8-27B-Q8_0.gguf` (~26.6 GB)
* MTP Draft Model: `mtp-Qwen3.8-27B-Q8_0.gguf` (~2.95 GB, 18 draft tensors)
* Converted from official verified upstream commit: `PRIMARY=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`

The MTP head speculatively generates 3 future tokens per step (`--spec-draft-n-max 3`), validated by the base model in a single batch pass, boosting generation throughput from ~50 tok/s to **75–100+ tok/s**.

---

## 5. Claude Code Integration & Chat Template

### Chat Template (`templates/qwen3.8-claude.jinja`)
Standard Qwen templates reject multi-turn system prompts with an `HTTP 500: System message must be at the beginning`. The custom template in `templates/` fixes this by:
* Allowing system messages at any turn index without throwing exceptions.
* Correctly formatting tool calls and tool responses for Claude Code.

### Native Thinking Stream (`--reasoning on --reasoning-format deepseek`)
Instead of dumping raw `<think>...</think>` text into conversation history, `llama-server` intercepts reasoning tokens and streams them as native **Anthropic Thinking Blocks** (`type: "thinking"` / `type: "thinking_delta"`). Claude Code displays its native collapsible thinking spinner.

---

## Quickstart

### Option A: One-Step Automated Installation
```bash
./install.sh
```

### Option B: Manual Setup

#### 1. Download Model & MTP Weights
```bash
./download.sh
```

#### 2. Start the Dual-GPU Server
Run directly in a terminal:
```bash
./serve.sh
```

Or install and start as a **systemd service**:
```bash
sudo cp systemd/llama-qwen.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-qwen
```

#### 3. Add Shell Integration to `~/.bashrc`
```bash
[ -f "$HOME/Code/llama-claude/bash_module.sh" ] && . "$HOME/Code/llama-claude/bash_module.sh"
```

#### 4. Launch Claude Code
```bash
lclaude
```

---

## Benchmarks (`llama-bench`)

```text
CUDA Devices: 2 (RTX 5090 32GB x 2 = 64,213 MiB VRAM)
Target Arch:  Blackwell sm_120 (CUDA 13.3)
Model:        Qwen3.8-27B Q8_0 (26.9B params)

Prompt Processing (pp64):  1,528 – 1,638 tokens/sec
Generation Baseline (tg16): 48.5 – 50.0 tokens/sec
Generation with MTP (3-tok): 75.0 – 105.0 tokens/sec
```
