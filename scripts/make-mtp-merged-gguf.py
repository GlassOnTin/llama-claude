#!/usr/bin/env python3
"""Embed a qwen4exp MTP (nextn) block into the base model GGUF.

The UD-Q4_K_XL base ships without the MTP block (block_count 48, max
blk.47); the MTP sidecar (e.g. Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf) is a
49-block mtp-only export. Upstream's draft-mtp serving path (PR #29761)
expects a base GGUF that CONTAINS the MTP block, served with
--spec-type draft-mtp and no --spec-draft-model.

Copies: all KV + tensors from the base shards in order, then the sidecar's
blk.48.* tensors. Base KV patches: block_count += nextn_predict_layers,
nextn_predict_layers added, compress_ratios replaced with the sidecar's
(verified to extend the base's). The sidecar's top-level tensors
(token_embd, output.*, output_hc_*) are skipped, the base already has its
own. Streaming copy in the style of gguf_new_metadata.py, no RAM hold.
"""
import glob
import os
import sys

import gguf

BASE_DIR = "/home/ian/models/Qwen3.8-Flash-Next/UD-Q4_K_XL"
SIDECAR = "/home/ian/models/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf"
OUT = "/home/ian/models/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-MTP-merged.gguf"


def field_int(reader: gguf.GGUFReader, key: str) -> int | None:
    f = reader.get_field(key)
    return int(f.contents()) if f else None


def field_list(reader: gguf.GGUFReader, key: str) -> list | None:
    f = reader.get_field(key)
    return list(f.contents()) if f else None


def main() -> int:
    base_shards = sorted(
        glob.glob(os.path.join(BASE_DIR, "*.gguf")),
        key=lambda p: int(p.rsplit("-of-00004", 1)[0].rsplit("-", 1)[-1]))
    if len(base_shards) != 4:
        print(f"[-] expected 4 base shards, found {len(base_shards)}", file=sys.stderr)
        return 1

    sidecar = gguf.GGUFReader(SIDECAR)
    side_cnt = field_int(sidecar, "qwen4exp.block_count")
    side_nextn = field_int(sidecar, "qwen4exp.nextn_predict_layers")
    side_ratios = field_list(sidecar, "qwen4exp.attention.compress_ratios")
    print(f"sidecar: block_count={side_cnt} nextn={side_nextn} ratios_len={len(side_ratios)}",
          file=sys.stderr)

    base1 = gguf.GGUFReader(base_shards[0])
    base_cnt = field_int(base1, "qwen4exp.block_count")
    base_ratios = field_list(base1, "qwen4exp.attention.compress_ratios")
    print(f"base: block_count={base_cnt} ratios_len={len(base_ratios)}", file=sys.stderr)

    if side_nextn != 1:
        print(f"[-] sidecar nextn_predict_layers == {side_nextn}, merge assumes 1", file=sys.stderr)
        return 1
    if side_cnt != base_cnt + side_nextn:
        print(f"[-] sidecar block_count {side_cnt} != base {base_cnt} + {side_nextn}", file=sys.stderr)
        return 1
    if side_ratios[:base_cnt] != base_ratios:
        print("[-] sidecar compress_ratios do not extend the base's, refusing", file=sys.stderr)
        return 1

    writer = gguf.GGUFWriter(OUT, "qwen4exp", endianess=base1.endianess)
    print(f"writing: {OUT}", file=sys.stderr)

    # KV from the base's metadata shard (shard 1 carries all metadata), in order
    for field in base1.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        if field.name.startswith("split."):
            continue # single-file output; split.* would make the loader look for shards
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = field.contents()
        if field.name == "qwen4exp.block_count":
            value = base_cnt + side_nextn
            print(f"KVP {field.name}: {base_cnt} -> {value}", file=sys.stderr)
        elif field.name == "qwen4exp.attention.compress_ratios":
            value = side_ratios
            print(f"KVP {field.name}: {len(base_ratios)} -> {len(side_ratios)} entries", file=sys.stderr)
        writer.add_key_value(field.name, value, val_type, sub_type=sub_type)
    writer.add_key_value("qwen4exp.nextn_predict_layers", side_nextn, gguf.GGUFValueType.UINT32)
    print(f"KVA qwen4exp.nextn_predict_layers = {side_nextn}", file=sys.stderr)

    total_bytes = 0
    seen: set[str] = set()
    for reader in [gguf.GGUFReader(p) for p in base_shards] + [sidecar]:
        for tensor in reader.tensors:
            if reader is sidecar and not tensor.name.startswith("blk.48."):
                continue # the base has its own top-level copies
            if tensor.name in seen:
                print(f"[-] duplicate tensor {tensor.name}", file=sys.stderr)
                return 1
            seen.add(tensor.name)
            total_bytes += tensor.n_bytes
            writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                                   tensor.data.nbytes, tensor.tensor_type)
    print(f"tensors: {len(seen)} in {total_bytes / 1e9:.1f} GB", file=sys.stderr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    done = 0
    for reader in [gguf.GGUFReader(p) for p in base_shards] + [sidecar]:
        for tensor in reader.tensors:
            if reader is sidecar and not tensor.name.startswith("blk.48."):
                continue
            writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
            done += 1
            if done % 200 == 0:
                print(f"  {done} tensors", file=sys.stderr)
    writer.close()
    print(f"[+] wrote {OUT}: {os.path.getsize(OUT) / 1e9:.1f} GB, {done} tensors", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())