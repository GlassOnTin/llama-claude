{
  "server": "mimo-mxfp4-ot-6-8-8-ub512",
  "wall_s": 108.0,
  "timings": {
    "cache_n": 0,
    "prompt_n": 13573,
    "prompt_ms": 93862.427,
    "prompt_per_token_ms": 6.915378103588005,
    "prompt_per_second": 144.6052529624021,
    "predicted_n": 256,
    "predicted_ms": 13989.664,
    "predicted_per_token_ms": 54.86142745098039,
    "predicted_per_second": 18.227742996543732
  },
  "completion": "```kotlin\n            description = \"Write a local file to a path on any connected backend (local, SSH, SMB, rclone). Source must live under Haven's app cache (context.cacheDir) \u2014 the agent has no other writable surface, so this constraint blocks reads of arbitrary device files via the upload destination. Currently uses small-file semantics (loads the source into memory); streaming variants ship in a later #126 stage. Replaces upload_file_to_sftp; that still works as a deprecated alias.\",\n            inputSchema ="
}
