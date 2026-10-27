# Runtime, weights and inference settings

Releasing the raw votes makes every table in `results/` re-derivable without running a model.
It does not make a model reproduce the same votes: sampling at temperature 1.0 is reproducible
only within one build, and `results/ikili_tekrar_kontrolu.csv` measures how far one
configuration moves between two builds of the same library. This file records what would be
needed to attempt a re-run, at the level of detail that identifies each component exactly.

## Local model weights

Weights were pulled from the Ollama library registry as single GGUF blobs, not from a
Hugging Face repository. For tag `<name>:<size>` the manifest at
`https://registry.ollama.ai/v2/library/<name>/manifests/<size>` names one layer of media type
`application/vnd.ollama.image.model`; that layer's digest is the SHA-256 of the GGUF file and
is what the table below records. The blob is retrieved from
`https://registry.ollama.ai/v2/library/<name>/blobs/<digest>`.

| Paper name | Registry tag | GGUF SHA-256 | Bytes |
| --- | --- | --- | --- |
| Gemma 2 2B | `gemma2:2b` | `7462734796d67c40ecec2ca98eddf970e171dbb6b370e43fd633ee75b69abe1b` | 1,629,509,152 |
| Llama 3.2 3B | `llama3.2:3b` | `dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff` | 2,019,377,376 |
| Qwen2.5 3B | `qwen2.5:3b` | `5ee4f07cdb9beadbbb293e85803c569b01bd37ed059d2715faa7bb405f31caa6` | 1,929,903,008 |
| Mistral 7B | `mistral:7b` | `f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f` | 4,372,811,712 |
| Qwen2.5 7B | `qwen2.5:7b` | `2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730` | 4,683,073,952 |
| Llama 3.1 8B | `llama3.1:8b` | `667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29` | 4,920,738,944 |

The digests above were read from the registry manifests. For `gemma2:2b` the digest was also
verified against the local file by hashing it: same SHA-256, same byte count. The GGUF header
of that file reports `general.architecture = gemma2`, `general.basename = gemma-2.0`,
`general.size_label = 2B`, `general.file_type = 2` (Q4_0-family quantisation),
`general.quantization_version = 2`, GGUF format version 3, 288 tensors, 34 key-value pairs.

Registry tags are mutable in principle: a future `gemma2:2b` could resolve to a different
blob. The SHA-256 is the stable identifier, and a re-run should check it.

## Inference runtime

Inference ran through `llama-cpp-python` **0.3.36** on CPU, using the prebuilt shared
libraries that ship with the wheel rather than a locally compiled llama.cpp. The loaded
`libggml` carries soversion **0.25.3**, and `llama_print_system_info()` on the machine used
reports:

```
CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | DOTPROD = 1 | LLAMAFILE = 1 | ACCELERATE = 1 | REPACK = 1
```

The host was Apple silicon (arm64, macOS), 8 cores, with no GPU offload. The wheel's binaries
do not expose an upstream llama.cpp git commit through the C API, so the pinned
`llama-cpp-python` version together with the ggml soversion and these feature flags is the
most precise build identity available from inside the runtime; a re-run on a different CPU
will differ in the flags even at the same library version.

## Local sampling parameters

From `scripts/run_local_v4r.py` (identical to `run_local_v4.py` apart from the cluster and
output paths):

| Setting | Value |
| --- | --- |
| Context | `n_ctx = 3072` |
| GPU layers | `n_gpu_layers = 0` |
| Threads | `n_threads = 6` |
| Chat template | `chat_format` per model (`gemma` for Gemma 2 2B) |
| RNG seed | `seed = 1234` unless overridden by argv[7] |
| Temperature | 1.0 |
| Top-p | 0.95 |
| Max tokens | 32 |
| Output constraint | `response_format={"type": "json_object", "schema": SCHEMA}`, the schema being a single enumerated field over the 14 labels |
| Votes per cluster | `K = 10` |

Sampling is deliberately stochastic: the ten votes per cluster are the probability source, so
temperature 1.0 is part of the method rather than a loose default.

## API models

Two hosted models were called with a forced tool call rather than a JSON schema
(`scripts/api_run.py`): `tool_choice={"type": "tool", "name": "annotate"}`, temperature 1.0,
`max_tokens = 64`, `K = 10`. The exact model identifiers are `claude-sonnet-4-5-20250929` and
`claude-haiku-4-5-20251001`. Hosted endpoints are not pinned by a digest and may change
behaviour behind a fixed identifier; the released votes are the record of what they returned.

## Analysis environment

Clustering, conformal evaluation and figures ran under Python 3.12.15 with scanpy 1.12.4,
anndata 0.13.4, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.0.6, python-igraph
1.0.0 and leidenalg 0.12.0. Leiden was called as
`flavor="igraph", n_iterations=2, directed=False`. The conformal evaluation in
`scripts/recompute.py` uses 2,000 random calibration/test splits with seed 11; the multi-seed
replication uses 800 repetitions at seeds 1234, 7 and 99.
