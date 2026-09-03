# Setting up Voicing-Convo-V2-35B-MOE on a new machine

Step-by-step for vLLM and SGLang. Follow the sections in order; the whole thing
is six steps and the only unusual one is section 4, which you should not skip.

> **Internal only.** Keep this package out of the published model repo. It is
> deployed next to the serving engine: baked into the serving image, checked into
> an internal repo, or copied onto the host. The model repository itself contains
> only weights, tokenizer, config, and chat template.

The model declares its own architecture:

```json
{ "architectures": ["VoicingConvoForCausalLM"], "model_type": "voicing_convo" }
```

Neither engine knows those names on its own, so nothing loads the model until
this package is on `PYTHONPATH`. That is the whole job of `voicing_runtime/`: at
interpreter startup it registers the architecture with whichever engine is
installed, and registers the `voicing` reasoning and tool-call parsers. No
engine source is modified and nothing is pip-installed.

---

## 1. Prerequisites

| | |
|---|---|
| GPU | 96 GB for `TP=1` at 64K context (bf16 weights are 69.3 GB). Smaller cards need `TP=2` or more. |
| Engine | SGLang `0.5.16`, or vLLM `0.28.0`. Both verified. On Blackwell use CUDA 12.8+ builds. |
| Python | 3.12, in the engine's own environment |
| Token | `HF_TOKEN` with read access (the model repo is private) |

## 2. Download the model

```bash
export HF_TOKEN=hf_...
export MODEL_DIR=/models/Voicing-Convo-V2-35B-MOE

HF_HUB_DISABLE_XET=1 hf download voicing-ai/Voicing-Convo-V2-35B-MOE \
  --local-dir "$MODEL_DIR"
```

`HF_HUB_DISABLE_XET=1` makes each file a plain HTTP stream that resumes from its
partial file if the transfer breaks. Re-run the same command to continue; it only
refetches what is missing.

Check the download before going further:

```bash
ls "$MODEL_DIR"/model-*.safetensors | wc -l    # must be 21
du -sh "$MODEL_DIR"                            # ~69 GB
```

## 3. Install this package

Put it anywhere the engine's Python can read. It is not pip-installed.

```bash
export VOICING_RUNTIME=/opt/voicing-serving-runtime

# clone it (private repo, same HF_TOKEN as the model)
git clone https://user:$HF_TOKEN@huggingface.co/voicing-ai/voicing-serving-runtime \
  "$VOICING_RUNTIME"
# (any username works for read-only clones; pushing needs your real HF username)

# or, without git:
hf download voicing-ai/voicing-serving-runtime --local-dir "$VOICING_RUNTIME"

# or, from a tarball you copied over:
mkdir -p "$VOICING_RUNTIME" && tar xzf voicing-serving-runtime.tar.gz \
  -C "$VOICING_RUNTIME" --strip-components=1

chmod +x "$VOICING_RUNTIME"/scripts/*.sh
```

In a Dockerfile, copy it into the serving image instead:

```dockerfile
COPY voicing-serving-runtime /opt/voicing-serving-runtime
ENV PYTHONPATH=/opt/voicing-serving-runtime/voicing_runtime
```

## 4. Pre-flight checks

Run these in the engine's Python environment, **before** starting a server. They
take seconds, load no weights, and catch a version mismatch between this package
and the installed engine — which is the one failure mode that otherwise shows up
as a confusing crash at launch.

```bash
cd "$VOICING_RUNTIME"

# architecture + config resolve on the installed engine; checkpoint keys match
python tests/test_model_registration.py "$MODEL_DIR"

# chat template: default system prompt and the serving behaviours (33 cases)
python tests/test_chat_template.py "$MODEL_DIR"

# parsers, for whichever engine you are deploying
python tests/test_sglang_parsers.py "$MODEL_DIR"    # SGLang
python tests/test_vllm_parsers.py "$MODEL_DIR"      # vLLM
```

Every one must end in `N/N passed`. A failure prints the exception with file and
line; a registration failure names the engine API that moved.

## 5. Start the server

### SGLang

```bash
MODEL_DIR="$MODEL_DIR" PORT=8000 TP=1 "$VOICING_RUNTIME/scripts/serve_sglang.sh"
```

That runs the stock entry point with `PYTHONPATH` set for you:

```bash
export PYTHONPATH="$VOICING_RUNTIME/voicing_runtime"

python3 -m sglang.launch_server \
  --model "$MODEL_DIR" \
  --served-model-name voicing-ai/Voicing-Convo-V2-35B-MOE \
  --host 0.0.0.0 --port 8000 \
  --tp 1 --dtype bfloat16 \
  --mem-fraction-static 0.90 \
  --attention-backend flashinfer \
  --sampling-backend pytorch \
  --reasoning-parser voicing \
  --tool-call-parser voicing \
  --context-length 65536 \
  --chunked-prefill-size 4096
```

Overrides: `PORT`, `TP`, `HOST`, `CONTEXT_LEN`, `MEM_FRACTION`, `ATTN_BACKEND`,
`SERVED_NAME`. Extra flags are appended, e.g.
`scripts/serve_sglang.sh --attention-backend triton`.

### vLLM

```bash
MODEL_DIR="$MODEL_DIR" PORT=8000 TP=1 "$VOICING_RUNTIME/scripts/serve_vllm.sh"
```

which runs:

```bash
export PYTHONPATH="$VOICING_RUNTIME/voicing_runtime"
PARSERS="$VOICING_RUNTIME/voicing_parsers/vllm"

vllm serve "$MODEL_DIR" \
  --served-model-name voicing-ai/Voicing-Convo-V2-35B-MOE \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 65536 \
  --max-num-seqs 256 \
  --reasoning-parser-plugin "$PARSERS/voicing_reasoning_parser.py" \
  --reasoning-parser voicing \
  --tool-parser-plugin "$PARSERS/voicing_tool_parser.py" \
  --enable-auto-tool-choice \
  --tool-call-parser voicing
```

Overrides: `PORT`, `TP`, `HOST`, `MAX_MODEL_LEN`, `GPU_MEM_UTIL`,
`MAX_NUM_SEQS`, `SERVED_NAME`.

**`--max-num-seqs 256` is required.** This architecture allots one Mamba cache
block per decode sequence; vLLM's default of 1024 does not fit alongside the
weights and fails CUDA-graph capture after the weights have loaded.

## 6. Verify the running server

```bash
curl -s http://127.0.0.1:8000/v1/models

python "$VOICING_RUNTIME/tests/smoke_live_api.py" http://127.0.0.1:8000/v1
```

The smoke test is engine-agnostic and must report `6/6 passed`:

1. default Voicing identity with no system message sent
2. a tool call decoded into `tool_calls`, `finish_reason: tool_calls`
3. streaming: reasoning deltas strictly before content, no tag leakage
4. a system message mid-conversation is accepted
5. a conversation that opens with an assistant turn is accepted
6. a tool result replayed with `arguments` as a JSON string

It takes a few minutes because thinking-mode generation is long.

---

## Kubernetes

Only three things differ from a stock deployment: the `PYTHONPATH` env var, and
the two parser names.

```yaml
containers:
  - name: sglang
    command: ["python3", "-m", "sglang.launch_server"]
    env:
      - name: PYTHONPATH
        value: /opt/voicing-serving-runtime/voicing_runtime
    args:
      - "--model"
      - "/mnt/models/llm/Voicing-Convo-V2-35B-MOE"
      - "--served-model-name"
      - "voicing-ai/Voicing-Convo-V2-35B-MOE"
      - "--host"
      - "0.0.0.0"
      - "--port"
      - "8000"
      - "--tp"
      - "1"
      - "--dtype"
      - "bfloat16"
      - "--mem-fraction-static"
      - "0.90"
      - "--attention-backend"
      - "triton"
      - "--sampling-backend"
      - "pytorch"
      - "--reasoning-parser"
      - "voicing"
      - "--tool-call-parser"
      - "voicing"
      - "--context-length"
      - "65536"
      - "--chunked-prefill-size"
      - "4096"
```

`/opt/voicing-serving-runtime` comes from the image (`COPY` it in) or from a
mounted volume. The model volume needs only the model repo; this package must
**not** be synced into it.

## Client notes

- **Give thinking room.** The model reasons at length even on short prompts.
  With thinking on, use `temperature=1.0, top_p=0.95, top_k=20,
  presence_penalty=1.5` and `max_tokens` of 16000 or more. A small budget ends
  the stream mid-reasoning and looks like an empty reply.
- **For short conversational turns, switch thinking off** with
  `extra_body={"chat_template_kwargs": {"enable_thinking": false}}`. Responses
  come back immediately.
- **The reasoning field is named differently per engine.** SGLang returns
  `reasoning_content`; vLLM 0.28 returns `reasoning`. Read both.
- **Tool calls arrive decoded.** The model emits XML on the wire; the parsers
  turn it into standard OpenAI `tool_calls`, so clients need no special handling.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Loads but is oddly slow, or fails later with an unknown `model_type` | `PYTHONPATH` not set. SGLang does **not** error on an unknown architecture, it silently falls back to its generic transformers backend | point `PYTHONPATH` at `<package>/voicing_runtime`, then re-run `tests/test_model_registration.py`, which asserts the fallback was not used |
| `Model architecture VoicingConvoForCausalLM ... not supported` (vLLM) | `PYTHONPATH` not set, or pointing at the package root | it must point at `<package>/voicing_runtime` |
| `argument --tool-call-parser: invalid choice: 'voicing'` | same as above, for SGLang | same |
| `[voicing] failed to register ...` plus a traceback at startup | this package does not match the installed engine version | run `tests/test_model_registration.py`; the traceback names the moved API |
| `max_num_seqs (1024) exceeds available Mamba cache blocks` | vLLM default | pass `--max-num-seqs 256` |
| Empty `content`, `finish_reason: length` | thinking used the whole budget | raise `max_tokens`, or disable thinking |
| Download dies mid-file, retries restart from zero | Xet chunk transfer | `HF_HUB_DISABLE_XET=1`, then re-run to resume |

## What is in here

See the layout table in [README.md](./README.md). In short:
`voicing_runtime/` does the registration, `voicing_parsers/` holds the
reasoning and tool-call parsers for each engine, `scripts/` has the launchers,
and `tests/` has the checks used in sections 4 and 6.
[`voicing_parsers/README.md`](./voicing_parsers/README.md) explains how each
engine resolves parsers and why SGLang needs the `PYTHONPATH` route rather
than a flag.

## Versioning

The parsers and the registration shim are written against the released engine
sources they were verified on: **SGLang 0.5.16** and **vLLM 0.28.0**. Both take
their engine-side pieces from stable entry points, but an engine upgrade can
still move an internal API. When you bump an engine, re-run the pre-flight
checks in section 4 first. They fail loudly and name what moved.
