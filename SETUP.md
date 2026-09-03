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
| Model access | `HF_TOKEN` with read access to the private model repo |
| Runtime access | credentials for the private GitHub repo (token, SSH deploy key, or a copied checkout) |

## 2. Download the model

```bash
export HF_TOKEN=hf_...
export MODEL_DIR=/models/Voicing-Convo-V2-35B-MOE
export HF_HUB_DISABLE_XET=1

# Expect several failures on a 69 GB transfer. Each pass resumes and only
# refetches what is missing, so loop until all 21 shards are present.
for i in $(seq 1 40); do
  [ "$(ls "$MODEL_DIR"/model-*.safetensors 2>/dev/null | wc -l)" -eq 21 ] && break
  hf download voicing-ai/Voicing-Convo-V2-35B-MOE --local-dir "$MODEL_DIR" && break
  echo "pass $i incomplete, resuming..."; sleep 3
done
```

`HF_HUB_DISABLE_XET=1` makes each file a plain HTTP stream that resumes from its
partial file when a transfer breaks. **Do not skip the loop.** A single run of
`hf download` frequently ends with
`RemoteProtocolError: peer closed connection without sending complete message
body` partway through, leaving a directory that looks nearly complete. On a
recent clean install this needed six passes.

Verify before going further, and treat the shard count as the gate: the
directory size alone looks plausible while shards are still missing.

```bash
ls "$MODEL_DIR"/model-*.safetensors | wc -l    # must be 21, not "about 21"
du -sh "$MODEL_DIR"                            # ~69 GB
```

`voicing-check` in step 3 checks the shard count for you against the model
index, and `voicing-check --verify-weights "$MODEL_DIR"` additionally sha256s
every shard against the Hub's published checksums when you want certainty that
the bytes are right and not merely present.

### If a few shards never finish

Some hosts resolve the Hub to a CDN edge that drops long transfers. The symptom
is distinctive: the loop above keeps running but the same one or two shards stay
missing, and each pass adds only a few MB. Check which edge you are getting:

```bash
curl -sS -I -L -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/voicing-ai/Voicing-Convo-V2-35B-MOE/resolve/main/model-00018-of-00021.safetensors" \
  | grep -iE '^location|^x-amz-cf-pop'
```

`us.gcp.cdn.hf.co` has been unreliable from some machines; `us.aws.cdn.hf.co`
has not. You cannot choose the edge, but `curl` with byte-range resume copes
where the client library stalls. Fetch the stragglers directly:

```bash
BASE=https://huggingface.co/voicing-ai/Voicing-Convo-V2-35B-MOE/resolve/main
for F in model-00018-of-00021.safetensors model-00019-of-00021.safetensors; do
  for a in $(seq 1 60); do
    curl -sS -L -C - --retry 5 --retry-all-errors -H "Authorization: Bearer $HF_TOKEN" \
      -o "$MODEL_DIR/$F" "$BASE/$F" && break
  done
done
```

Then re-run `voicing-check --verify-weights "$MODEL_DIR"` to confirm the bytes
match the Hub before serving.

## 3. Install the serving runtime

Install it **after** the engine, into the same environment. It has no
dependencies of its own and adapts to whichever engine it finds.

```bash
pip install vtext-editor

# or from GitHub
pip install "git+https://github.com/VoicingAI/voicing-convo-llm-runtime.git"
```

The repository is private, so the machine needs credentials for it. Either
configure `git` with a GitHub token or SSH key, or install from a checkout you
have already copied over:

```bash
# with a GitHub personal access token
pip install "git+https://$GITHUB_TOKEN@github.com/VoicingAI/voicing-convo-llm-runtime.git"

# over SSH, if the machine has a deploy key
pip install "git+ssh://git@github.com/VoicingAI/voicing-convo-llm-runtime.git"

# from a local checkout or an unpacked tarball
pip install /path/to/voicing-convo-llm-runtime
```

Pin a release when you want reproducible images: append `@v1.0.0` or `@<commit>`
to any of the git URLs.

> **`uv`-created virtualenvs have no `pip`.** If `python -m pip` reports
> `No module named pip`, use `uv pip install ...` with the same argument. vLLM
> environments built with `uv venv` are the common case.

That is the whole setup. Installing the package registers two plugin entry
points, `vllm.general_plugins` and `sglang.srt.plugins`, which each engine loads
by itself in the launcher, the engine core and every worker process. There is no
`PYTHONPATH` to set, no plugin file paths to pass, and no engine source to edit.

Confirm it took:

```bash
voicing-check "$MODEL_DIR"
```

```
vtext-editor 1.0.0
  registered for: transformers, sglang
  sglang: architecture=ok reasoning-parser=ok tool-call-parser=ok
  model:  /models/Voicing-Convo-V2-35B-MOE -> ['VoicingConvoForCausalLM'] / voicing_convo ok

OK
```

## 4. Pre-flight checks

`voicing-check` in step 3 covers the common case. The full suites go further and
are worth running once per machine and after any engine upgrade. They take
seconds and load no weights. Run them from a checkout of this repo, in the
engine's Python environment:

```bash
git clone https://github.com/VoicingAI/voicing-convo-llm-runtime.git && cd voicing-convo-llm-runtime

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

```bash
voicing-serve sglang --model "$MODEL_DIR" --port 8000
voicing-serve vllm   --model "$MODEL_DIR" --port 8000
```

`voicing-serve` builds a sensible command for the engine and execs it. Add
`--dry-run` to print the command instead of running it. Common options:
`--tp`, `--host`, `--port`, `--max-model-len`, `--gpu-memory-utilization`,
`--served-model-name`. Anything else is passed through to the engine, and a flag
you supply overrides the default of the same name:

```bash
voicing-serve sglang --model "$MODEL_DIR" --tp 2 --attention-backend triton
voicing-serve vllm   --model "$MODEL_DIR" --tp 2 --enable-prefix-caching
```

### Or call the engines directly

Nothing about `voicing-serve` is required; the package works with the stock
commands because the parsers and the architecture are already registered.

```bash
python -m sglang.launch_server \
  --model "$MODEL_DIR" \
  --served-model-name voicing-ai/Voicing-Convo-V2-35B-MOE \
  --host 0.0.0.0 --port 8000 \
  --tp 1 --dtype bfloat16 \
  --mem-fraction-static 0.90 \
  --attention-backend flashinfer --sampling-backend pytorch \
  --reasoning-parser voicing --tool-call-parser voicing \
  --context-length 65536 --chunked-prefill-size 4096
```

```bash
vllm serve "$MODEL_DIR" \
  --served-model-name voicing-ai/Voicing-Convo-V2-35B-MOE \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 65536 --max-num-seqs 256 \
  --reasoning-parser voicing \
  --enable-auto-tool-choice --tool-call-parser voicing
```

**`--max-num-seqs 256` is required for vLLM.** This architecture allots one
Mamba cache block per decode sequence; the default of 1024 does not fit
alongside the weights and fails CUDA-graph capture after loading.

## 6. Verify the running server

```bash
curl -s http://127.0.0.1:8000/v1/models

python tests/smoke_live_api.py http://127.0.0.1:8000/v1
```

The smoke test is engine-agnostic and must report `7/7 passed`:

1. default Voicing identity with no system message sent
2. a tool call decoded into `tool_calls`, `finish_reason: tool_calls`
3. streaming: content streams cleanly with thinking off; with thinking on,
   reasoning comes first and think tags never leak
4. a system message mid-conversation is accepted
5. a conversation that opens with an assistant turn is accepted
6. a tool result replayed with `arguments` as a JSON string

It takes a few minutes because thinking-mode generation is long.

---

## Kubernetes

Install the runtime into the serving image, then use the stock command. No env
vars, no plugin flags, nothing mounted beside the model.

```dockerfile
FROM lmsysorg/sglang:v0.5.16
RUN pip install vtext-editor
```

```yaml
containers:
  - name: sglang
    command: ["python3", "-m", "sglang.launch_server"]
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

Compared with a stock deployment, the only differences are the two parser values
and the one line in the Dockerfile. The model volume carries only the model.

## Vendor names in engine logs

Both engines log a few of their own internal identifiers that carry the upstream
vendor's name. They are internal to vLLM and SGLang and cannot be renamed
without patching those projects, so the runtime handles them two ways.

**SGLang is fixed properly.** It inspects the chat template to suggest a parser
and logs what it found. The runtime registers a detection rule for this model,
so it reports `reasoning_parser=voicing, tool_call_parser=voicing` and
`--reasoning-parser auto` resolves correctly.

**vLLM is redacted at the logging layer.** Its kernel module filenames, warm-up
message, and compiled custom-op names are rewritten on the way out by a
`logging.Filter` the runtime installs. Nothing in vLLM is modified. The
substitution keeps the meaningful part so lines stay traceable:

| vLLM writes | you see |
|---|---|
| `[qwen_gdn_linear_attn.py:158]` | `[gdn_linear_attn.py:158]` |
| `Warming up Qwen Triton kernels for model_type=qwen3_5_moe_text` | `Warming up GDN Triton kernels for model_type=voicing_convo` |
| `vllm::qwen_gdn_attention_core` | `vllm::gdn_attention_core` |

Verified: a full startup log from either engine contains zero matches for the
vendor name.

Set `VOICING_REDACT_LOGS=0` to disable the rewriting, for instance when
reporting a bug upstream and you want the engine's own identifiers verbatim.

> This is cosmetic and covers routine operational logs. The names still exist in
> the installed engine's source, in `pip show`, and in tracebacks from a crash
> inside those modules. It is not a security boundary.

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
| `argument --tool-call-parser: invalid choice: 'voicing'` | the runtime is not installed in the environment the engine runs in | `voicing-check`; install with the engine's own installer (see the `uv` note in step 3) |
| Loads but is oddly slow, or fails later with an unknown `model_type` | same, on SGLang. It does **not** error on an unknown architecture, it silently falls back to its generic transformers backend | `voicing-check`, then `tests/test_model_registration.py`, which asserts the fallback was not used |
| `Model architecture VoicingConvoForCausalLM ... not supported` (vLLM) | same | `voicing-check` |
| `No module named pip` when installing | the venv was created by `uv` | `uv pip install ...` |
| `voicing-check` reports the wrong engine | you have more than one venv active, or `which python` is not the engine's | activate one venv in a clean shell and re-check |
| `max_num_seqs (1024) exceeds available Mamba cache blocks` | vLLM default | `--max-num-seqs 256` (already set by `voicing-serve`) |
| `error: port 8000 is already in use` | something else owns the port; on managed GPU hosts a portal or reverse proxy often does | `--port 18000`, or stop the listener. `voicing-serve` checks this before launching |
| SGLang: `Initialization failed. warmup error ... AssertionError: res=<Response [401]>` | the port was taken, so SGLang's own warmup request reached the other service instead of itself | same as above; the message is misleading, the cause is the port |
| Empty `content`, `finish_reason: length` | thinking used the whole budget | raise `max_tokens`, or disable thinking |
| `RemoteProtocolError: peer closed connection...` while downloading | normal on a 69 GB transfer | expected; the loop in step 2 resumes and refetches only what is missing |
| Download looked finished but the model will not load | fewer than 21 shards; one `hf download` run often stops early | `voicing-check "$MODEL_DIR"` names the missing shards; re-run the step 2 loop |
| The same one or two shards never complete, each pass gaining only a few MB | the Hub resolved to a CDN edge that drops long transfers | use the `curl -C -` fallback in step 2 |

## What is in here

```
src/voicing_runtime/
  register.py        the entry point both engines call
  model.py           architecture + config registration, per engine
  cli.py             voicing-serve, voicing-check
  parsers/           reasoning and tool-call parsers for both engines
tests/               the suites used in sections 4 and 6
docs/PARSERS.md      how each engine resolves parsers, and why
```

## Versioning

The parsers and the registration code are written against the released engine
sources they were verified on: **SGLang 0.5.16** and **vLLM 0.28.0**. Both take
their engine-side pieces from stable entry points, but an engine upgrade can
still move an internal API. After bumping an engine, run `voicing-check` and the
section 4 suites before sending traffic. They fail loudly and name what moved.
