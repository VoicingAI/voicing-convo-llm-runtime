# voicing-convo-llm-runtime

Serves **Voicing-Convo-V2-35B-MOE** on **vLLM** or **SGLang**. Install it after
the engine and the stock commands just work.

```bash
pip install "git+https://github.com/VoicingAI/voicing-convo-llm-runtime.git"

voicing-serve sglang --model /models/Voicing-Convo-V2-35B-MOE --port 8000
voicing-serve vllm   --model /models/Voicing-Convo-V2-35B-MOE --port 8000
```

**Full guide: [SETUP.md](./SETUP.md).**

> **Internal.** Deliberately kept out of the published model repository, which
> carries only weights, tokenizer, config, and chat template.

## What installing it does

The model declares its own architecture:

```json
{ "architectures": ["VoicingConvoForCausalLM"], "model_type": "voicing_convo" }
```

Neither engine knows that name, and neither knows how to read the model's
reasoning or its XML tool calls. This package declares two plugin entry points,
`vllm.general_plugins` and `sglang.srt.plugins`, which each engine loads by
itself in the launcher, the engine core, and every worker process. That
registers:

- the **architecture** and **config class**, with transformers, SGLang, and vLLM
- the **`voicing` reasoning parser**, so thinking comes back as a separate field
- the **`voicing` tool-call parser**, so tool calls arrive as normal OpenAI `tool_calls`

No `PYTHONPATH`, no plugin file paths, no engine source changes, and no
dependencies of its own.

## Commands

| | |
|---|---|
| `voicing-serve sglang \| vllm --model DIR` | build and run the engine command (`--dry-run` to print it) |
| `voicing-check [DIR]` | confirm registration against the installed engine |

## Layout

```
src/voicing_runtime/
  register.py        the entry point both engines call
  model.py           architecture + config registration, per engine
  cli.py             voicing-serve, voicing-check
  parsers/           reasoning and tool-call parsers for both engines
tests/               five suites: registration, chat template, both parsers, live smoke
docs/PARSERS.md      how each engine resolves parsers, and why
```

## Verified against

| Engine | Version | Pre-flight | Live smoke test |
|---|---|---|---|
| SGLang | 0.5.16 | 3/3 + 12/12 + 33/33 | 6/6 |
| vLLM | 0.28.0 | 3/3 + 5/5 + 33/33 | 6/6 |

An engine upgrade can move an internal API. Run `voicing-check` and the suites
in [SETUP.md](./SETUP.md) section 4 after any upgrade; they fail loudly and name
what moved.
