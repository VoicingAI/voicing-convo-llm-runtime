# Voicing Serving Runtime

Everything needed to serve **Voicing-Convo-V2-35B-MOE** on **vLLM** or **SGLang**,
kept out of the model repository on purpose.

**Start here: [SETUP.md](./SETUP.md)** is the step-by-step guide for a new machine.

> **Internal.** This package is deployed next to the inference engine: baked into
> the serving image, cloned onto the host, or mounted. It must never be copied
> into the published model repository, which contains only weights, tokenizer,
> config, and chat template.

## What it does

The model declares its own architecture:

```json
{ "architectures": ["VoicingConvoForCausalLM"], "model_type": "voicing_convo" }
```

Neither engine knows those names on its own, so the model does not load until
this package is on `PYTHONPATH`. At interpreter startup it registers:

- the **architecture** and **config class**, with transformers, SGLang, and vLLM
- the **`voicing` reasoning parser**, so thinking is returned separately from the answer
- the **`voicing` tool-call parser**, so the model's XML tool calls arrive as normal OpenAI `tool_calls`

Registration happens in the server process and in every worker the engine spawns.
No engine source is modified and nothing is pip-installed.

## Layout

```
SETUP.md                       step-by-step deployment guide  <- start here
voicing_runtime/
  sitecustomize.py             auto-imported at startup; registers everything
  voicing_convo.py             architecture + config registration, per engine
voicing_parsers/
  README.md                    how each engine resolves parsers, and why
  sglang/
    voicing_reasoning_detector.py    --reasoning-parser voicing
    voicing_tool_detector.py         --tool-call-parser voicing
    launch_voicing_server.py         wrapper alternative to PYTHONPATH
  vllm/
    voicing_parser_core.py           shared state machine (reasoning + tool calls)
    voicing_reasoning_parser.py      --reasoning-parser voicing
    voicing_tool_parser.py           --tool-call-parser voicing
scripts/
  serve_sglang.sh              parameterised launchers
  serve_vllm.sh
tests/
  test_model_registration.py   architecture/config resolve; keys match checkpoint
  test_chat_template.py        33 chat-template cases
  test_sglang_parsers.py       12 parser cases against the installed SGLang
  test_vllm_parsers.py          5 parser cases against the installed vLLM
  smoke_live_api.py             6 end-to-end cases against a running server
```

## Verified against

| Engine | Version | Pre-flight | Live smoke test |
|---|---|---|---|
| SGLang | 0.5.16 | 12/12 + 3/3 + 33/33 | 6/6 |
| vLLM | 0.28.0 | 5/5 + 3/3 | 6/6 |

An engine upgrade can move an internal API. Re-run the pre-flight checks in
[SETUP.md](./SETUP.md) section 4 after any upgrade; they fail loudly and name
what moved.
