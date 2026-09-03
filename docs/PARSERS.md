# Voicing Parsers

> Deployment steps are in [SETUP.md](../SETUP.md). This file is the reference for
> how each engine resolves parsers and why the package registers them the way it does.

Reasoning and tool-call parsers for **Voicing-Convo-V2-35B-MOE**, packaged for
[vLLM](https://github.com/vllm-project/vllm) and
[SGLang](https://github.com/sgl-project/sglang).

Both engines need to know two things about the model's output format:

- **Reasoning** is wrapped in `<think> ... </think>`, so it can be surfaced as
  `reasoning_content` instead of leaking into `content`.
- **Tool calls** use an XML form rather than JSON:

  ```
  <tool_call>
  <function=get_weather>
  <parameter=city>
  Bengaluru
  </parameter>
  </function>
  </tool_call>
  ```

These files register both under the name `voicing`.

```
<package>/voicing_parsers/
├── vllm/
│   ├── voicing_parser_core.py        # shared state machine (reasoning + tool calls)
│   ├── voicing_reasoning_parser.py   # --reasoning-parser voicing
│   └── voicing_tool_parser.py        # --tool-call-parser voicing
└── sglang/
    ├── voicing_reasoning_detector.py # --reasoning-parser voicing
    ├── voicing_tool_detector.py      # --tool-call-parser voicing
    └── launch_voicing_server.py      # wrapper alternative to PYTHONPATH
```

## vLLM

vLLM loads parsers from a file path, so both plugins are passed on the command
line. The model architecture registers through `PYTHONPATH` (it has to reach
vLLM's engine-core process, which the plugin flags do not). No installation step.

```bash
vllm serve $MODEL_DIR \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --max-num-seqs 256 \
  --reasoning-parser-plugin voicing_reasoning_parser.py \
  --reasoning-parser voicing \
  --tool-parser-plugin voicing_tool_parser.py \
  --enable-auto-tool-choice \
  --tool-call-parser voicing
```

`--reasoning-parser-plugin` may be dropped if you only want tool calling, and
the three tool flags may be dropped if you only want reasoning separation.

## SGLang

Unlike vLLM, SGLang has no flag that takes a parser file path. Both
`--reasoning-parser` and `--tool-call-parser` are `choices`-constrained against
registries read while the argument parser is built:

```python
choices=["auto"] + list(ReasoningParser.DetectorMap.keys())
choices=["auto"] + list(FunctionCallParser.ToolCallParserEnum.keys())
```

So the detectors must already be registered before argv is parsed — passing a
path is rejected by argparse as an invalid choice.

Put this package's `voicing_runtime/` directory on `PYTHONPATH`. Python imports
`sitecustomize` at interpreter startup, and the one bundled there registers the
model architecture **and** both detectors — so the stock command works unchanged
(`voicing_parsers/sglang` still works as a PYTHONPATH target for compatibility):

```bash
python -m sglang.launch_server \
  --model-path $MODEL_DIR \
  --port 8000 \
  --tp-size 8 \
  --mem-fraction-static 0.8 \
  --context-length 262144 \
  --reasoning-parser voicing \
  --tool-call-parser voicing
```

This covers **every** process, including the workers SGLang spawns — which
matters, because workers reach the registries without going through
`server_args`. Registration is a no-op if SGLang is not installed, and
`sitecustomize.py` chains to any other `sitecustomize` on the path rather than
shadowing it.

`launch_voicing_server.py` is a wrapper equivalent for setups that cannot set an
env var; it takes the same flags as `python -m sglang.launch_server`.

## Verify before deploying

Parser code is tied to the engine version it was written against. Run the
self-test for your engine **on the serving box** before pointing traffic at it;
it exercises exactly what the server does at startup and per request, so a
mismatch fails here instead of at launch:

```bash
# from this package, in the engine's Python environment
python tests/test_sglang_parsers.py $MODEL_DIR   # SGLang: import, registration, argparse, reasoning + tool decode, streaming, PYTHONPATH route
python tests/test_vllm_parsers.py $MODEL_DIR     # vLLM: plugin load by path, name resolution, reasoning + tool decode, streaming
```

Each exits non-zero on any failure and prints the exception with file and line.

Once a server is up, run the live smoke test against it. It is engine-agnostic
(reads `reasoning` on vLLM >= 0.28 and `reasoning_content` on SGLang) and covers
identity, tool calling, streaming, and the chat-template behaviours end to end:

```bash
python tests/smoke_live_api.py http://127.0.0.1:8000/v1
```

The `voicing-serve` console command: `serve_sglang.sh` and
`serve_vllm.sh`, parameterised by `MODEL_DIR`, `PORT`, `TP`, and friends, with
any extra flags appended. Both were verified live (SGLang 0.5.16, vLLM 0.28.0)
on an RTX PRO 6000. Note vLLM needs `--max-num-seqs` lowered (the script sets
256): this hybrid architecture allots one Mamba cache block per decode sequence
and the default 1024 fails CUDA-graph capture.
The SGLang suite also asserts the Voicing reasoning detector behaves identically
to the upstream detector it derives from, on every stream shape, so silent drift
between the two is caught.

## Notes

- Both engines are given the same grammar, so tool-call behaviour is identical
  across them.
- `structural_tag_model` / `get_structural_tag_name()` keep the upstream
  identifier `qwen_3_coder`. That string names a grammar template built into
  **xgrammar**, the constrained-decoding backend — it is an engine-internal id,
  not a label that appears anywhere in the served output. Renaming it would
  break grammar-constrained tool calling.
- The SGLang detectors are based on the sources shipped in **SGLang 0.5.16** and
  verified against it on an RTX PRO 6000 box: registration at interpreter startup,
  both argparse choices, reasoning extraction, and tool-call decoding in both the
  one-shot and streaming paths. They depend only on `BaseFormatDetector`,
  `BaseReasoningFormatDetector`, and `core_types`, which are stable across 0.5.x
  and `main`. The vLLM parsers target `vllm>=0.19.0` and were verified
  structurally against `main`, not run.

## License

Apache 2.0, matching the upstream engines these parsers were adapted from.
