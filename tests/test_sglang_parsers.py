#!/usr/bin/env python3
"""Self-check for the Voicing SGLang parsers against the SGLang that is installed.

Run this on the serving box BEFORE deploying. It exercises exactly what the
server does at startup and at request time, so a version mismatch between
`voicing_parsers/sglang/` and the installed SGLang fails here, loudly, rather
than at launch:

    source /venv/main/bin/activate          # or wherever SGLang lives
    python tests/test_sglang_parsers.py /path/to/Voicing-Convo-V2-35B-MOE

Checks:
  1. both detector modules import (catches removed helpers / moved modules)
  2. they register under "voicing" in both SGLang registries
  3. `--reasoning-parser voicing --tool-call-parser voicing` pass argparse
  4. reasoning extraction, one-shot and streaming
  5. tool-call decoding, one-shot and streaming, including schema-typed args
  6. SGLang's own plugin loader registers everything in a clean interpreter
"""

import os
import subprocess
import sys
import traceback

MODEL_DIR = os.path.abspath(
    sys.argv[1] if len(sys.argv) > 1
    else os.environ.get("VOICING_MODEL_DIR")
    or sys.exit("error: pass the model directory as an argument or set VOICING_MODEL_DIR")
)

results = []


def check(name, fn):
    try:
        fn()
        results.append((True, name, ""))
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        where = f" ({os.path.basename(tb[-1].filename)}:{tb[-1].lineno})" if tb else ""
        results.append((False, name, f"{type(e).__name__}: {e}{where}"))


def main():
    try:
        import sglang  # noqa: F401
    except ModuleNotFoundError:
        print("SGLang is not installed in this interpreter; nothing to test.")
        return 2
    print(f"SGLang {getattr(sglang, '__version__', '?')}  ({os.path.dirname(sglang.__file__)})")

    import voicing_runtime
    voicing_runtime.register()

    # 1. import
    def _import():
        from voicing_runtime.parsers import sglang_reasoning, sglang_tool  # noqa: F401
    check("detector modules import against installed SGLang", _import)

    # 2. registration
    def _registered():
        from sglang.srt.parser.reasoning_parser import ReasoningParser
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        assert ReasoningParser.DetectorMap.get("voicing").__name__ == "VoicingReasoningDetector"
        assert FunctionCallParser.ToolCallParserEnum.get("voicing").__name__ == "VoicingToolDetector"
    check("registered as 'voicing' in both registries", _registered)

    # 3. argparse
    def _argparse():
        import argparse
        from sglang.srt.server_args import ServerArgs
        p = argparse.ArgumentParser()
        ServerArgs.add_cli_args(p)
        ns = p.parse_args(["--model-path", "x", "--reasoning-parser", "voicing", "--tool-call-parser", "voicing"])
        assert ns.reasoning_parser == "voicing" and ns.tool_call_parser == "voicing"
    check("--reasoning-parser voicing / --tool-call-parser voicing accepted by argparse", _argparse)

    # 4. reasoning
    def _reasoning_oneshot():
        from sglang.srt.parser.reasoning_parser import ReasoningParser
        det = ReasoningParser.DetectorMap["voicing"]()
        r = det.detect_and_parse("<think>plan it</think>Hello there")
        assert r.reasoning_text == "plan it", r.reasoning_text
        assert r.normal_text == "Hello there", r.normal_text
    check("reasoning: one-shot split", _reasoning_oneshot)

    def _token_chunks(text):
        # The server streams token by token, so tags such as </think> always
        # arrive whole. Character-sliced chunks would split them, which the
        # upstream base detector does not handle either -- not a real case.
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        return [tok.decode([i]) for i in tok.encode(text, add_special_tokens=False)]

    def _stream(det, text):
        reasoning, normal = "", ""
        for c in _token_chunks(text):
            r = det.parse_streaming_increment(c)
            reasoning += r.reasoning_text or ""
            normal += r.normal_text or ""
        return reasoning, normal

    def _reasoning_stream_prefixed():
        from sglang.srt.parser.reasoning_parser import ReasoningParser
        r, n = _stream(ReasoningParser.DetectorMap["voicing"](), "<think>plan it</think>Hello there")
        assert r.strip() == "plan it", r
        assert n.strip() == "Hello there", n
    check("reasoning: streaming, <think> present in output", _reasoning_stream_prefixed)

    def _reasoning_stream_server_shape():
        # Real shape: the chat template already emitted "<think>\n" in the
        # generation prompt, so output starts inside reasoning and the server
        # starts the detector in the reasoning state.
        from sglang.srt.parser.reasoning_parser import ReasoningParser
        r, n = _stream(ReasoningParser.DetectorMap["voicing"](force_reasoning=True), "plan it</think>Hello there")
        assert r.strip() == "plan it", r
        assert n.strip() == "Hello there", n
    check("reasoning: streaming, server shape (no <think>, forced reasoning state)", _reasoning_stream_server_shape)

    def _reasoning_matches_stock():
        # Guard against drift: the Voicing detector must behave exactly like the
        # upstream detector it was derived from, on every stream shape.
        from sglang.srt.parser.reasoning_parser import ReasoningParser
        for text, kw in (("<think>plan it</think>Hello", {}), ("plan it</think>Hello", {"force_reasoning": True}),
                         ("plan it<tool_call>x", {"force_reasoning": True}), ("no think at all", {})):
            v = _stream(ReasoningParser.DetectorMap["voicing"](**kw), text)
            q = _stream(ReasoningParser.DetectorMap["qwen3"](**kw), text)
            assert v == q, (text, v, q)
    check("reasoning: identical to the upstream detector on all stream shapes", _reasoning_matches_stock)

    def _reasoning_via_parser_class():
        from sglang.srt.parser.reasoning_parser import ReasoningParser
        rp = ReasoningParser(model_type="voicing")
        r = rp.parse_non_stream("<think>x</think>y")
        assert r == ("x", "y"), r
    check("reasoning: via ReasoningParser(model_type='voicing') as the server does", _reasoning_via_parser_class)

    # 5. tools
    def _tools():
        from sglang.srt.entrypoints.openai.protocol import Function, Tool
        return [Tool(type="function", function=Function(
            name="get_weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}))]

    CALL = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nBengaluru\n</parameter>\n"
            "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>")

    def _tool_oneshot():
        import json
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        det = FunctionCallParser.ToolCallParserEnum["voicing"]()
        r = det.detect_and_parse(CALL, _tools())
        assert len(r.calls) == 1 and r.calls[0].name == "get_weather", r.calls
        args = json.loads(r.calls[0].parameters)
        assert args == {"city": "Bengaluru", "days": 3}, args  # days typed from schema
    check("tools: one-shot decode with schema-typed integer", _tool_oneshot)

    def _tool_stream():
        import json
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        det = FunctionCallParser.ToolCallParserEnum["voicing"]()
        calls = []
        for i in range(0, len(CALL), 7):
            calls += det.parse_streaming_increment(CALL[i:i + 7], _tools()).calls
        names = [c.name for c in calls if c.name]
        args = json.loads("".join(c.parameters for c in calls if c.parameters))
        assert names == ["get_weather"], names
        assert args == {"city": "Bengaluru", "days": 3}, args
    check("tools: streaming decode reassembles the same call", _tool_stream)

    def _tool_via_parser_class():
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        fp = FunctionCallParser(_tools(), "voicing")
        assert fp.has_tool_call(CALL)
        normal, calls = fp.parse_non_stream("Sure. " + CALL)
        assert calls and calls[0].name == "get_weather", calls
    check("tools: via FunctionCallParser(tools, 'voicing') as the server does", _tool_via_parser_class)

    # 6. fresh interpreter via PYTHONPATH — the deployment route
    def _fresh_interpreter():
        # No PYTHONPATH: the engine loads the plugin entry point itself.
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        code = ("from sglang.srt.plugins import load_plugins; load_plugins();"
                "from sglang.srt.parser.reasoning_parser import ReasoningParser as R;"
                "from sglang.srt.function_call.function_call_parser import FunctionCallParser as F;"
                "print('voicing' in R.DetectorMap and 'voicing' in F.ToolCallParserEnum)")
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
        assert out.stdout.strip().endswith("True"), f"stdout={out.stdout!r}\nstderr={out.stderr[-800:]}"
        assert "[voicing] failed" not in out.stderr, out.stderr[-800:]
    check("SGLang plugin loader registers everything in a clean interpreter", _fresh_interpreter)

    passed = sum(1 for ok, _, _ in results if ok)
    for ok, name, err in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {err}" if err else ""))
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
