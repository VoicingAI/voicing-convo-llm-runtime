#!/usr/bin/env python3
"""Self-check for the Voicing vLLM parser plugins against the vLLM that is installed.

Run this on the serving box BEFORE deploying:

    python tests/test_vllm_parsers.py /path/to/Voicing-Convo-V2-35B-MOE

Loads the two plugin files exactly the way `vllm serve` does
(`--reasoning-parser-plugin` / `--tool-parser-plugin`), resolves them by the
name "voicing", instantiates them with this model's tokenizer, and runs
reasoning extraction and tool-call decoding, one-shot and streaming.
"""

import os
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
        import vllm  # noqa: F401
    except ModuleNotFoundError:
        print("vLLM is not installed in this interpreter; nothing to test.")
        return 2
    print(f"vLLM {getattr(vllm, '__version__', '?')}  ({os.path.dirname(vllm.__file__)})")

    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    # 1. load the plugin files the way the server does
    def _load():
        import voicing_runtime
        voicing_runtime.register()
        ReasoningParserManager.get_reasoning_parser("voicing")
        ToolParserManager.get_tool_parser("voicing")
    check("package registers both parsers as 'voicing' (no plugin flags)", _load)

    def _tok():
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(MODEL_DIR)

    class _Req:  # minimal stand-in for ChatCompletionRequest fields the parsers read
        chat_template_kwargs = None
        tools = None

    # 2. reasoning
    def _reasoning_oneshot():
        cls = ReasoningParserManager.get_reasoning_parser("voicing")
        rp = cls(_tok())
        reasoning, content = rp.extract_reasoning("plan it</think>Hello there", _Req())
        assert reasoning.strip() == "plan it", reasoning
        assert content.strip() == "Hello there", content
    check("reasoning: one-shot split (generation prompt already opened <think>)", _reasoning_oneshot)

    def _reasoning_end_detection():
        cls = ReasoningParserManager.get_reasoning_parser("voicing")
        tok = _tok()
        rp = cls(tok)
        ids_open = tok.encode("<think>still thinking", add_special_tokens=False)
        ids_closed = tok.encode("<think>done</think>answer", add_special_tokens=False)
        assert not rp.is_reasoning_end(ids_open)
        assert rp.is_reasoning_end(ids_closed)
    check("reasoning: is_reasoning_end on token ids", _reasoning_end_detection)

    # 3. tools
    CALL = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nBengaluru\n</parameter>\n"
            "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>")

    def _tool_oneshot():
        import json
        cls = ToolParserManager.get_tool_parser("voicing")
        tp = cls(_tok())
        r = tp.extract_tool_calls("Sure. " + CALL, _Req())
        assert r.tools_called, r
        assert r.tool_calls[0].function.name == "get_weather", r.tool_calls
        args = json.loads(r.tool_calls[0].function.arguments)
        assert args["city"] == "Bengaluru" and str(args["days"]) == "3", args
        assert (r.content or "").strip() == "Sure.", repr(r.content)
    check("tools: one-shot decode", _tool_oneshot)

    def _tool_stream():
        import json
        cls = ToolParserManager.get_tool_parser("voicing")
        tp = cls(_tok())
        prev, names, args = "", [], ""
        for i in range(0, len(CALL), 7):
            cur = CALL[: i + 7]
            delta = tp.extract_tool_calls_streaming(prev, cur, cur[len(prev):], [], [], [], _Req())
            prev = cur
            if delta and getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    if tc.function and tc.function.name:
                        names.append(tc.function.name)
                    if tc.function and tc.function.arguments:
                        args += tc.function.arguments
        if hasattr(tp, "finish_streaming"):
            tail = tp.finish_streaming()
            if tail and getattr(tail, "tool_calls", None):
                for tc in tail.tool_calls:
                    if tc.function and tc.function.arguments:
                        args += tc.function.arguments
        assert names == ["get_weather"], names
        parsed = json.loads(args)
        assert parsed["city"] == "Bengaluru", parsed
    check("tools: streaming decode reassembles the same call", _tool_stream)

    passed = sum(1 for ok, _, _ in results if ok)
    for ok, name, err in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {err}" if err else ""))
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
