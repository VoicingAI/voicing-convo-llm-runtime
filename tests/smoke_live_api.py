#!/usr/bin/env python3
"""Live smoke test for a running Voicing-Convo-V2-35B-MOE server (SGLang or vLLM).

    python tests/smoke_live_api.py [http://host:port/v1] [served-model-name]

Exercises, over the OpenAI-compatible API, everything the parsers and chat
template are responsible for:
  1. default Voicing identity (no system message sent)
  2. tool call decoded into `tool_calls` with finish_reason=tool_calls
  3. streaming: content streams with thinking off; with thinking on, reasoning
     comes first and tags never leak
  4. mid-conversation system message accepted (template patch 1)
  5. assistant-first conversation accepted (template patch 2)
  6. tool-call replay with `arguments` as a JSON string (template patch 3)

Reads reasoning from `reasoning` (vLLM >= 0.28) or `reasoning_content` (SGLang,
older vLLM). Uses the model card's thinking-mode sampling and a large budget
where thinking is on: this model reasons at length even for trivial prompts, and
a small max_tokens ends the stream mid-reasoning, which looks like "no content".
Exits non-zero on any failure.
"""

import json
import sys
import traceback
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "voicing-ai/Voicing-Convo-V2-35B-MOE"
THINK = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}
NOTHINK = {"temperature": 0.7, "top_p": 0.8, "chat_template_kwargs": {"enable_thinking": False}}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                         "required": ["city"]}}}]


def post(body, stream=False):
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=900)
    return r if stream else json.loads(r.read())


def reasoning_of(d):
    return d.get("reasoning") or d.get("reasoning_content") or ""


results = []


def check(name, fn):
    try:
        info = fn() or ""
        results.append((True, name, info))
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        results.append((False, name, f"{type(e).__name__}: {e} (line {tb[-1].lineno})"))


def t_identity():
    m = post({"model": MODEL, "messages": [{"role": "user", "content": "Who are you and who made you? One sentence."}],
              "max_tokens": 200, **NOTHINK})["choices"][0]["message"]
    c = (m.get("content") or "").strip()
    assert "Voicing" in c, c
    return c[:80]


def t_tool_call():
    r = post({"model": MODEL, "messages": [{"role": "user", "content": "What is the weather in Bengaluru right now, in celsius?"}],
              "tools": TOOLS, "max_tokens": 4000, **THINK})
    m = r["choices"][0]["message"]
    calls = m.get("tool_calls") or []
    assert r["choices"][0]["finish_reason"] == "tool_calls", r["choices"][0]["finish_reason"]
    assert calls and calls[0]["function"]["name"] == "get_weather", calls
    args = json.loads(calls[0]["function"]["arguments"])
    assert args.get("city", "").lower().startswith("beng"), args
    return f"{calls[0]['function']['name']}({calls[0]['function']['arguments']})"


def _stream(body):
    """Consume a streaming response into (order, reasoning, content, finish)."""
    order, R, C, fin = "", "", "", None
    for line in post(body, stream=True):
        line = line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        ch = json.loads(line[5:])["choices"][0]
        d = ch["delta"]
        if reasoning_of(d):
            order += "r"; R += reasoning_of(d)
        if d.get("content"):
            order += "c"; C += d["content"]
        fin = ch.get("finish_reason") or fin
    return order, R, C, fin


def t_streaming_content():
    """Content streams cleanly with thinking off. Deterministic and fast."""
    order, R, C, fin = _stream({"model": MODEL,
                                "messages": [{"role": "user", "content": "Say hello in five words."}],
                                "max_tokens": 200, "stream": True, **NOTHINK})
    assert "c" in order, f"no content deltas (finish={fin})"
    assert not R, f"unexpected reasoning with thinking off: {R[:80]!r}"
    assert "</think>" not in C and "<think>" not in C, "think tags leaked into content"
    assert fin == "stop", fin
    return f"{order.count('c')} content deltas -> {C.strip()[:60]!r}"


def t_streaming_reasoning():
    """With thinking on, reasoning streams first and never leaks its tags.

    This model can reason for tens of thousands of tokens on a trivial prompt,
    so a fixed budget cannot guarantee it reaches the answer. Ordering and tag
    separation are asserted always; the content assertion only applies when the
    model actually finished.
    """
    order, R, C, fin = _stream({"model": MODEL,
                                "messages": [{"role": "user", "content": "Say hello in five words."}],
                                "max_tokens": 32000, "stream": True, **THINK})
    assert "r" in order, "no reasoning deltas at all"
    assert "c" not in order.rstrip("c"), "content interleaved with reasoning"
    assert "</think>" not in R, "</think> leaked into reasoning"
    if fin == "length":
        return (f"{order.count('r')} reasoning deltas, ordering and tags correct; "
                f"hit the {32000}-token budget before answering ({len(R)} reasoning chars), "
                "which is this model's normal long-thinking behaviour")
    assert "c" in order, f"finished as {fin} with no content"
    return f"{order.count('r')} reasoning / {order.count('c')} content deltas -> {C.strip()[:50]!r}"


def t_mid_system():
    m = post({"model": MODEL, "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello! How can I help?"},
              {"role": "system", "content": "Service failure detected. Apologize briefly and end the call."},
              {"role": "user", "content": "ok"}], "max_tokens": 120, **NOTHINK})["choices"][0]["message"]
    c = (m.get("content") or "").strip()
    assert c, "empty content"
    return c[:80]


def t_assistant_first():
    m = post({"model": MODEL, "messages": [{"role": "assistant", "content": "Hello, this is Voicing calling about your appointment."},
              {"role": "user", "content": "which appointment?"}], "max_tokens": 120, **NOTHINK})["choices"][0]["message"]
    c = (m.get("content") or "").strip()
    assert c, "empty content"
    return c[:80]


def t_json_string_args():
    m = post({"model": MODEL, "messages": [{"role": "user", "content": "weather in Bengaluru?"},
              {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
                  "name": "get_weather", "arguments": "{\"city\": \"Bengaluru\", \"unit\": \"c\"}"}}]},
              {"role": "tool", "tool_call_id": "call_1", "content": "{\"temperature\": 31, \"conditions\": \"clear\"}"}],
              "tools": TOOLS, "max_tokens": 150, **NOTHINK})["choices"][0]["message"]
    c = (m.get("content") or "").strip()
    assert "31" in c, c
    return c[:80]


def main():
    models = json.loads(urllib.request.urlopen(BASE + "/models", timeout=30).read())["data"]
    print(f"server: {BASE}  engine: {models[0].get('owned_by')}  models: {[m['id'] for m in models]}")
    check("1. default Voicing identity", t_identity)
    check("2. tool call decoded, finish_reason=tool_calls", t_tool_call)
    check("3a. streaming: content streams cleanly (thinking off)", t_streaming_content)
    check("3b. streaming: reasoning first, tags never leak", t_streaming_reasoning)
    check("4. mid-conversation system message (patch 1)", t_mid_system)
    check("5. assistant-first conversation (patch 2)", t_assistant_first)
    check("6. tool-call replay with JSON-string arguments (patch 3)", t_json_string_args)
    passed = sum(1 for ok, _, _ in results if ok)
    for ok, name, info in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {info}" if info else ""))
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
