"""Regression suite for the Voicing-Convo-V2-35B-MOE chat template.

Covers the default Voicing system prompt and the serving behaviours the
template guarantees: mid-conversation system messages, assistant-first turns,
tool-call arguments arriving as either a dict or a JSON string, and reasoning
extraction. Run after any template edit -- and remember the template lives in
BOTH chat_template.jinja and the chat_template key of tokenizer_config.json.

    python tests/test_chat_template.py /path/to/Voicing-Convo-V2-35B-MOE
"""
import json, os, sys, traceback
from pathlib import Path
from transformers import AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else (os.environ.get("VOICING_MODEL_DIR") or sys.exit("error: pass the model directory as an argument or set VOICING_MODEL_DIR"))
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
DEFAULT_SYS = "You are Voicing, a helpful AI assistant created by Voicing AI."

def r(msgs, **kw):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)

TOOLS = [{"type":"function","function":{"name":"get_weather","description":"w",
          "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]

results = []
def check(name, fn):
    try:
        fn(); results.append((True, name, ""))
    except Exception as e:
        results.append((False, name, f"{type(e).__name__}: {e}"))

# ---------- branding ----------
check("default system prompt injected when absent",
      lambda: (lambda o: (_ for _ in ()).throw(AssertionError(o)) if DEFAULT_SYS not in o else None)(r([{"role":"user","content":"hi"}])))
check("explicit system prompt overrides default",
      lambda: [__import__("sys"), (lambda o: None if (DEFAULT_SYS not in o and "Be terse." in o) else (_ for _ in ()).throw(AssertionError(o)))(r([{"role":"system","content":"Be terse."},{"role":"user","content":"hi"}]))])
check("default system prompt appears in tools branch too",
      lambda: (lambda o: None if DEFAULT_SYS in o and "# Tools" in o else (_ for _ in ()).throw(AssertionError(o)))(r([{"role":"user","content":"hi"}], tools=TOOLS)))
check("explicit system prompt honoured in tools branch",
      lambda: (lambda o: None if ("Be terse." in o and DEFAULT_SYS not in o) else (_ for _ in ()).throw(AssertionError(o)))(r([{"role":"system","content":"Be terse."},{"role":"user","content":"hi"}], tools=TOOLS)))
check("no Qwen branding leaks into any render",
      lambda: (lambda o: None if "qwen" not in o.lower() else (_ for _ in ()).throw(AssertionError(o)))(r([{"role":"user","content":"hi"}], tools=TOOLS)))

# ---------- PATCH 1: mid-conversation system ----------
check("P1 system message mid-conversation renders",
      lambda: (lambda o: None if "<|im_start|>system\nEscalate now.<|im_end|>" in o else (_ for _ in ()).throw(AssertionError(o)))(
          r([{"role":"user","content":"hi"},{"role":"assistant","content":"hello"},
             {"role":"system","content":"Escalate now."},{"role":"user","content":"ok"}])))
check("P1 multiple mid-conversation system messages",
      lambda: (lambda o: None if o.count("<|im_start|>system") == 3 else (_ for _ in ()).throw(AssertionError(f"count={o.count('<|im_start|>system')}\n{o}")))(
          r([{"role":"system","content":"base"},{"role":"user","content":"hi"},
             {"role":"system","content":"mid one"},{"role":"assistant","content":"a"},
             {"role":"system","content":"mid two"},{"role":"user","content":"b"}])))
check("P1 system as the very last message before generation",
      lambda: (lambda o: None if "Wrap up." in o else (_ for _ in ()).throw(AssertionError(o)))(
          r([{"role":"user","content":"hi"},{"role":"system","content":"Wrap up."}])))
check("P1 mid-conversation system with tools declared",
      lambda: (lambda o: None if "Escalate now." in o and "# Tools" in o else (_ for _ in ()).throw(AssertionError(o)))(
          r([{"role":"user","content":"hi"},{"role":"system","content":"Escalate now."},
             {"role":"user","content":"b"}], tools=TOOLS)))

# ---------- PATCH 2: assistant-first ----------
check("P2 assistant-first outbound greeting",
      lambda: (lambda o: None if "this is Voicing calling" in o else (_ for _ in ()).throw(AssertionError(o)))(
          r([{"role":"assistant","content":"Hello, this is Voicing calling."},{"role":"user","content":"who?"}])))
check("P2 assistant-only conversation (no user turn at all)",
      lambda: (lambda o: None if "Hi there" in o else (_ for _ in ()).throw(AssertionError(o)))(
          r([{"role":"assistant","content":"Hi there"}])))
check("P2 system + assistant-first, no user turn",
      lambda: (lambda o: None if "Greeting" in o else (_ for _ in ()).throw(AssertionError(o)))(
          r([{"role":"system","content":"You call people."},{"role":"assistant","content":"Greeting"}])))
check("P2 conversation of only tool_response-shaped user turns",
      lambda: (lambda o: None if "<tool_response>" in o else (_ for _ in ()).throw(AssertionError(o)))(
          r([{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"get_weather","arguments":{"city":"X"}}}]},
             {"role":"tool","content":"31C"}])))

# ---------- PATCH 3: tool call arguments ----------
check("P3 arguments as JSON string",
      lambda: (lambda o: None if "get_weather" in o else (_ for _ in ()).throw(AssertionError(o)))(
          r([{"role":"user","content":"w?"},
             {"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"get_weather","arguments":'{"city": "Bengaluru"}'}}]}])))
check("P3 arguments as dict renders <parameter=> tags",
      lambda: (lambda o: None if "<parameter=city>\nBengaluru\n</parameter>" in o else (_ for _ in ()).throw(AssertionError(o)))(
          r([{"role":"user","content":"w?"},
             {"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"get_weather","arguments":{"city":"Bengaluru"}}}]}])))
check("P3 empty arguments dict",
      lambda: (lambda o: None if "get_weather" in o else (_ for _ in ()).throw(AssertionError(o)))(
          r([{"role":"user","content":"w?"},
             {"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"get_weather","arguments":{}}}]}])))
check("P3 empty arguments JSON string",
      lambda: (lambda o: None if "get_weather" in o else (_ for _ in ()).throw(AssertionError(o)))(
          r([{"role":"user","content":"w?"},
             {"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"get_weather","arguments":"{}"}}]}])))
check("P3 non-string argument values (int/bool/list/nested)",
      lambda: (lambda o: None if "<parameter=n>" in o else (_ for _ in ()).throw(AssertionError(o)))(
          r([{"role":"user","content":"x"},
             {"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"f","arguments":{"n":3,"b":True,"l":[1,2],"d":{"k":"v"}}}}]}])))
check("P3 multiple tool calls in one assistant turn",
      lambda: (lambda o: None if o.count("<tool_call>") == 2 else (_ for _ in ()).throw(AssertionError(f"n={o.count('<tool_call>')}\n{o}")))(
          r([{"role":"user","content":"x"},
             {"role":"assistant","content":"","tool_calls":[
                 {"type":"function","function":{"name":"a","arguments":{"p":1}}},
                 {"type":"function","function":{"name":"b","arguments":'{"q": 2}'}}]}])))
check("P3 tool call with no arguments key at all",
      lambda: (lambda o: None if "get_weather" in o else (_ for _ in ()).throw(AssertionError(o)))(
          r([{"role":"user","content":"x"},
             {"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"get_weather"}}]}])))

# ---------- PATCH 4: think extraction ----------
check("P4 embedded <think> extracted, not double-wrapped",
      lambda: (lambda o: None if o.count("<think>") == 2 and "reasoning here" in o else (_ for _ in ()).throw(AssertionError(f"n={o.count('<think>')}\n{o}")))(
          r([{"role":"user","content":"q"},{"role":"assistant","content":"<think>reasoning here</think>answer"},
             {"role":"user","content":"q2"}], preserve_thinking=True)))
check("P4 explicit reasoning_content field honoured",
      lambda: (lambda o: None if "explicit reasoning" in o else (_ for _ in ()).throw(AssertionError(o)))(
          r([{"role":"user","content":"q"},
             {"role":"assistant","content":"answer","reasoning_content":"explicit reasoning"},
             {"role":"user","content":"q2"}], preserve_thinking=True)))
check("P4 assistant with no think block at all",
      lambda: (lambda o: None if "plain answer" in o else (_ for _ in ()).throw(AssertionError(o)))(
          r([{"role":"user","content":"q"},{"role":"assistant","content":"plain answer"},{"role":"user","content":"q2"}])))

# ---------- general / regression ----------
check("multi-turn tool loop end to end",
      lambda: (lambda o: None if "<tool_response>" in o and "31C" in o else (_ for _ in ()).throw(AssertionError(o)))(
          r([{"role":"user","content":"weather?"},
             {"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"get_weather","arguments":{"city":"B"}}}]},
             {"role":"tool","content":"31C"},
             {"role":"assistant","content":"It is 31C."},
             {"role":"user","content":"thanks"}], tools=TOOLS)))
check("consecutive tool responses grouped in one user turn",
      lambda: (lambda o: None if o.count("<tool_response>") == 2 else (_ for _ in ()).throw(AssertionError(f"n={o.count('<tool_response>')}\n{o}")))(
          r([{"role":"user","content":"x"},
             {"role":"assistant","content":"","tool_calls":[
                 {"type":"function","function":{"name":"a","arguments":{}}},
                 {"type":"function","function":{"name":"b","arguments":{}}}]},
             {"role":"tool","content":"r1"},{"role":"tool","content":"r2"}])))
check("enable_thinking=False emits empty think block",
      lambda: (lambda o: None if o.endswith("<think>\n\n</think>\n\n") else (_ for _ in ()).throw(AssertionError(repr(o[-40:]))))(
          r([{"role":"user","content":"hi"}], enable_thinking=False)))
check("enable_thinking default opens a think block",
      lambda: (lambda o: None if o.endswith("<think>\n") else (_ for _ in ()).throw(AssertionError(repr(o[-40:]))))(
          r([{"role":"user","content":"hi"}])))
check("assistant content None does not crash",
      lambda: (lambda o: None if "get_weather" in o else (_ for _ in ()).throw(AssertionError(o)))(
          r([{"role":"user","content":"x"},
             {"role":"assistant","content":None,"tool_calls":[{"type":"function","function":{"name":"get_weather","arguments":{}}}]}])))
check("multimodal image content still renders vision tokens",
      lambda: (lambda o: None if "<|vision_start|><|image_pad|><|vision_end|>" in o else (_ for _ in ()).throw(AssertionError(o)))(
          r([{"role":"user","content":[{"type":"image_url","image_url":{"url":"http://x/y.png"}},{"type":"text","text":"what?"}]}])))
check("add_generation_prompt=False omits the trailing assistant header",
      lambda: (lambda o: None if not o.rstrip().endswith("<think>") else (_ for _ in ()).throw(AssertionError(repr(o[-40:]))))(
          tok.apply_chat_template([{"role":"user","content":"hi"}], tokenize=False, add_generation_prompt=False)))
def _tokenizes():
    # transformers v5 returns a BatchEncoding from tokenize=True
    out = tok.apply_chat_template([{"role":"user","content":"hi"}], tokenize=True, add_generation_prompt=True)
    ids = out["input_ids"] if hasattr(out, "keys") else out
    assert len(ids) > 5, len(ids)
    assert DEFAULT_SYS in tok.decode(ids)
check("template tokenizes and decodes back to the Voicing prompt", _tokenizes)
check("empty messages still raises (guard intentionally kept)",
      lambda: [(_ for _ in ()).throw(AssertionError("should have raised")) if _try_empty() else None])

def _try_empty():
    try:
        r([]); return True
    except Exception:
        return False

# rerun that one properly
results = [x for x in results if x[1] != "empty messages still raises (guard intentionally kept)"]
check("empty messages still raises (guard intentionally kept)",
      lambda: None if not _try_empty() else (_ for _ in ()).throw(AssertionError("should have raised")))
check("unexpected role still raises (guard intentionally kept)",
      lambda: None if not _try_role() else (_ for _ in ()).throw(AssertionError("should have raised")))

def _try_role():
    try:
        r([{"role":"wizard","content":"x"}]); return True
    except Exception:
        return False
results = [x for x in results if x[1] != "unexpected role still raises (guard intentionally kept)"]
check("unexpected role still raises (guard intentionally kept)",
      lambda: None if not _try_role() else (_ for _ in ()).throw(AssertionError("should have raised")))

# ---------- report ----------
passed = sum(1 for ok,_,_ in results if ok)
for ok, name, err in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {err}" if err else ""))
print(f"\n{passed}/{len(results)} passed")
sys.exit(0 if passed == len(results) else 1)
