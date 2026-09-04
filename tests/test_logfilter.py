#!/usr/bin/env python3
"""Self-check for vendor-log redaction. No model directory required.

    python tests/test_logfilter.py

Covers the two gaps that leaked Qwen on the serving box:

  1. Transformers @auto_docstring prints ``[ERROR] `loss` is part of
     Qwen3_5Moe... not documented`` to stdout. Those must be dropped.
  2. PascalCase class names (``Qwen3_5MoeForConditionalGeneration``) must
     redact, not only lowercase ``qwen3_5_moe``.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import traceback

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

results = []


def check(name, fn):
    try:
        info = fn() or ""
        results.append((True, name, info))
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        results.append((False, name, f"{type(e).__name__}: {e} (line {tb[-1].lineno})"))


DOC_ERR = (
    "[ERROR] `loss` is part of Qwen3_5MoeCausalLMOutputWithPast.__init__'s "
    "signature, but not documented. Make sure to add it to the docstring of "
    "the function in /usr/lib/python3.13/site-packages/transformers/models/"
    "qwen3_5_moe/modeling_qwen3_5_moe.py."
)
DOC_ERR_VL = (
    "[ERROR] `logits` is part of Qwen2_5_VLCausalLMOutputWithPast.__init__'s "
    "signature, but not documented. Make sure to add it to the docstring of "
    "the function in /usr/lib/python3.13/site-packages/transformers/models/"
    "qwen2_5_vl/modeling_qwen2_5_vl.py."
)


def _redact_cases():
    from voicing_runtime.logfilter import _redact

    got = _redact("Load weight end. type=Qwen3_5MoeForConditionalGeneration")
    assert "qwen" not in got.lower(), got
    assert "VoicingConvo" in got, got

    got = _redact("model_type=qwen3_5_moe_text")
    assert got == "model_type=voicing_convo", got

    got = _redact("[qwen_gdn_linear_attn.py:158]")
    assert got == "[gdn_linear_attn.py:158]", got


def _drop_docstring_print():
    from voicing_runtime import logfilter as lf

    lf._INSTALLED = False
    os.environ.pop("VOICING_REDACT_LOGS", None)
    real = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    try:
        assert lf.install()
        print(DOC_ERR)
        print(DOC_ERR_VL)
        print("server ready")
    finally:
        sys.stdout = real
        lf._INSTALLED = False
    out = buf.getvalue()
    assert "not documented" not in out, out
    assert "Qwen" not in out, out
    assert "server ready" in out, out


def _logging_pascalcase():
    from voicing_runtime.logfilter import VoicingLogFilter

    record = logging.LogRecord(
        name="sglang", level=logging.INFO, pathname="x", lineno=1,
        msg="Load weight end. type=Qwen3_5MoeForConditionalGeneration",
        args=(), exc_info=None,
    )
    VoicingLogFilter().filter(record)
    text = str(record.msg)
    assert "qwen" not in text.lower(), text
    assert "VoicingConvo" in text, text


def _logging_drops_docstring_lint():
    from voicing_runtime.logfilter import VoicingLogFilter

    record = logging.LogRecord(
        name="transformers", level=logging.ERROR, pathname="x", lineno=1,
        msg=DOC_ERR, args=(), exc_info=None,
    )
    assert VoicingLogFilter().filter(record) is False


if __name__ == "__main__":
    check("redact: PascalCase class + lowercase ids", _redact_cases)
    check("stdout: drop HF docstring [ERROR] prints, keep other lines", _drop_docstring_print)
    check("logging.Filter: PascalCase class name", _logging_pascalcase)
    check("logging.Filter: drop docstring lint records", _logging_drops_docstring_lint)
    failed = 0
    for ok, name, info in results:
        print(("ok   " if ok else "FAIL ") + name + (f"  {info}" if info else ""))
        failed += not ok
    print(f"{sum(ok for ok, _, _ in results)}/{len(results)} passed")
    sys.exit(1 if failed else 0)
