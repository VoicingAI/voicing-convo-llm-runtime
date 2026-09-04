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


def _assert_no_qwen(text: str) -> None:
    assert "qwen" not in text.lower(), text


def _redact_cases():
    from voicing_runtime.logfilter import _redact

    got = _redact("Load weight end. type=Qwen3_5MoeForConditionalGeneration")
    _assert_no_qwen(got)
    assert "VoicingConvo" in got, got

    got = _redact("model_type=qwen3_5_moe_text")
    assert got == "model_type=voicing_convo", got

    got = _redact("[qwen_gdn_linear_attn.py:158]")
    assert got == "[gdn_linear_attn.py:158]", got

    leftovers = (
        "Using fa3 as multimodal attention backend for Qwen2_5_VL",
        "File /usr/lib/python3.13/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py",
        "QWEN_BAR failed; Qwen2MoeForCausalLM; hello qWenMix",
        "incompatible with multimodal model qwen2.5-vl",
    )
    for raw in leftovers:
        got = _redact(raw)
        _assert_no_qwen(got)
        assert "voicing" in got.lower(), got


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


def _handler_stream_captured_before_install():
    """A StreamHandler created on the real stderr must still redact after install()."""
    from voicing_runtime import logfilter as lf

    lf._INSTALLED = False
    os.environ.pop("VOICING_REDACT_LOGS", None)
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s"))
    log = logging.getLogger("voicing_runtime_test_qwen_leak")
    log.handlers[:] = [handler]
    log.propagate = False
    log.setLevel(logging.INFO)
    try:
        assert lf.install()
        log.info("Load weight end. type=Qwen2MoeForCausalLM path=qwen3_5_moe")
        log.error("traceback File .../models/qwen2_5_vl/modeling_qwen2_5_vl.py")
    finally:
        log.handlers.clear()
        lf._INSTALLED = False
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    out = buf.getvalue()
    _assert_no_qwen(out)
    assert "voicing" in out.lower(), out


def _stderr_and_exc_have_no_qwen():
    from voicing_runtime import logfilter as lf

    lf._INSTALLED = False
    os.environ.pop("VOICING_REDACT_LOGS", None)
    out_buf, err_buf = io.StringIO(), io.StringIO()
    sys.stdout, sys.stderr = out_buf, err_buf
    try:
        assert lf.install()
        print("stdout Qwen3Moe", file=sys.stdout)
        print("stderr QWEN_KERNEL", file=sys.stderr)
        try:
            raise RuntimeError("failed loading qwen3_5_moe")
        except RuntimeError:
            sys.stderr.write("".join(__import__("traceback").format_exc()))
    finally:
        lf._INSTALLED = False
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    _assert_no_qwen(out_buf.getvalue())
    _assert_no_qwen(err_buf.getvalue())


if __name__ == "__main__":
    check("redact: PascalCase class + lowercase ids", _redact_cases)
    check("stdout: drop HF docstring [ERROR] prints, keep other lines", _drop_docstring_print)
    check("logging.Filter: PascalCase class name", _logging_pascalcase)
    check("logging.Filter: drop docstring lint records", _logging_drops_docstring_lint)
    check("StreamHandler attached before install() still redacts", _handler_stream_captured_before_install)
    check("stderr / traceback writes have no leftover qwen", _stderr_and_exc_have_no_qwen)
    failed = 0
    for ok, name, info in results:
        print(("ok   " if ok else "FAIL ") + name + (f"  {info}" if info else ""))
        failed += not ok
    print(f"{sum(ok for ok, _, _ in results)}/{len(results)} passed")
    sys.exit(1 if failed else 0)
