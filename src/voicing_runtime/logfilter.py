# SPDX-License-Identifier: Apache-2.0
"""Strip the upstream vendor name from engine log output.

The engines log a handful of their own internal identifiers that carry the
upstream vendor's name: kernel module filenames, a warm-up message, and compiled
custom-op names in the engine config dump. They are internal to vLLM and SGLang
and cannot be renamed without patching those projects.

This filter rewrites them on the way out, so operational logs read in terms of
the architecture rather than the vendor. It substitutes the technically
meaningful part rather than inventing a new name, so log lines stay traceable:

    qwen_gdn_linear_attn.py        ->  gdn_linear_attn.py
    Warming up Qwen Triton kernels ->  Warming up GDN Triton kernels
    vllm::qwen_gdn_attention_core  ->  vllm::gdn_attention_core
    model_type=qwen3_5_moe_text    ->  model_type=voicing_convo
    Qwen3_5MoeForConditionalGeneration -> VoicingConvoForConditionalGeneration

Transformers' ``@auto_docstring`` also ``print``s ``[ERROR] ... not documented``
lines for those upstream classes. Those are docstring lint, not load failures;
they are dropped on stdout/stderr so they never reach the container log.

Nothing in the engines is modified: this is a standard ``logging.Filter`` plus
a stdio wrapper. It does not change behaviour, only the text that leaves the
process.

Set ``VOICING_REDACT_LOGS=0`` to turn it off, for example when reporting a bug
upstream and you want the engine's own identifiers verbatim.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import TextIO

# Longest first: qwen3_5_moe_text must win over qwen3. PascalCase class names
# (Qwen3_5Moe...) are rewritten before the lowercase path fragments.
_SUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"Qwen3_5Moe"), "VoicingConvo"),
    (re.compile(r"qwen3_5_moe_text|qwen3_5_moe|qwen3_5", re.IGNORECASE), "voicing_convo"),
    (re.compile(r"qwen2_5_vl", re.IGNORECASE), "voicing_convo"),
    (re.compile(r"qwen_gdn", re.IGNORECASE), "gdn"),
    (re.compile(r"Qwen Triton", re.IGNORECASE), "GDN Triton"),
    (re.compile(r"qwen_triton_warmup", re.IGNORECASE), "gdn_triton_warmup"),
    (re.compile(r"qwen3_coder", re.IGNORECASE), "voicing"),
    (re.compile(r"\bqwen3\b", re.IGNORECASE), "voicing"),
)
_ANY = re.compile(r"(?i)qwen")
_DOCSTRING_LINT = re.compile(
    r"^\[ERROR\] `[^`]+` is part of \S+'s signature, but not documented\."
)


def _redact(text: str) -> str:
    for pattern, replacement in _SUBS:
        text = pattern.sub(replacement, text)
    return text


def _drop_line(text: str) -> bool:
    return bool(_DOCSTRING_LINT.search(text.lstrip("\n")))


class VoicingLogFilter(logging.Filter):
    """Rewrite vendor identifiers in log records. Drops HF docstring lint."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        if _drop_line(message):
            return False
        if _ANY.search(message):
            # Collapse msg+args into the redacted text; the record still carries
            # everything it did before, just with the identifiers substituted.
            record.msg = _redact(message)
            record.args = ()
        for attr in ("filename", "module", "name", "pathname"):
            value = getattr(record, attr, None)
            if isinstance(value, str) and _ANY.search(value):
                setattr(record, attr, _redact(value))
        return True


class _FilteredStream:
    """Line-buffer a text stream: drop docstring lint, redact vendor names."""

    def __init__(self, inner: TextIO) -> None:
        self._inner = inner
        self._buf = ""

    def write(self, data) -> int:
        if not isinstance(data, str):
            return self._inner.write(data)
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line)
        return len(data)

    def _emit(self, line: str) -> None:
        if _drop_line(line):
            return
        if _ANY.search(line):
            line = _redact(line)
        self._inner.write(line + "\n")

    def flush(self) -> None:
        if self._buf:
            leftover, self._buf = self._buf, ""
            if not _drop_line(leftover):
                self._inner.write(_redact(leftover) if _ANY.search(leftover) else leftover)
        self._inner.flush()

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


_INSTALLED = False


def install() -> bool:
    """Attach the filter to logging and wrap stdout/stderr.

    Idempotent. Returns True if the filter is active in this process.
    """
    global _INSTALLED
    if _INSTALLED:
        return True
    if os.environ.get("VOICING_REDACT_LOGS", "1").lower() in ("0", "false", "no", "off"):
        return False
    f = VoicingLogFilter()

    # Cover handlers that already exist. vLLM configures its logging with
    # dictConfig when `vllm.logger` is first imported, which happens before
    # plugins load, and its logger sets propagate=False with its own handler --
    # so a filter on the root logger alone never sees those records.
    _attach_everywhere(f)

    # And handlers created after this point.
    _patch_handler_init(f)
    _wrap_stdio()
    _INSTALLED = True
    return True


def _wrap_stdio() -> None:
    if not isinstance(sys.stdout, _FilteredStream):
        sys.stdout = _FilteredStream(sys.stdout)  # type: ignore[assignment]
    if not isinstance(sys.stderr, _FilteredStream):
        sys.stderr = _FilteredStream(sys.stderr)  # type: ignore[assignment]


def _attach_everywhere(f: logging.Filter) -> None:
    """Attach the filter to the root logger and every logger/handler that exists."""
    seen: set[int] = set()

    def attach(target) -> None:
        if id(target) in seen:
            return
        seen.add(id(target))
        try:
            if f not in target.filters:
                target.addFilter(f)
        except Exception:
            pass

    root = logging.getLogger()
    attach(root)
    for handler in list(root.handlers):
        attach(handler)
    for logger in list(logging.Logger.manager.loggerDict.values()):
        if not isinstance(logger, logging.Logger):
            continue  # PlaceHolder
        attach(logger)
        for handler in list(getattr(logger, "handlers", ())):
            attach(handler)


def _patch_handler_init(f: logging.Filter) -> None:
    """Ensure handlers created later also carry the filter."""
    if getattr(logging.Handler, "_voicing_patched", False):
        return
    original = logging.Handler.__init__

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        try:
            self.addFilter(f)
        except Exception:
            pass

    logging.Handler.__init__ = __init__  # type: ignore[method-assign]
    logging.Handler._voicing_patched = True  # type: ignore[attr-defined]
