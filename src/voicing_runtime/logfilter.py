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

Nothing in the engines is modified: this is a standard `logging.Filter` on the
root logger. It does not change behaviour, only the text of log records.

Set ``VOICING_REDACT_LOGS=0`` to turn it off, for example when reporting a bug
upstream and you want the engine's own identifiers verbatim.
"""

from __future__ import annotations

import logging
import os
import re

# Longest first: qwen3_5_moe_text must win over qwen3.
_SUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"qwen3_5_moe_text|qwen3_5_moe|qwen3_5"), "voicing_convo"),
    (re.compile(r"qwen_gdn"), "gdn"),
    (re.compile(r"Qwen Triton"), "GDN Triton"),
    (re.compile(r"qwen_triton_warmup"), "gdn_triton_warmup"),
    (re.compile(r"qwen3_coder"), "voicing"),
    (re.compile(r"\bqwen3\b"), "voicing"),
)
_ANY = re.compile(r"(?i)qwen")


def _redact(text: str) -> str:
    for pattern, replacement in _SUBS:
        text = pattern.sub(replacement, text)
    return text


class VoicingLogFilter(logging.Filter):
    """Rewrite vendor identifiers in log records. Never drops a record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
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


_INSTALLED = False


def install() -> bool:
    """Attach the filter to the root logger and every existing handler.

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
    _INSTALLED = True
    return True


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
