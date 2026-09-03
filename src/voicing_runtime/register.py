# SPDX-License-Identifier: Apache-2.0
"""Entry point both engines call to register the Voicing model and parsers.

vLLM calls this through ``vllm.general_plugins`` and SGLang through
``sglang.srt.plugins``. Each engine loads its plugins in the launcher, the
engine core and the worker processes, so registration reaches every process
without any environment variable.

Everything here is idempotent: engines call it more than once per process.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging

log = logging.getLogger("voicing_runtime")

ARCH = "VoicingConvoForCausalLM"
MODEL_TYPE = "voicing_convo"

_STATE: dict[str, bool] = {}


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def register() -> dict[str, bool]:
    """Register the architecture and parsers with every installed engine.

    Returns a map of engine name to whether it was registered. Safe to call
    repeatedly; the work is done once per process.
    """
    if _STATE:
        return dict(_STATE)

    # Importing the model module performs the architecture and config
    # registration for whichever of transformers / SGLang / vLLM is installed.
    from . import model

    _STATE.update(model.REGISTERED)

    # Parsers. vLLM resolves these lazily by name, SGLang needs them present
    # before it builds its argparse choices; both are satisfied here because the
    # engines load plugins before either happens.
    if _installed("sglang"):
        for mod in ("sglang_reasoning", "sglang_tool"):
            importlib.import_module(f".parsers.{mod}", __package__)
    if _installed("vllm"):
        for mod in ("vllm_reasoning", "vllm_tool"):
            importlib.import_module(f".parsers.{mod}", __package__)

    log.debug("voicing runtime registered: %s", _STATE)
    return dict(_STATE)


def registered() -> dict[str, bool]:
    """Engines registered so far in this process (empty before `register()`)."""
    return dict(_STATE)
