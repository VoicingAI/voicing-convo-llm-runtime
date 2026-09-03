# SPDX-License-Identifier: Apache-2.0
"""Voicing tool-call parser plugin for vLLM.

Registered under the name ``voicing`` by :func:`voicing_runtime.register.register`,
which the ``vllm.general_plugins`` entry point calls in every vLLM process. No
``--tool-parser-plugin`` flag is needed; just ``--tool-call-parser voicing``.
"""

from vllm.parser.engine.adapters import make_adapters
from vllm.tool_parsers import ToolParserManager

from .vllm_core import VoicingParser

_VoicingReasoningAdapter, _VoicingToolAdapter = make_adapters(VoicingParser)


class VoicingToolParser(_VoicingToolAdapter):  # type: ignore[valid-type, misc]
    # Identifier of the xgrammar builtin structural-tag template used for
    # grammar-constrained tool calling. This is an inference-backend grammar id,
    # not a brand name - it must match a template name xgrammar ships, so it is
    # deliberately left unchanged.
    structural_tag_model = "qwen_3_coder"


# Immediate (non-lazy) registration: this module was loaded by path, so a lazy
# ``module_path`` lookup could not re-import it by name later.
ToolParserManager.register_module(name="voicing", module=VoicingToolParser)

__all__ = ["VoicingToolParser"]
