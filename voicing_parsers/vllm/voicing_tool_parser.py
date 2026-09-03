# SPDX-License-Identifier: Apache-2.0
"""Voicing tool-call parser plugin for vLLM.

Registers the tool parser under the name ``voicing``.

    vllm serve voicing-ai/Voicing-Convo-V2-35B-MOE \
      --tool-parser-plugin /path/to/voicing_parsers/vllm/voicing_tool_parser.py \
      --enable-auto-tool-choice --tool-call-parser voicing
"""

import os
import sys

# vLLM loads this file with ``import_from_path(basename, path)``, which does not
# put the plugin's own directory on sys.path. Do it here so the shared core
# module below resolves.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from vllm.parser.engine.adapters import make_adapters  # noqa: E402
from vllm.tool_parsers import ToolParserManager  # noqa: E402

from voicing_parser_core import VoicingParser  # noqa: E402

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
