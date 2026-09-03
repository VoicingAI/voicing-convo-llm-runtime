# SPDX-License-Identifier: Apache-2.0
"""Voicing reasoning parser plugin for vLLM.

Registers the reasoning parser under the name ``voicing``.

    vllm serve voicing-ai/Voicing-Convo-V2-35B-MOE \
      --reasoning-parser-plugin /path/to/voicing_parsers/vllm/voicing_reasoning_parser.py \
      --reasoning-parser voicing
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
from vllm.reasoning import ReasoningParserManager  # noqa: E402

from voicing_parser_core import VoicingParser  # noqa: E402

VoicingReasoningParser, _VoicingToolAdapter = make_adapters(VoicingParser)

# Immediate (non-lazy) registration: this module was loaded by path, so a lazy
# ``module_path`` lookup could not re-import it by name later.
ReasoningParserManager.register_module(name="voicing", module=VoicingReasoningParser)

__all__ = ["VoicingReasoningParser"]
