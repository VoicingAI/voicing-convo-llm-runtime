# SPDX-License-Identifier: Apache-2.0
"""Voicing reasoning parser plugin for vLLM.

Registered under the name ``voicing`` by :func:`voicing_runtime.register.register`,
which the ``vllm.general_plugins`` entry point calls in every vLLM process. No
``--reasoning-parser-plugin`` flag is needed; just ``--reasoning-parser voicing``.
    vllm serve voicing-ai/Voicing-Convo-V2-35B-MOE \
      --reasoning-parser-plugin /path/to/voicing_parsers/vllm/voicing_reasoning_parser.py \
      --reasoning-parser voicing
"""

from vllm.parser.engine.adapters import make_adapters
from vllm.reasoning import ReasoningParserManager

from .vllm_core import VoicingParser

VoicingReasoningParser, _VoicingToolAdapter = make_adapters(VoicingParser)

# Immediate (non-lazy) registration: this module was loaded by path, so a lazy
# ``module_path`` lookup could not re-import it by name later.
ReasoningParserManager.register_module(name="voicing", module=VoicingReasoningParser)

__all__ = ["VoicingReasoningParser"]
