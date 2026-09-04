# SPDX-License-Identifier: Apache-2.0
"""Serving runtime for Voicing-Convo-V2-35B-MOE.

Installing this package is all that is required: it declares plugin entry points
for both vLLM (``vllm.general_plugins``) and SGLang (``sglang.srt.plugins``),
which each engine loads in every process it starts. That registers

  * the model architecture ``VoicingConvoForCausalLM`` / ``model_type`` ``voicing_convo``
  * the ``voicing`` reasoning parser
  * the ``voicing`` tool-call parser

so the stock engine commands work with no PYTHONPATH, no plugin file paths and
no changes to engine source.
"""

from .register import ARCH, MODEL_TYPE, register, registered

__all__ = ["ARCH", "MODEL_TYPE", "register", "registered"]
__version__ = "1.0.4"
