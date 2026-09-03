# SPDX-License-Identifier: Apache-2.0
"""Voicing reasoning detector for SGLang.

Adapted for Voicing AI from the SGLang `<think>`/`</think>` reasoning detector
and kept in step with the `Voicing-Convo-V2-35B-MOE` chat template.

Registers itself under the reasoning parser name ``voicing``.
"""

from sglang.srt.parser.reasoning_parser import (
    BaseReasoningFormatDetector,
    ReasoningParser,
)


class VoicingReasoningDetector(BaseReasoningFormatDetector):
    """
    Detector for Voicing models (e.g. voicing-ai/Voicing-Convo-V2-35B-MOE).
    Assumes reasoning format:
      (<think>)*(.*)</think>

    Thinking mode can be switched off per request with the `enable_thinking`
    chat-template kwarg:
      - enable_thinking=True: "<think>reasoning content</think>The answer is 42."
      - enable_thinking=False: "The answer is 42." (no thinking tokens)

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        think_excluded_tokens = [
            "<tool_call>",
            "</tool_call>",
            "<|im_end|>",
            "<|endoftext|>",
        ]
        super().__init__(
            "<think>",
            "</think>",
            think_excluded_tokens=think_excluded_tokens,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            # The model sometimes opens ``<tool_call>`` without closing
            # ``</think>``; treat it as an implicit reasoning close.
            tool_start_token="<tool_call>",
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            thinks_internally=True,
            reasoning_default="enable_thinking",
            force_nonempty_content=force_nonempty_content,
        )


# --- Registration -----------------------------------------------------------
# Adds `--reasoning-parser voicing` to SGLang. Import this module before
# ServerArgs builds its argparse choices (see launch_voicing_server.py).
ReasoningParser.DetectorMap["voicing"] = VoicingReasoningDetector

__all__ = ["VoicingReasoningDetector"]
