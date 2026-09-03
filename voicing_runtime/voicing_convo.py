# SPDX-License-Identifier: Apache-2.0
"""Voicing-Convo-V2 model registration for transformers, SGLang, and vLLM.

The checkpoint declares its own identity:

    "architectures": ["VoicingConvoForCausalLM"],
    "model_type":    "voicing_convo"

Each engine resolves those two strings through a registry. This module registers
them, at import time, against the engine classes that implement this hybrid
Gated-DeltaNet / MoE architecture -- no engine source is modified. It is loaded
automatically by ``sitecustomize.py`` in this directory when the directory is on
``PYTHONPATH``, or can be imported explicitly:

    import voicing_convo  # registers for whichever engines are installed

All classes are defined at module level on purpose: SGLang and vLLM pickle the
model config to spawn worker processes, and only module-level classes pickle by
reference. Each worker re-imports this module via sitecustomize before
unpickling, so the reference resolves.

An engine that is not installed is skipped silently; an engine whose API has
moved raises loudly.
"""

from __future__ import annotations

import importlib.util
import logging

ARCH = "VoicingConvoForCausalLM"
MODEL_TYPE = "voicing_convo"

# The serving engines (SGLang, vLLM) build their layers by comparing the config
# object's ``model_type`` against the architecture family they implement, and
# raise on anything else. The checkpoint files carry only the Voicing name; the
# engine-facing config objects registered below report the family id at
# runtime so that dispatch works without touching engine source. transformers
# does not dispatch on it, so its config keeps the Voicing name throughout.
ENGINE_FAMILY = "qwen3_5_moe_text"

log = logging.getLogger("voicing_convo")

# Text-only serving uses one-dimensional positions. If a config still carries the
# M-RoPE fields from the multimodal training stack, drop them: with identical
# t/h/w positions they are equivalent to plain RoPE, and removing them keeps the
# engines off their multimodal code paths.
_MROPE_KEYS = ("mrope_section", "mrope_interleaved")


def _strip_mrope(config) -> None:
    rp = getattr(config, "rope_parameters", None)
    if isinstance(rp, dict):
        for k in _MROPE_KEYS:
            rp.pop(k, None)


def _installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


REGISTERED: dict[str, bool] = {"transformers": False, "sglang": False, "vllm": False}

# --------------------------------------------------------------------------- #
# transformers
# --------------------------------------------------------------------------- #
if _installed("transformers"):
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    class VoicingConvoConfig(Qwen3_5MoeTextConfig):
        model_type = MODEL_TYPE

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _strip_mrope(self)

    class VoicingConvoForCausalLM(Qwen3_5MoeForCausalLM):
        # AutoModel registration requires the model's config_class to be the
        # registered config class, so the model gets a Voicing subclass too.
        config_class = VoicingConvoConfig

    try:
        AutoConfig.register(MODEL_TYPE, VoicingConvoConfig, exist_ok=True)
    except TypeError:  # older transformers without exist_ok
        AutoConfig.register(MODEL_TYPE, VoicingConvoConfig)
    AutoModelForCausalLM.register(VoicingConvoConfig, VoicingConvoForCausalLM, exist_ok=True)
    REGISTERED["transformers"] = True

# --------------------------------------------------------------------------- #
# SGLang
# --------------------------------------------------------------------------- #
if _installed("sglang"):
    from typing import Optional

    import torch
    from torch import nn

    from sglang.srt.configs.qwen3_5 import Qwen3_5MoeTextConfig as _SglTextConfig
    from sglang.srt.distributed import get_pp_group as _sgl_get_pp_group
    from sglang.srt.layers.logits_processor import LogitsProcessor as _SglLogitsProcessor
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead as _SglParallelLMHead
    from sglang.srt.models.qwen3_5 import Qwen3_5MoeForCausalLM as _SglInner
    from sglang.srt.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration as _SglVL
    from sglang.srt.models.registry import ModelRegistry as _SglRegistry
    from sglang.srt.utils.hf_transformers import common as _sgl_hf_common

    # Helpers whose module path moves between releases: take them from the
    # engine's own model module, which has already imported them.
    from sglang.srt.models import qwen3_5 as _sgl_q35

    _SglPPMissingLayer = _sgl_q35.PPMissingLayer
    _sgl_add_prefix = _sgl_q35.add_prefix
    _sgl_get_server_args = _sgl_q35.get_server_args

    class SglVoicingConvoForCausalLM(nn.Module):
        """Standalone text-only entry class for SGLang.

        SGLang's text model for this family (``Qwen3_5MoeForCausalLM``) is the
        *inner* module -- embeddings, layers, norm -- and is only ever used
        wrapped by the vision class, which owns ``lm_head`` and applies the
        logits processor. This wrapper does exactly that without the vision
        tower, mirroring SGLang's own standalone ``Qwen3NextForCausalLM``.
        Weight loading reuses the engine's own loader (bound to this module):
        it already expects ``model.*`` and ``lm_head.weight`` names and skips
        vision/MTP tensors.
        """

        packed_modules_mapping = _SglInner.packed_modules_mapping
        supported_lora_modules = getattr(_SglInner, "supported_lora_modules", [])

        def __init__(self, config, quant_config=None, prefix: str = "") -> None:
            super().__init__()
            self.config = config
            self.quant_config = quant_config
            self.pp_group = _sgl_get_pp_group()
            if quant_config is not None and hasattr(quant_config, "packed_modules_mapping"):
                quant_config.packed_modules_mapping = self.packed_modules_mapping
            self.model = _SglInner(config=config, quant_config=quant_config, prefix=_sgl_add_prefix("model", prefix))
            # the engine's loader reads these from the module it is bound to
            self.start_layer = getattr(self.model, "start_layer", getattr(self.model, "_start_layer", 0))
            self.end_layer = getattr(self.model, "end_layer", getattr(self.model, "_end_layer", config.num_hidden_layers))
            if self.pp_group.is_last_rank:
                if self.pp_group.world_size == 1 and getattr(config, "tie_word_embeddings", False):
                    self.lm_head = self.model.embed_tokens
                else:
                    self.lm_head = _SglParallelLMHead(
                        config.vocab_size,
                        config.hidden_size,
                        quant_config=quant_config,
                        use_attn_tp_group=_sgl_get_server_args().enable_dp_lm_head,
                        prefix=_sgl_add_prefix("lm_head", prefix),
                    )
            else:
                self.lm_head = _SglPPMissingLayer()
            self.logits_processor = _SglLogitsProcessor(config)
            self.capture_aux_hidden_states = False
            # The engine's loader reads these; mirror the vision wrapper's logic
            # (shared-expert fusion is an AMD/aiter-only path).
            self.num_fused_shared_experts = 0
            if getattr(_sgl_q35, "_use_aiter", False) and not _sgl_q35._disable_shared_experts_fusion():
                self.num_fused_shared_experts = _SglVL._get_num_fused_shared_experts(self)
            self.enable_shared_expert_fusion = self.num_fused_shared_experts > 0

        @torch.no_grad()
        def forward(self, input_ids, positions, forward_batch, input_embeds=None, inputs_embeds=None, **kwargs):
            embeds = input_embeds if input_embeds is not None else inputs_embeds
            hidden_states = self.model(input_ids, positions, forward_batch, input_embeds=embeds, **kwargs)
            if not self.pp_group.is_last_rank:
                return hidden_states
            aux_hidden_states = None
            if self.capture_aux_hidden_states:
                hidden_states, aux_hidden_states = hidden_states
            return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states)

        def load_weights(self, weights):
            # engine-owned loader, bound to this module: handles the fused-expert
            # layout, stacked projections, and lm_head routing for this family.
            return _SglVL.load_weights(self, weights)

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def get_embed_and_head(self):
            return self.model.embed_tokens.weight, self.lm_head.weight

        def set_embed_and_head(self, embed, head):
            del self.model.embed_tokens.weight
            del self.lm_head.weight
            self.model.embed_tokens.weight = embed
            self.lm_head.weight = head
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        def get_hidden_dim(self, module_name: str, layer_idx: int):
            return self.model.get_hidden_dim(module_name, layer_idx)

        @classmethod
        def get_model_config_for_expert_location(cls, config):
            fn = getattr(_SglInner, "get_model_config_for_expert_location", None)
            return fn(config) if fn is not None else None

        def set_eagle3_layers_to_capture(self, layer_ids: Optional[list] = None):
            if not self.pp_group.is_last_rank:
                return
            self.capture_aux_hidden_states = True
            self.model.set_eagle3_layers_to_capture(layer_ids)

    _SglModel = SglVoicingConvoForCausalLM

    class SglVoicingConvoConfig(_SglTextConfig):
        model_type = MODEL_TYPE

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _strip_mrope(self)
            self.model_type = ENGINE_FAMILY  # see ENGINE_FAMILY above

    # model_type -> config class. SGLang consults its own registry before
    # transformers' AutoConfig, so this is the only place it needs to be; the
    # AutoConfig mapping stays on the transformers subclass for transformers users.
    _sgl_hf_common._CONFIG_REGISTRY[MODEL_TYPE] = SglVoicingConvoConfig
    # architecture -> model class (the standalone wrapper above)
    _SglRegistry.models[ARCH] = _SglModel
    REGISTERED["sglang"] = True

# --------------------------------------------------------------------------- #
# vLLM
# --------------------------------------------------------------------------- #
if _installed("vllm"):
    from vllm.model_executor.models import ModelRegistry as _VllmRegistry
    from vllm.model_executor.models import config as _vllm_models_config
    from vllm.transformers_utils import config as _vllm_tu_config

    # Resolve vLLM's text config through its own registry rather than by module
    # path: the class name/module moved between releases, the registry key did not.
    _VllmTextConfig = _vllm_tu_config._CONFIG_REGISTRY["qwen3_5_moe_text"]

    class VllmVoicingConvoConfig(_VllmTextConfig):
        model_type = MODEL_TYPE

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _strip_mrope(self)
            self.model_type = ENGINE_FAMILY  # see ENGINE_FAMILY above

    # model_type -> config class
    _vllm_tu_config._CONFIG_REGISTRY[MODEL_TYPE] = VllmVoicingConvoConfig
    # architecture -> model class (string form: no CUDA init at import time)
    _VllmRegistry.register_model(ARCH, "vllm.model_executor.models.qwen3_5:Qwen3_5MoeForCausalLM")
    # per-architecture config hook: sets the Mamba cache dtype and strips M-RoPE
    _vllm_models_config.MODELS_CONFIG_MAP[ARCH] = _vllm_models_config.Qwen3_5ForCausalLMConfig
    # Triton/FLA kernel warm-up is keyed on model_type; opt in (performance only)
    try:
        from vllm.model_executor.warmup import qwen_triton_warmup as _w
        _w._QWEN_MODEL_TYPES = frozenset(_w._QWEN_MODEL_TYPES) | {MODEL_TYPE}
    except Exception as e:  # pragma: no cover
        log.debug("warm-up opt-in skipped: %s", e)
    REGISTERED["vllm"] = True

_REGISTERED = REGISTERED  # backwards-compatible alias
