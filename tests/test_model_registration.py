#!/usr/bin/env python3
"""Self-check that Voicing-Convo-V2-35B-MOE loads by its own name on this machine.

    python tests/test_model_registration.py /path/to/Voicing-Convo-V2-35B-MOE

The checkpoint declares `architectures: ["VoicingConvoForCausalLM"]` and
`model_type: "voicing_convo"`. The installed `voicing-convo-llm-runtime` package
registers those names with whichever of transformers, SGLang, and vLLM is
present. This test proves the registration on the installed versions, without
loading weights:

  * transformers: AutoConfig -> VoicingConvoConfig, AutoModelForCausalLM ->
    VoicingConvoForCausalLM (meta device), and its parameter names match the
    checkpoint index exactly
  * SGLang: its config loader returns the registered class, the hybrid
    Gated-DeltaNet cache path recognises it, the architecture resolves, the
    model is text-only, and the Voicing parsers are still registered
  * vLLM: config + architecture resolve, hybrid detected, text-only, and the
    per-architecture config hook is wired
  (the SGLang/vLLM config objects report the engine's internal family id at
   runtime -- that is how layer dispatch works without editing engine source;
   the checkpoint files themselves carry only the Voicing names)
  * no config file in the repo mentions the upstream vendor
Exits non-zero on any failure.
"""

import importlib.util
import json
import os
import re
import sys
import traceback

MODEL_DIR = os.path.abspath(
    sys.argv[1] if len(sys.argv) > 1
    else os.environ.get("VOICING_MODEL_DIR")
    or sys.exit("error: pass the model directory as an argument or set VOICING_MODEL_DIR")
)
ARCH, MODEL_TYPE = "VoicingConvoForCausalLM", "voicing_convo"
results = []


def check(name, fn):
    try:
        info = fn() or ""
        results.append((True, name, info))
    except (Exception, SystemExit) as e:
        tb = traceback.extract_tb(e.__traceback__)
        results.append((False, name, f"{type(e).__name__}: {e} (line {tb[-1].lineno})"))


def have(mod):
    return importlib.util.find_spec(mod) is not None


def main():
    # Call the same entry point the engines call through their plugin systems.
    import voicing_runtime
    import voicing_runtime.model as voicing_convo
    done = voicing_runtime.register()
    print("registered for:", [k for k, v in done.items() if v])

    def _no_vendor_strings():
        hits = []
        for f in ("config.json", "tokenizer_config.json", "generation_config.json"):
            t = open(os.path.join(MODEL_DIR, f)).read()
            t = json.dumps({k: v for k, v in json.loads(t).items() if k != "chat_template"})
            hits += [f"{f}:{m}" for m in re.findall(r"(?i)qwen|alibaba", t)]
        for f in ("preprocessor_config.json", "video_preprocessor_config.json"):
            assert not os.path.exists(os.path.join(MODEL_DIR, f)), f"{f} should not exist in a text-only checkpoint"
        assert not hits, hits
        cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
        assert cfg["architectures"] == [ARCH] and cfg["model_type"] == MODEL_TYPE, (cfg["architectures"], cfg["model_type"])
        return f"{cfg['architectures'][0]} / {cfg['model_type']}"
    check("config files carry only Voicing identifiers", _no_vendor_strings)

    def _transformers():
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        c = AutoConfig.from_pretrained(MODEL_DIR)
        assert type(c).__name__ == "VoicingConvoConfig" and c.model_type == MODEL_TYPE, type(c).__name__
        with torch.device("meta"):
            m = AutoModelForCausalLM.from_config(c)
        assert type(m).__name__ == "VoicingConvoForCausalLM", type(m).__name__
        idx = set(json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))["weight_map"])
        keys = set(m.state_dict().keys())
        assert keys == idx, f"missing={sorted(keys - idx)[:3]} extra={sorted(idx - keys)[:3]}"
        t = AutoTokenizer.from_pretrained(MODEL_DIR)
        assert "qwen" not in type(t).__name__.lower(), type(t).__name__
        return f"{type(m).__name__}, {sum(p.numel() for p in m.parameters())/1e9:.2f}B params, {len(keys)} tensors match index"
    check("transformers: AutoConfig/AutoModelForCausalLM resolve; keys match checkpoint", _transformers)

    if have("sglang"):
        def _sglang():
            from sglang.srt.utils.hf_transformers.config import get_config
            from sglang.srt.configs.qwen3_next import Qwen3NextConfig
            from sglang.srt.configs.hybrid_arch import hybrid_gdn_config
            from sglang.srt.configs.model_config import is_multimodal_model
            from sglang.srt.models.registry import ModelRegistry
            c = get_config(MODEL_DIR, trust_remote_code=False)
            assert type(c).__name__ == "SglVoicingConvoConfig" and isinstance(c, Qwen3NextConfig), type(c).__name__
            assert c.model_type == voicing_convo.ENGINE_FAMILY, c.model_type  # engine-facing family id
            class MC: pass
            mc = MC(); mc.hf_config = c; mc.is_draft_model = False
            assert hybrid_gdn_config(mc) is not None, "hybrid GDN cache path not recognised"
            assert not any(k in c.rope_parameters for k in ("mrope_section", "mrope_interleaved"))
            assert ARCH in ModelRegistry.models, "architecture not in the SGLang registry"
            cls = ModelRegistry.resolve_model_cls([ARCH]); cls = cls[0] if isinstance(cls, tuple) else cls
            # SGLang does NOT raise on an unknown architecture: it silently falls
            # back to its generic transformers backend. Catch that explicitly.
            assert "Transformers" not in cls.__name__, (
                f"resolved to the generic fallback {cls.__name__}; the runtime did not register")
            assert not is_multimodal_model([ARCH])
            import argparse
            from sglang.srt.server_args import ServerArgs
            p = argparse.ArgumentParser(); ServerArgs.add_cli_args(p)
            ns = p.parse_args(["--model-path", "x", "--reasoning-parser", "voicing", "--tool-call-parser", "voicing"])
            return f"{type(c).__name__} -> {cls.__name__}, hybrid on, text-only, parsers {ns.reasoning_parser}/{ns.tool_call_parser}"
        check("SGLang: config, hybrid path, architecture, text-only, parsers", _sglang)
    else:
        print("  (SGLang not installed; skipped)")

    if have("vllm"):
        def _vllm():
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            from vllm.transformers_utils.config import get_config
            from vllm.model_executor.models import ModelRegistry
            from vllm.model_executor.models.interfaces import supports_multimodal, is_hybrid
            from vllm.model_executor.models.config import MODELS_CONFIG_MAP
            c = get_config(MODEL_DIR, trust_remote_code=False)
            assert type(c).__name__ == "VllmVoicingConvoConfig", type(c).__name__
            assert c.model_type == voicing_convo.ENGINE_FAMILY, c.model_type  # engine-facing family id
            cls = ModelRegistry._try_load_model_cls(ARCH)
            assert cls is not None and is_hybrid(cls) and not supports_multimodal(cls)
            assert ARCH in MODELS_CONFIG_MAP
            return f"{type(c).__name__} -> {cls.__name__}, hybrid on, text-only, config hook {MODELS_CONFIG_MAP[ARCH].__name__}"
        check("vLLM: config, architecture, hybrid, text-only, config hook", _vllm)
    else:
        print("  (vLLM not installed; skipped)")

    passed = sum(1 for ok, _, _ in results if ok)
    for ok, name, info in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {info}" if info else ""))
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
