# SPDX-License-Identifier: Apache-2.0
"""Console commands installed with this package.

    voicing-serve sglang --model /models/Voicing-Convo-V2-35B-MOE
    voicing-serve vllm   --model /models/Voicing-Convo-V2-35B-MOE
    voicing-check /models/Voicing-Convo-V2-35B-MOE

`voicing-serve` builds a sensible default command for the chosen engine and
execs it. Anything after the known options is passed straight through, and any
flag you supply yourself overrides the default of the same name.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

DEFAULTS = {
    "served_name": "voicing-ai/Voicing-Convo-V2-35B-MOE",
    "host": "0.0.0.0",
    "port": "8000",
    "tp": "1",
    "max_len": "65536",
    "mem": "0.90",
}


def _base(argv: list[str] | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voicing-serve", add_help=True,
                                description="Serve Voicing-Convo-V2-35B-MOE on vLLM or SGLang.")
    p.add_argument("engine", choices=["sglang", "vllm"], help="inference engine to launch")
    p.add_argument("--model", "--model-path", dest="model", default=os.environ.get("VOICING_MODEL_DIR"),
                   help="model directory (default: $VOICING_MODEL_DIR)")
    p.add_argument("--served-model-name", dest="served_name", default=DEFAULTS["served_name"])
    p.add_argument("--host", default=DEFAULTS["host"])
    p.add_argument("--port", default=DEFAULTS["port"])
    p.add_argument("--tp", "--tensor-parallel-size", dest="tp", default=DEFAULTS["tp"])
    p.add_argument("--max-model-len", "--context-length", dest="max_len", default=DEFAULTS["max_len"])
    p.add_argument("--gpu-memory-utilization", "--mem-fraction-static", dest="mem", default=DEFAULTS["mem"])
    p.add_argument("--max-num-seqs", dest="max_num_seqs", default="256",
                   help="vLLM only; the default of 1024 does not fit this architecture's Mamba cache")
    p.add_argument("--attention-backend", dest="attn", default="flashinfer", help="SGLang only")
    p.add_argument("--dry-run", action="store_true", help="print the command instead of running it")
    return p


def serve(argv: list[str] | None = None) -> int:
    args, passthrough = _base().parse_known_args(argv)
    if not args.model:
        sys.exit("error: pass --model or set VOICING_MODEL_DIR")
    model = os.path.abspath(args.model)
    if not os.path.isfile(os.path.join(model, "config.json")):
        sys.exit(f"error: no config.json in {model}")

    if args.engine == "sglang":
        cmd = [sys.executable, "-m", "sglang.launch_server",
               "--model", model, "--served-model-name", args.served_name,
               "--host", args.host, "--port", str(args.port),
               "--tp", str(args.tp), "--dtype", "bfloat16",
               "--mem-fraction-static", str(args.mem),
               "--attention-backend", args.attn, "--sampling-backend", "pytorch",
               "--reasoning-parser", "voicing", "--tool-call-parser", "voicing",
               "--context-length", str(args.max_len), "--chunked-prefill-size", "4096"]
    else:
        vllm = shutil.which("vllm")
        if not vllm:
            sys.exit("error: 'vllm' not found on PATH; install vLLM in this environment")
        cmd = [vllm, "serve", model, "--served-model-name", args.served_name,
               "--host", args.host, "--port", str(args.port),
               "--tensor-parallel-size", str(args.tp), "--dtype", "bfloat16",
               "--gpu-memory-utilization", str(args.mem),
               "--max-model-len", str(args.max_len), "--max-num-seqs", str(args.max_num_seqs),
               "--reasoning-parser", "voicing",
               "--enable-auto-tool-choice", "--tool-call-parser", "voicing"]

    # user-supplied flags win over our defaults
    supplied = {a for a in passthrough if a.startswith("--")}
    out, skip = [], False
    for i, tok in enumerate(cmd):
        if skip:
            skip = False
            continue
        if tok.startswith("--") and tok in supplied:
            skip = i + 1 < len(cmd) and not cmd[i + 1].startswith("--")
            continue
        out.append(tok)
    cmd = out + passthrough

    if args.dry_run:
        print(" ".join(cmd))
        return 0
    print("+ " + " ".join(cmd), flush=True)
    os.execv(cmd[0], cmd)


def check(argv: list[str] | None = None) -> int:
    """Verify the runtime registers correctly against the installed engine."""
    p = argparse.ArgumentParser(prog="voicing-check",
                                description="Check that this runtime registers with the installed engine.")
    p.add_argument("model", nargs="?", default=os.environ.get("VOICING_MODEL_DIR"),
                   help="model directory (default: $VOICING_MODEL_DIR)")
    args = p.parse_args(argv)

    from .register import ARCH, MODEL_TYPE, register as do_register
    done = do_register()
    engines = [k for k, v in done.items() if v]
    print(f"voicing-convo-llm-runtime {__import__('voicing_runtime').__version__}")
    print(f"  registered for: {', '.join(engines) if engines else 'nothing'}")

    if not any(done.get(e) for e in ("sglang", "vllm")):
        print("\n  no inference engine found in this environment.")
        print("  Install vLLM or SGLang first, then install this package into the same environment.")
        print("  (uv-created venvs have no pip: use 'uv pip install' instead.)")
        print("\nINCOMPLETE")
        return 1

    ok = True
    if done.get("sglang"):
        from sglang.srt.models.registry import ModelRegistry
        from sglang.srt.parser.reasoning_parser import ReasoningParser
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        a = ARCH in ModelRegistry.models
        r = "voicing" in ReasoningParser.DetectorMap
        t = "voicing" in FunctionCallParser.ToolCallParserEnum
        print(f"  sglang: architecture={'ok' if a else 'MISSING'} reasoning-parser={'ok' if r else 'MISSING'} tool-call-parser={'ok' if t else 'MISSING'}")
        ok &= a and r and t
    if done.get("vllm"):
        from vllm.model_executor.models import ModelRegistry as VR
        from vllm.reasoning import ReasoningParserManager
        from vllm.tool_parsers import ToolParserManager
        a = VR._try_load_model_cls(ARCH) is not None
        r = "voicing" in ReasoningParserManager.list_registered()
        t = "voicing" in ToolParserManager.list_registered()
        print(f"  vllm:   architecture={'ok' if a else 'MISSING'} reasoning-parser={'ok' if r else 'MISSING'} tool-call-parser={'ok' if t else 'MISSING'}")
        ok &= a and r and t

    if args.model:
        import json
        cfg = json.load(open(os.path.join(os.path.abspath(args.model), "config.json")))
        good = cfg.get("architectures") == [ARCH] and cfg.get("model_type") == MODEL_TYPE
        print(f"  model:  {args.model} -> {cfg.get('architectures')} / {cfg.get('model_type')} {'ok' if good else 'MISMATCH'}")
        ok &= good
    else:
        print("  model:  (not checked; pass a model directory or set VOICING_MODEL_DIR)")

    print("\nOK" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(serve(sys.argv[1:]))
