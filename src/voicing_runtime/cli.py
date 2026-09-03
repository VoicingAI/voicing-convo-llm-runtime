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

    # Fail clearly on a busy port. Otherwise vLLM reports "Address already in
    # use" from a worker traceback, and SGLang binds, then fails its own warmup
    # request against whatever already owns the port -- typically an
    # AssertionError with a 401, which points nowhere near the real cause.
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((args.host if args.host != "0.0.0.0" else "", int(args.port)))
        except OSError:
            owner = ""
            try:
                import subprocess
                out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, timeout=5).stdout
                for line in out.splitlines():
                    if f":{args.port} " in line:
                        owner = line.split("users:")[-1].strip() if "users:" in line else line.strip()
                        break
            except Exception:
                pass
            sys.exit(f"error: port {args.port} is already in use{' by ' + owner if owner else ''}.\n"
                     f"       Pick another with --port, or stop whatever is listening.")

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
    p.add_argument("--verify-weights", action="store_true",
                   help="also sha256 every shard against the Hub's checksums (slow; needs HF_TOKEN)")
    args = p.parse_args(argv)

    from .register import ARCH, MODEL_TYPE, register as do_register
    done = do_register()
    engines = [k for k, v in done.items() if v]
    print(f"vtext-editor {__import__('voicing_runtime').__version__}")
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
        md = os.path.abspath(args.model)
        cfg = json.load(open(os.path.join(md, "config.json")))
        good = cfg.get("architectures") == [ARCH] and cfg.get("model_type") == MODEL_TYPE
        print(f"  model:  {md} -> {cfg.get('architectures')} / {cfg.get('model_type')} {'ok' if good else 'MISMATCH'}")
        ok &= good

        # Completeness. An interrupted download leaves a directory that looks
        # nearly right; the index names every shard that must be present.
        idx_path = os.path.join(md, "model.safetensors.index.json")
        if os.path.isfile(idx_path):
            want = sorted(set(json.load(open(idx_path))["weight_map"].values()))
            missing = [f for f in want if not os.path.isfile(os.path.join(md, f))]
            # A half-written shard is present but short. Every safetensors file
            # declares its own length in its header, so truncation is detectable
            # locally, without the network.
            empty = [f for f in want if f not in missing
                     and _truncated(os.path.join(md, f))]
            partial = []
            dl = os.path.join(md, ".cache", "huggingface", "download")
            if os.path.isdir(dl):
                partial = [f for f in os.listdir(dl) if f.endswith(".incomplete")]
            # Leftover .incomplete files are stale artifacts of an interrupted
            # download. They waste disk but say nothing about the shards that
            # actually landed, so they must not fail the check on their own.
            status = "ok" if not (missing or empty) else "INCOMPLETE"
            good = len(want) - len(missing) - len(empty)
            print(f"  shards: {good}/{len(want)} complete {status}")
            for f in missing[:5]:
                print(f"            missing: {f}")
            if len(missing) > 5:
                print(f"            ... and {len(missing) - 5} more")
            for f in empty[:5]:
                print(f"            truncated: {f}")
            if missing or empty:
                for f in partial[:3]:
                    print(f"            still downloading: {f}")
                print("            re-run the download loop in SETUP.md step 2")
            elif partial:
                waste = sum(os.path.getsize(os.path.join(dl, f)) for f in partial) / 2**30
                print(f"            note: {len(partial)} stale .incomplete file(s) "
                      f"({waste:.1f} GiB) left in {os.path.relpath(dl, md)}; safe to delete")
            ok &= not (missing or empty)

            if args.verify_weights and not missing:
                ok &= _verify_weights(md, want)
        else:
            print("  shards: no model.safetensors.index.json; cannot check completeness")
            ok = False
    else:
        print("  model:  (not checked; pass a model directory or set VOICING_MODEL_DIR)")

    print("\nOK" if ok else "\nFAILED")
    return 0 if ok else 1


def _truncated(path: str) -> bool:
    """True if a safetensors file is shorter than its own header says it is."""
    import json
    import struct

    try:
        size = os.path.getsize(path)
        if size < 8:
            return True
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            if size < 8 + n:
                return True
            header = json.loads(fh.read(n))
        end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"),
                  default=0)
        return size < 8 + n + end
    except Exception:
        return True  # unreadable or malformed counts as not usable


def _verify_weights(model_dir: str, shards: list[str]) -> bool:
    """sha256 every shard against the Hub's published checksums."""
    import hashlib

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("  weights: huggingface_hub not installed; cannot verify")
        return False
    token = os.environ.get("HF_TOKEN")
    try:
        info = HfApi(token=token).model_info(DEFAULTS["served_name"], files_metadata=True)
    except Exception as e:
        print(f"  weights: could not reach the Hub ({type(e).__name__}); set HF_TOKEN")
        return False
    hub = {s.rfilename: s.lfs.sha256 for s in info.siblings if s.lfs}

    bad, checked = [], 0
    for name in shards:
        want = hub.get(name)
        if want is None:
            continue
        h = hashlib.sha256()
        with open(os.path.join(model_dir, name), "rb") as fh:
            for chunk in iter(lambda: fh.read(64 << 20), b""):
                h.update(chunk)
        checked += 1
        if h.hexdigest() != want:
            bad.append(name)
        if sys.stdout.isatty():
            print(f"\r  weights: {checked}/{len(shards)} verified", end="", flush=True)
    if sys.stdout.isatty():
        print()
    print(f"  weights: {checked}/{len(shards)} verified against the Hub")
    if bad:
        for f in bad:
            print(f"            CORRUPT: {f}")
        print("            delete those files and re-run the download loop")
    return not bad


if __name__ == "__main__":
    raise SystemExit(serve(sys.argv[1:]))
