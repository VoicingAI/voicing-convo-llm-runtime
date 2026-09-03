# SPDX-License-Identifier: Apache-2.0
"""Register the Voicing-Convo-V2 model and parsers at interpreter startup.

Put this directory on PYTHONPATH and every Python process -- servers and the
workers they spawn -- registers:

  * the model:   architectures "VoicingConvoForCausalLM" / model_type
                 "voicing_convo" for transformers, SGLang, and vLLM
  * the parsers: "--reasoning-parser voicing" / "--tool-call-parser voicing"
                 for SGLang (vLLM loads its parsers by file path)

    PYTHONPATH=/opt/voicing-serving-runtime/voicing_runtime \
      python3 -m sglang.launch_server --model /models/Voicing-Convo-V2-35B-MOE ...

This package is deployed alongside the serving engine, never inside the model
repository. Nothing is registered if the corresponding engine is not installed,
so unrelated python3 invocations are unaffected. Any other failure is printed,
never hidden.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARSERS_SGLANG = os.path.join(os.path.dirname(_HERE), "voicing_parsers", "sglang")
for _p in (_HERE, _PARSERS_SGLANG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Chain to any other sitecustomize further along sys.path rather than shadowing it.
_self = sys.modules.pop("sitecustomize", None)
try:
    import importlib.machinery
    import importlib.util

    _rest = [p for p in sys.path if os.path.abspath(p) not in (_HERE, _PARSERS_SGLANG)]
    _spec = importlib.machinery.PathFinder.find_spec("sitecustomize", _rest)
    if _spec is not None:
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        sys.modules["sitecustomize"] = _mod
except Exception:
    pass
finally:
    if _self is not None:
        sys.modules.setdefault("sitecustomize", _self)


def _loud(what, exc):
    import traceback

    sys.stderr.write(f"[voicing] failed to register {what}; the model will not load by its Voicing name:\n")
    traceback.print_exception(exc)


try:
    import voicing_convo  # noqa: F401  (model registration for installed engines)
except Exception as _e:
    _loud("the Voicing-Convo model", _e)

import importlib.util as _ilu

if _ilu.find_spec("sglang") is not None:
    try:
        import voicing_reasoning_detector  # noqa: F401
        import voicing_tool_detector  # noqa: F401
    except Exception as _e:
        _loud("the SGLang parsers", _e)
