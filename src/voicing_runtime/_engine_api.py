# SPDX-License-Identifier: Apache-2.0
"""Version-tolerant lookup of engine internals.

SGLang and vLLM move and rename the private helpers this package borrows. Keep
that tolerance here, in a module with no torch or engine imports, so it can be
unit-tested against synthetic release layouts on any machine.
"""

from __future__ import annotations

import importlib
import inspect
import logging

log = logging.getLogger("voicing_convo")


def _pick(engine, what, *candidates):
    """Resolve an engine internal that moves or is renamed between releases.

    Each candidate is ``(module, attribute)``, where ``module`` is either an
    already-imported module object or a dotted path imported on demand. The
    first hit wins.

    None of these are public API, and the engines reshuffle them between point
    releases. Verified by probing both installs: SGLang 0.5.16's ``qwen3_5``
    re-exports the server-args accessor as ``get_server_args``, 0.5.15's exports
    it under neither name, and in both the canonical ``sglang.srt.server_args``
    calls it ``get_global_server_args``. ``add_prefix`` and ``PPMissingLayer``
    have likewise moved, and the HF config registry gained a second home at
    ``hf_transformers.common`` alongside ``hf_transformers_utils``.

    Binding to a single location is what broke this plugin on SGLang 0.5.15, so
    try every home we have seen instead of assuming one.

    Candidates inside the engine's own model module should come first, so we
    match whatever that release's model code itself uses. On total failure raise
    ImportError naming every place we looked: an unsupported engine version
    should say so, not surface a bare AttributeError from deep inside an import.
    """
    tried = []
    for module, attr in candidates:
        if isinstance(module, str):
            try:
                module = importlib.import_module(module)
            except ImportError:
                tried.append("%s.%s (no such module)" % (module, attr))
                continue
        found = getattr(module, attr, None)
        if found is not None:
            return found
        tried.append("%s.%s" % (module.__name__, attr))
    raise ImportError(
        "vtext-editor could not locate %s's %s in the installed "
        "%s %s. Looked in: %s. This package is verified against SGLang "
        "0.5.15-0.5.16 and vLLM 0.28.0; on another release, report this version "
        "so the lookup can be extended."
        % (engine, what, engine, _engine_version(engine), ", ".join(tried))
    )


def _engine_version(engine: str) -> str:
    try:
        return getattr(importlib.import_module(engine), "__version__", "unknown")
    except ImportError:
        return "not installed"


def _accepted_kwargs(func, kwargs):
    """Drop keyword arguments ``func`` does not accept.

    Engine base classes gain keyword arguments between releases, and passing one
    to an older base raises TypeError. SGLang 0.5.16 added
    ``force_nonempty_content`` to ``BaseReasoningFormatDetector.__init__``;
    forwarding it on 0.5.15 aborted the reasoning detector -- and with it server
    startup -- with:

        TypeError: BaseReasoningFormatDetector.__init__() got an unexpected
        keyword argument 'force_nonempty_content'

    A parameter the older base does not define is a parameter it has no
    behaviour for, so dropping it restores that release's own default rather
    than changing semantics. Anything dropped is logged, so a silent difference
    is still a visible one.

    A signature taking ``**kwargs`` accepts everything; a callable we cannot
    introspect (a C builtin) is passed through untouched.
    """
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    keep = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(keep))
    if dropped:
        log.info(
            "%s does not accept %s on this engine release; using its defaults",
            getattr(func, "__qualname__", func), ", ".join(dropped),
        )
    return keep
