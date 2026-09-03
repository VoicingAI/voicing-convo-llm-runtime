#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch an SGLang server with the Voicing parsers registered.

Prefer the PYTHONPATH route -- it needs no wrapper and covers every process:

    PYTHONPATH=/path/to/voicing_parsers/sglang \
    python3 -m sglang.launch_server --model <model> \
      --reasoning-parser voicing --tool-call-parser voicing

This script is a convenience equivalent for setups that cannot set an env var.
It accepts every flag `python -m sglang.launch_server` accepts.

The detector imports are at module scope on purpose: SGLang spawns worker
processes, and `multiprocessing` spawn re-imports this file as `__mp_main__`
without running `main()`. Registering inside `main()` would leave those workers
without the parsers.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Imported for side effects; each registers itself into the registry that the
# corresponding CLI flag validates against. Module scope, not inside main() --
# see the note above about spawned workers.
import voicing_reasoning_detector  # noqa: F401,E402
import voicing_tool_detector  # noqa: F401,E402


def main() -> None:
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    load_plugins()
    server_args = prepare_server_args(sys.argv[1:])
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
