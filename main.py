import os
import sys
import traceback

from src.ow.chadman import agent as _CHADMAN_AGENT


def agent(obs, config=None):
    try:
        return _CHADMAN_AGENT(obs, config)
    except Exception:
        if os.environ.get("OW_AGENT_DEBUG"):
            print("Chadman agent exception:", file=sys.stderr)
            traceback.print_exc()
        return []
