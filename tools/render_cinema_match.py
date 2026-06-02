from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLONED_REPO = ROOT / ".slim" / "clonedeps" / "repos" / "MatthewWHuang__orbit-wars"
VISUALIZE_PATH = CLONED_REPO / "visualize.py"


def _load_visualize_module() -> ModuleType:
    if not VISUALIZE_PATH.exists():
        raise FileNotFoundError(f"visualize.py not found at expected path: {VISUALIZE_PATH}")
    spec = importlib.util.spec_from_file_location("ow_clone_visualize", VISUALIZE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Unable to load module spec for visualize.py: {VISUALIZE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def render_env_json_to_cinema_html(env_json: Any, output: str | Path, bot_names: list[str] | None = None) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    visualize = _load_visualize_module()
    visualize.write_html(env_json, str(out), bot_names=bot_names, auto_cinema=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    try:
        kaggle_env = importlib.import_module("kaggle_environments")
    except ModuleNotFoundError as exc:
        raise SystemExit("kaggle_environments is required") from exc

    config = {"seed": int(args.seed)}
    if args.steps is not None:
        config["episodeSteps"] = int(args.steps)
    env = kaggle_env.make("orbit_wars", configuration=config)
    env.run([args.candidate, args.baseline])

    output = render_env_json_to_cinema_html(env, args.out, bot_names=[args.candidate, args.baseline])
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
