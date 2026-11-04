"""v2.1 launch: measured-connectome brain controls the body automatically by default."""

import argparse
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    from flymaze_v2 import __version__
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--check", action="store_true", help="check platform, dependencies and selected brain without opening a GUI")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--manual", action="store_true", help="explicitly switch to keyboard control")
    modes.add_argument("--baseline", action="store_true", help="use bundled GRU baseline instead of connectome")
    parser.add_argument("--model", type=Path, help="alternative checkpoint; must match selected controller kind")
    maps = parser.add_mutually_exclusive_group()
    maps.add_argument("--difficulty", choices=("easy", "medium", "hard"))
    maps.add_argument("--maze", type=Path)
    parser.add_argument("--vary-maze", action="store_true")
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=10007)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--delay", type=float, default=None, help="GUI wall-clock interval, not simulated dt")
    parser.add_argument("--headless", action="store_true", help="run same feedback loop without a GUI")
    parser.add_argument("--episodes", type=int, default=1, help="headless episode count")
    parser.add_argument("--report", type=Path, help="headless JSON output (existing files are never overwritten)")
    args = parser.parse_args(argv)
    if args.manual and (args.model or args.headless or args.report):
        parser.error("manual mode cannot use model/headless/report")
    if args.episodes < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("episodes and max-steps must be positive")
    if not args.headless and (args.report is not None or args.episodes != 1):
        parser.error("--report and multiple episodes require --headless")
    if args.delay is not None and (not math.isfinite(args.delay) or args.delay < 0):
        parser.error("delay must be finite and nonnegative")
    try:
        from flymaze_v2.platform_support import validate_platform
        platform_info = validate_platform()
        import torch
        from flymaze_v2.environment import ContinuousMaze, from_config
        from flymaze_v2.runtime import ControlSession, load_controller, default_controller_path
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        identity, brain, checkpoint = {}, None, None
        if args.manual:
            layout = args.maze.read_text(encoding="utf-8-sig").splitlines() if args.maze else None
            env = ContinuousMaze(layout, difficulty=args.difficulty, vary_maze=args.vary_maze,
                                 seed=args.seed, max_steps=args.max_steps or 800)
        else:
            kind = "gru" if args.baseline else "connectome"
            model = args.model if args.model else default_controller_path(ROOT/"models", kind)
            if not model.is_file():
                raise FileNotFoundError(f"missing brain checkpoint: {model}; extract the complete v2.1 package including models/")
            brain, checkpoint, identity = load_controller(model, expected_kind=kind)
            env = from_config(checkpoint["environment"], seed=args.seed, path=args.maze,
                              difficulty=args.difficulty, vary_maze=args.vary_maze, max_steps=args.max_steps)
            training = checkpoint["training"]
            progress = (f"guided updates={training.get('selected_update', '?')}/{training.get('updates', '?')}"
                        if training.get("algorithm") == "recurrent_behavior_cloning_dagger"
                        else f"training episodes={training.get('episodes', 'unknown')}")
            print(f"Fly Maze {__version__}: AUTO {kind}; {brain.hidden_size} recurrent states; {progress}", flush=True)
            print(f"Loaded model: {model.resolve()}", flush=True)
            print(f"Training algorithm: {checkpoint['training'].get('algorithm', 'unknown')}", flush=True)
            if checkpoint["training"].get("algorithm") == "recurrent_behavior_cloning_dagger" and (args.maze or args.difficulty):
                print("This guided model was validated on its fixed training map; performance on new maps is unverified.", flush=True)
            print("Every motor command comes from brain.act. Playback does not train the model.", flush=True)
        if args.check:
            import numpy
            try:
                import tkinter
                tkinter_available = True
            except ImportError:
                tkinter_available = False
            print(json.dumps({"platform": platform_info, "torch": torch.__version__, "numpy": numpy.__version__,
                              "tkinter_importable": tkinter_available, "controller": brain.kind if brain else "manual",
                              "recurrent_states": brain.hidden_size if brain else None,
                              "note": "Tk import does not prove that a desktop display is available."}, indent=2))
            return 0
        if args.headless:
            stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
            target = args.report or ROOT/"outputs"/"v2_1_sessions"/f"headless_{stamp}.json"
            if target.exists():
                raise FileExistsError(f"report already exists: {target}")
            session = ControlSession(env, brain, sample=args.sample, identity=identity)
            reports = []
            for index in range(args.episodes):
                torch.manual_seed(args.seed+index)
                session.reset(args.random_start)
                while not (env.terminated or env.truncated):
                    session.step()
                reports.append(session.report())
                print(f"episode={index+1} success={env.terminated} steps={env.steps} "
                      f"distance={env.path_length:.3f} collisions={env.collisions}", flush=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as handle:
                json.dump({"project_version": __version__, "seed": args.seed, "episodes": reports},
                          handle, ensure_ascii=False, indent=2, allow_nan=False)
            print(f"Brain/body trace: {target}", flush=True)
        else:
            from flymaze_v2.viewer import Viewer
            Viewer(env, brain, random_start=args.random_start, sample=args.sample, delay=args.delay,
                   identity=identity, log_dir=ROOT/"outputs"/"v2_1_sessions").run()
    except ImportError as error:
        print(f"Missing Python dependency: {error}\nUse the project .venv, or install requirements.txt first.", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
