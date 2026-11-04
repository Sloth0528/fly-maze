"""Fly Maze v2.1: continuous environment and measured-connectome experiments."""

import argparse
import math
from pathlib import Path
import sys

from . import __version__
from .environment import ContinuousMaze, from_config


def add_maps(parser):
    maps = parser.add_mutually_exclusive_group()
    maps.add_argument("--maze", type=Path)
    maps.add_argument("--difficulty", choices=("easy", "medium", "hard"))
    parser.add_argument("--vary-maze", action="store_true")
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int)


def create_env(args):
    layout = args.maze.read_text(encoding="utf-8-sig").splitlines() if args.maze else None
    return ContinuousMaze(layout, difficulty=args.difficulty, vary_maze=args.vary_maze,
                          seed=7 if args.seed is None else args.seed,
                          max_steps=args.max_steps or (240 if args.command == "train-guided" else 800),
                          dt=args.dt, speed=args.speed, turn_rate=math.radians(args.turn_rate),
                          radius=args.radius, reward_mode=args.reward_mode)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-connectome", help="download measured FlyWire v783 and extract regional subgraph")
    prepare.add_argument("--output", type=Path, default=Path("data/flywire_cx_512.npz"))
    prepare.add_argument("--source", type=Path, help="use an existing official connections CSV(.gz)")
    prepare.add_argument("--cache", type=Path, default=Path("data/raw/connections.csv.gz"))
    prepare.add_argument("--nodes", type=int, default=512)
    prepare.add_argument("--regions", nargs="+", default=["EB", "FB", "PB", "NO"])

    train = commands.add_parser("train", help="train GRU or measured-connectome policy")
    train.add_argument("--brain", choices=("gru", "connectome"), default="gru")
    train.add_argument("--connectome", type=Path, default=Path("data/flywire_cx_512.npz"))
    train.add_argument("--hidden-size", type=int, default=64, help="GRU only; connectome size comes from the data")
    train.add_argument("--episodes", type=int, default=1000)
    train.add_argument("--lr", type=float, default=0.0003)
    train.add_argument("--gamma", type=float, default=0.99)
    train.add_argument("--rollout", type=int, default=64)
    train.add_argument("--output", type=Path, required=True)

    guided = commands.add_parser("train-guided", help="fixed-map demonstration and DAgger training; teacher is training-only")
    guided.add_argument("--brain", choices=("gru", "connectome"), default="connectome")
    guided.add_argument("--connectome", type=Path, default=Path("data/flywire_cx_512.npz"))
    guided.add_argument("--hidden-size", type=int, default=64)
    guided.add_argument("--updates", type=int, default=300)
    guided.add_argument("--demonstrations", type=int, default=48)
    guided.add_argument("--batch-size", type=int, default=8)
    guided.add_argument("--lr", type=float, default=0.001)
    guided.add_argument("--validation-seed", type=int, default=20021)
    guided.add_argument("--output", type=Path, required=True)

    preview = commands.add_parser("preview", help="manual continuous steering or checkpoint playback")
    preview.add_argument("--model", type=Path)
    preview.add_argument("--sample", action="store_true")
    preview.add_argument("--delay", type=float, help="display interval in seconds; simulated dt stays fixed")

    for command in (train, guided, preview):
        add_maps(command)
        command.add_argument("--dt", type=float, default=0.1)
        command.add_argument("--speed", type=float, default=1.5, help="cell units per simulated second")
        command.add_argument("--turn-rate", type=float, default=180.0, help="degrees per simulated second")
        command.add_argument("--radius", type=float, default=0.16)
        command.add_argument("--reward-mode", choices=("shaped", "sparse"), default="shaped")

    evaluate = commands.add_parser("evaluate", help="evaluate one model without updating weights")
    evaluate.add_argument("--model", type=Path, required=True)
    compare = commands.add_parser("compare", help="paired test maps/start poses for two or more checkpoints")
    compare.add_argument("--models", nargs="+", type=Path, required=True)
    for command in (evaluate, compare):
        add_maps(command)
        command.add_argument("--episodes", type=int, default=20)
        command.add_argument("--sample", action="store_true")
        command.add_argument("--output", type=Path)
        command.add_argument("--trace", type=Path, help="SVG of last episode of first model")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-connectome":
            from .connectome import prepare as prepare_data
            prepare_data(args.output, cache=args.cache, nodes=args.nodes, regions=args.regions, source=args.source)
            return
        import torch
        torch.set_num_threads(1)
        seed = (10007 if args.command in ("evaluate", "compare") else 7) if args.seed is None else args.seed
        torch.manual_seed(seed)
        if args.max_steps is not None and args.max_steps < 1:
            raise ValueError("max-steps must be positive")
        if args.command in ("train", "train-guided"):
            from .models import build
            from .experiment import train as run_training
            if args.hidden_size < 1:
                raise ValueError("hidden-size must be positive")
            brain = build(args.brain, hidden_size=args.hidden_size, connectome=args.connectome)
            if args.command == "train-guided":
                from .guided import train_guided
                train_guided(create_env(args), brain, args.output, seed=seed, updates=args.updates,
                             demonstrations=args.demonstrations, batch_size=args.batch_size,
                             lr=args.lr, validation_seed=args.validation_seed)
            else:
                run_training(create_env(args), brain, args.output, episodes=args.episodes, seed=seed,
                             lr=args.lr, gamma=args.gamma, rollout=args.rollout, random_start=args.random_start)
        elif args.command == "preview":
            from .models import load
            from .viewer import Viewer
            if args.delay is not None and (not math.isfinite(args.delay) or args.delay < 0):
                raise ValueError("delay must be finite and nonnegative")
            if args.model:
                # Body and sensors are checkpoint-owned; reject CLI body overrides.
                if any(argument.split("=", 1)[0] in ("--dt", "--speed", "--turn-rate", "--radius", "--reward-mode")
                       for argument in (sys.argv[1:] if argv is None else argv)):
                    raise ValueError("model playback uses body/sensor/reward settings from its checkpoint")
                brain, checkpoint = load(args.model)
                env = from_config(checkpoint["environment"], seed=seed, path=args.maze,
                                  difficulty=args.difficulty, vary_maze=args.vary_maze, max_steps=args.max_steps)
            else:
                brain, env = None, create_env(args)
            Viewer(env, brain, random_start=args.random_start, sample=args.sample, delay=args.delay).run()
        else:
            from .experiment import evaluate_models
            paths = [args.model] if args.command == "evaluate" else args.models
            if args.command == "compare" and len(paths) < 2:
                raise ValueError("compare requires at least two models")
            evaluate_models(paths, episodes=args.episodes, seed=seed, random_start=args.random_start,
                            sample=args.sample, path=args.maze, difficulty=args.difficulty,
                            vary_maze=args.vary_maze, max_steps=args.max_steps, output=args.output, trace=args.trace)
    except (ValueError, OSError, FloatingPointError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
