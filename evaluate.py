"""不更新權重的獨立評估。固定起點成功不代表能解陌生迷宮。"""

import argparse
import torch

from brain import load_brain
from maze import maze_from_config
from maze_layouts import DIFFICULTIES


def run_episode(env, brain, random_start=False, greedy=True):
    observation = env.reset(random_start=random_start)
    memory = brain.initial_memory()
    done = False
    total_reward = 0.0
    collisions = 0
    with torch.inference_mode():
        while not done:
            action, _, _, _, memory, _ = brain.act(observation, memory, greedy=greedy)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            collisions += int(info["collision"])
            done = terminated or truncated
    return {"success": env.terminated, "steps": env.steps,
            "reward": total_reward, "collisions": collisions,
            "optimal_steps": env.optimal_steps,
            "efficiency": env.optimal_steps / env.steps if env.terminated else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="outputs/fly_brain.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--sample", action="store_true", help="依動作機率抽樣；預設選最高機率")
    maps = parser.add_mutually_exclusive_group()
    maps.add_argument("--maze", help="替換成未參與訓練的文字地圖")
    maps.add_argument("--difficulty", choices=DIFFICULTIES)
    parser.add_argument("--vary-maze", action="store_true", help="以 seed 指定測試地圖序列")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes 必須大於 0")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    brain, checkpoint = load_brain(args.model)
    config = checkpoint["config"]
    env = maze_from_config(config, seed=args.seed, max_steps=args.max_steps,
                           path=args.maze, difficulty=args.difficulty, vary_maze=args.vary_maze)
    results = [run_episode(env, brain, args.random_start, not args.sample)
               for _ in range(args.episodes)]
    successes = [r for r in results if r["success"]]
    print(f"Success: {len(successes)}/{len(results)} ({len(successes)/len(results):.1%})")
    print(f"Mean steps (all episodes): {sum(r['steps'] for r in results)/len(results):.1f}")
    if successes:
        print(f"Mean steps (successful only): {sum(r['steps'] for r in successes)/len(successes):.1f}")
    print(f"Mean collisions: {sum(r['collisions'] for r in results)/len(results):.1f}")
    print(f"Mean path efficiency (failures=0): {sum(r['efficiency'] for r in results)/len(results):.3f}")
    print(f"Mode: {'sample' if args.sample else 'greedy'}; random_start={args.random_start}; "
          f"vary_maze={args.vary_maze}; seed={args.seed}; observation={env.observation_mode}")
    print(env.ascii())


if __name__ == "__main__":
    main()
