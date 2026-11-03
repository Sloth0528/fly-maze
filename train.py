"""按回合訓練的 recurrent Actor-Critic 教學版；不是 PPO，也不是真實果蠅腦。"""

import argparse
import csv
from pathlib import Path
import torch
from torch import nn

from brain import Brain
from maze import make_maze
from maze_layouts import DIFFICULTIES


def train_episode(env, brain, optimizer, gamma=0.97, random_start=False):
    observation = env.reset(random_start=random_start)
    memory = brain.initial_memory()
    log_probs, values, entropies, rewards = [], [], [], []
    terminated = truncated = False
    while not (terminated or truncated):
        action, log_prob, value, entropy, memory, _ = brain.act(observation, memory)
        observation, reward, terminated, truncated, _ = env.step(action)
        log_probs.append(log_prob)
        values.append(value)
        entropies.append(entropy)
        rewards.append(reward)

    # 真正抵達食物才把後續價值設為 0；時間上限只是截斷，仍估計後續價值。
    with torch.no_grad():
        next_value = 0.0 if terminated else brain(observation, memory)[1].item()
    returns = []
    total = next_value
    for reward in reversed(rewards):
        total = reward + gamma * total
        returns.insert(0, total)
    returns = torch.tensor(returns, dtype=torch.float32)
    values = torch.stack(values)
    advantage = returns - values
    # actor 學會偏好高回報動作；critic 學會估計價值；entropy 鼓勵探索。
    actor_loss = -(torch.stack(log_probs) * advantage.detach()).mean()
    critic_loss = 0.5 * advantage.square().mean()
    loss = actor_loss + critic_loss - 0.05 * torch.stack(entropies).mean()
    optimizer.zero_grad()
    loss.backward()  # 梯度會沿著本回合的 GRU 記憶傳回去。
    nn.utils.clip_grad_norm_(brain.parameters(), 0.5)
    optimizer.step()  # 真正改變網路權重的位置。
    return {"reward": sum(rewards), "steps": env.steps,
            "success": int(env.terminated), "loss": loss.item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, help="未指定時採用難度預設；原始地圖為 40")
    parser.add_argument("--lr", type=float, default=0.0003)
    maps = parser.add_mutually_exclusive_group()
    maps.add_argument("--maze", help="自訂文字地圖路徑")
    maps.add_argument("--difficulty", choices=DIFFICULTIES)
    parser.add_argument("--vary-maze", action="store_true", help="每回合換圖，需搭配 --difficulty")
    parser.add_argument("--observation-mode", choices=("world", "egocentric"))
    parser.add_argument("--reward-mode", choices=("legacy", "shaped", "sparse"))
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/fly_brain.pt"))
    args = parser.parse_args()
    if args.episodes < 1 or args.lr <= 0:
        parser.error("episodes 與 lr 必須大於 0")
    if args.vary_maze and args.difficulty is None:
        parser.error("--vary-maze 需要 --difficulty")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps 必須大於 0")
    log_path = args.output.with_suffix(".csv")
    if args.output.suffix.lower() != ".pt":
        parser.error("output 請使用 .pt 副檔名")
    if args.output.exists() or log_path.exists():
        parser.error("輸出已存在；請用 --output outputs/另一個名稱.pt 保留舊實驗")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    budget = args.max_steps if args.max_steps is not None or args.difficulty else 40
    env = make_maze(args.maze, budget, args.seed, difficulty=args.difficulty,
                    vary_maze=args.vary_maze, observation_mode=args.observation_mode,
                    reward_mode=args.reward_mode)
    brain = Brain(env.observation_size)
    optimizer = torch.optim.Adam(brain.parameters(), lr=args.lr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    recent = []
    with log_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", "reward", "steps", "success", "loss"])
        writer.writeheader()
        for episode in range(1, args.episodes + 1):
            metrics = train_episode(env, brain, optimizer, random_start=args.random_start)
            writer.writerow({"episode": episode, **metrics})
            recent.append(metrics["success"])
            recent = recent[-100:]
            if episode == 1 or episode % 100 == 0 or episode == args.episodes:
                print(f"episode={episode:4d} train_success(last {len(recent)})="
                      f"{sum(recent) / len(recent):.1%} reward={metrics['reward']:.2f} "
                      f"loss={metrics['loss']:.3f}", flush=True)
    checkpoint = {
        "format_version": 1,
        "state_dict": brain.state_dict(),
        "config": {"observation_size": brain.observation_size,
                   "hidden_size": brain.hidden_size, **env.config(), "seed": args.seed,
                   "episodes": args.episodes, "algorithm": "recurrent_actor_critic"},
    }
    with args.output.open("xb") as handle:
        torch.save(checkpoint, handle)
    print(f"Saved model: {args.output}\nTraining log: {log_path}")


if __name__ == "__main__":
    main()
