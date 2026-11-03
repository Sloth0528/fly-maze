"""觀看模型：終端機動畫或 Tkinter 視窗；均不進行學習。"""

import argparse
import time
import torch

from brain import load_brain
from fly import ACTION_NAMES, DIRECTIONS
from maze import maze_from_config
from maze_layouts import DIFFICULTIES


def play(env, brain, random_start=False, greedy=True):
    observation = env.reset(random_start=random_start)
    memory = brain.initial_memory()
    while True:
        with torch.inference_mode():
            action, _, value, _, memory, probabilities = brain.act(observation, memory, greedy)
        observation, reward, terminated, truncated, _ = env.step(action)
        yield {"action": ACTION_NAMES[action], "probabilities": probabilities,
               "value": value.item(), "memory": memory.flatten().tolist(),
               "reward": reward, "done": terminated or truncated}
        if terminated or truncated:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="outputs/fly_brain.pt")
    parser.add_argument("--gui", action="store_true", help="用 Tkinter 顯示格子地圖")
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    maps = parser.add_mutually_exclusive_group()
    maps.add_argument("--maze")
    maps.add_argument("--difficulty", choices=DIFFICULTIES)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("delay 不可小於 0")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    brain, checkpoint = load_brain(args.model)
    config = checkpoint["config"]
    env = maze_from_config(config, seed=args.seed, max_steps=args.max_steps,
                           path=args.maze, difficulty=args.difficulty)
    frames = play(env, brain, args.random_start, not args.sample)

    def status(frame):
        probabilities = ", ".join(f"{name}={p:.2f}" for name, p in
                                  zip(ACTION_NAMES, frame["probabilities"]))
        return (f"step={env.steps}  action={frame['action']}  reward={frame['reward']:.2f}\n"
                f"{probabilities}\nvalue={frame['value']:.2f}  "
                f"GRU[0:4]={[round(v, 2) for v in frame['memory'][:4]]}")

    if not args.gui:
        for frame in frames:
            print(env.ascii())
            print(status(frame), "\n", flush=True)
            if args.delay:
                time.sleep(args.delay)
        print("SUCCESS" if env.terminated else "TIME LIMIT")
        return

    import tkinter as tk
    root = tk.Tk()
    root.title("Fly Maze - GRU Actor-Critic")
    cell = 48
    canvas = tk.Canvas(root, width=env.width * cell, height=env.height * cell, bg="white")
    canvas.pack()
    label = tk.Label(root, text="", justify="left", font=("Consolas", 10))
    label.pack(padx=10, pady=10)

    def tick():
        frame = next(frames, None)
        if frame is None:
            return
        canvas.delete("all")
        visited = set(env.path)
        for y, row in enumerate(env.layout):
            for x, tile in enumerate(row):
                color = "#263442" if tile == "#" else "#fff9de"
                if (x, y) in visited and tile != "#":
                    color = "#b7dfef"
                if tile == "G":
                    color = "#90d49b"
                canvas.create_rectangle(x*cell, y*cell, (x+1)*cell, (y+1)*cell,
                                        fill=color, outline="#d0d0d0")
                if tile in "SG":
                    canvas.create_text((x+0.5)*cell, (y+0.5)*cell, text=tile)
        cx, cy = (env.fly.x+0.5)*cell, (env.fly.y+0.5)*cell
        canvas.create_oval(cx-12, cy-12, cx+12, cy+12, fill="#d45d45")
        dx, dy = DIRECTIONS[env.fly.heading]
        canvas.create_line(cx, cy, cx+dx*21, cy+dy*21, arrow=tk.LAST, width=3)
        text = status(frame)
        if frame["done"]:
            text += "\nSUCCESS" if env.terminated else "\nTIME LIMIT"
        label.config(text=text)
        if not frame["done"]:
            root.after(max(1, int(args.delay * 1000)), tick)

    root.after(0, tick)
    root.mainloop()


if __name__ == "__main__":
    main()
