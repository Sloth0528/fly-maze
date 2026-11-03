"""無需 PyTorch 或已訓練模型的迷宮實驗台；方向鍵手動操作。"""

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from fly import DIRECTIONS, FORWARD, LEFT, RIGHT
from maze import make_maze
from maze_layouts import DIFFICULTIES

WALL, FLOOR, UNKNOWN = "#263545", "#f3efe2", "#0d1623"


class Preview:
    def __init__(self, root, difficulty="easy", seed=7, maze_path=None):
        self.root, self.maze_path = root, maze_path
        self.cell = 30
        self.total_reward = 0.0
        root.title("Fly Maze Lab | 果蠅迷宮實驗台")
        root.configure(bg="#101b2b")
        top = tk.Frame(root, bg="#101b2b")
        top.pack(fill="x", padx=24, pady=(20, 8))
        tk.Label(top, text="FLY MAZE LAB", fg="#7de4c5", bg="#101b2b",
                 font=("Consolas", 21, "bold")).pack(anchor="w")
        tk.Label(top, text="局部感知 → 記憶模型 → 前進／轉向", fg="#e6ebef",
                 bg="#101b2b", font=("Microsoft JhengHei", 11)).pack(anchor="w")
        controls = tk.Frame(root, bg="#101b2b")
        controls.pack(fill="x", padx=24, pady=8)
        self.difficulty = tk.StringVar(value=difficulty)
        selector = ttk.Combobox(controls, textvariable=self.difficulty,
                                values=list(DIFFICULTIES), state="readonly", width=9)
        selector.pack(side="left", padx=(0, 10))
        if maze_path:
            selector.configure(state="disabled")
        tk.Label(controls, text="Seed", bg="#101b2b", fg="white").pack(side="left")
        self.seed = tk.StringVar(value=str(seed))
        ttk.Entry(controls, textvariable=self.seed, width=10).pack(side="left", padx=6)
        ttk.Button(controls, text="建立迷宮", command=self.new_map).pack(side="left", padx=4)
        ttk.Button(controls, text="下一張", command=self.next_map).pack(side="left", padx=4)
        ttk.Button(controls, text="重設", command=self.reset).pack(side="left", padx=4)
        ttk.Button(controls, text="匯出地圖", command=self.export).pack(side="left", padx=4)
        self.message = tk.StringVar(value="方向鍵 ↑ 前進、← 左轉、→ 右轉；R 重設。")
        tk.Label(root, textvariable=self.message, bg="#101b2b", fg="#b1c0cd",
                 font=("Microsoft JhengHei", 10)).pack(anchor="w", padx=24, pady=4)
        body = tk.Frame(root, bg="#101b2b")
        body.pack(padx=24, pady=12)
        left = tk.Frame(body, bg="#101b2b")
        left.pack(side="left", anchor="n")
        tk.Label(left, text="觀察者全圖｜藍色為路徑", bg="#101b2b", fg="white",
                 font=("Microsoft JhengHei", 11)).pack(anchor="w", pady=(0, 8))
        self.canvas = tk.Canvas(left, highlightthickness=0, bg=UNKNOWN)
        self.canvas.pack()
        right = tk.Frame(body, bg="#101b2b")
        right.pack(side="left", anchor="n", padx=(22, 0))
        tk.Label(right, text="模型視野｜上方永遠是前方", bg="#101b2b", fg="white",
                 font=("Microsoft JhengHei", 11)).pack(anchor="w", pady=(0, 8))
        self.senses = tk.Canvas(right, width=210, height=210, bg=UNKNOWN, highlightthickness=0)
        self.senses.pack()
        self.stats = tk.StringVar()
        tk.Label(right, textvariable=self.stats, justify="left", anchor="w", bg="#101b2b",
                 fg="#e6ebef", font=("Microsoft JhengHei", 11)).pack(fill="x", pady=14)
        buttons = tk.Frame(right, bg="#101b2b")
        buttons.pack()
        for text, action in (("← 左轉", LEFT), ("↑ 前進", FORWARD), ("右轉 →", RIGHT)):
            ttk.Button(buttons, text=text, command=lambda a=action: self.act(a)).pack(side="left")
        for key, action in (("<Up>", FORWARD), ("<Left>", LEFT), ("<Right>", RIGHT)):
            root.bind(key, lambda event, a=action: self.key_action(event, a))
        root.bind("r", lambda event: self.reset() if not self.editing(event) else None)
        self.new_map()

    @staticmethod
    def editing(event):
        return isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Combobox))

    def key_action(self, event, action):
        if not self.editing(event):
            self.act(action)
            return "break"

    def new_map(self):
        try:
            seed = int(self.seed.get())
        except ValueError:
            self.message.set("Seed 請輸入整數。")
            return
        self.env = make_maze(self.maze_path, seed=seed,
                             difficulty=None if self.maze_path else self.difficulty.get(),
                             observation_mode="egocentric", reward_mode="shaped")
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.message.set("方向鍵 ↑ 前進、← 左轉、→ 右轉；R 重設。")
        self.draw()

    def next_map(self):
        try:
            self.seed.set(str(int(self.seed.get()) + 1))
        except ValueError:
            self.message.set("Seed 請輸入整數。")
            return
        self.new_map()

    def reset(self):
        self.env.reset()
        self.total_reward = self.last_reward = 0.0
        self.message.set("已重設位置與本回合探索紀錄。")
        self.draw()

    def export(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(title="匯出可供 train.py 使用的地圖",
                                           defaultextension=".txt", filetypes=[("迷宮地圖", "*.txt")])
        if path:
            try:
                Path(path).write_text("\n".join(self.env.layout) + "\n", encoding="utf-8")
                self.message.set(f"已匯出：{path}")
            except OSError as error:
                self.message.set(f"匯出失敗：{error}")

    def act(self, action):
        if self.env.terminated or self.env.truncated:
            return
        _, self.last_reward, terminated, truncated, info = self.env.step(action)
        self.total_reward += self.last_reward
        self.message.set("找到食物！按 R 再試一次。" if terminated else
                         "時間到！按 R 重設。" if truncated else
                         "碰到牆壁。" if info["collision"] else "探索中……")
        self.draw()

    @staticmethod
    def tile(canvas, x, y, size, color, label=""):
        canvas.create_rectangle(x * size, y * size, (x + 1) * size, (y + 1) * size,
                                fill=color, outline="#172435")
        if label:
            canvas.create_text((x + .5) * size, (y + .5) * size, text=label,
                               fill="#102b29", font=("Consolas", 13, "bold"))

    def draw(self):
        env, size = self.env, self.cell
        self.canvas.configure(width=env.width * size, height=env.height * size)
        self.canvas.delete("all")
        for y, row in enumerate(env.layout):
            for x, tile in enumerate(row):
                color = WALL if tile == "#" else "#a4c8db" if (x, y) in env.visited else FLOOR
                if tile == "G":
                    color = "#7de4a9"
                self.tile(self.canvas, x, y, size, color, tile if tile in "SG" else "")
        cx, cy = (env.fly.x + .5) * size, (env.fly.y + .5) * size
        dx, dy = DIRECTIONS[env.fly.heading]
        self.canvas.create_oval(cx - 8, cy - 8, cx + 8, cy + 8, fill="#ee8064", outline="")
        self.canvas.create_line(cx, cy, cx + 13 * dx, cy + 13 * dy,
                                arrow=tk.LAST, width=3, fill="#602f26")
        self.senses.delete("all")
        for i, (x, y, visible) in enumerate(env.local_cells()):
            color = UNKNOWN if not visible else WALL if env.is_wall(x, y) else FLOOR
            if visible and (x, y) == env.goal:
                color = "#7de4a9"
            self.tile(self.senses, i % 5, i // 5, 42, color,
                      "G" if visible and (x, y) == env.goal else "")
        self.senses.create_text(105, 105, text="↑", fill="#e26749", font=("Consolas", 23, "bold"))
        self.stats.set(f"動作步數     {env.steps} / {env.max_steps}\n"
                       f"碰撞次數     {env.collisions}\n"
                       f"探索格數     {len(env.visited)}\n"
                       f"本步獎勵     {self.last_reward:+.2f}\n"
                       f"累積獎勵     {self.total_reward:+.2f}\n\n"
                       "模型輸入 83 維／動作 3 種\n"
                       "深色格：未知或被牆遮擋\n"
                       "此視窗為手動操作")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--difficulty", choices=DIFFICULTIES, default="easy")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--maze", help="預覽自訂文字地圖")
    args = parser.parse_args()
    root = tk.Tk()
    Preview(root, args.difficulty, args.seed, args.maze)
    root.mainloop()


if __name__ == "__main__":
    main()
