"""Manual/model Tkinter preview and standalone SVG trajectory export."""

import html
import math
from pathlib import Path
from types import SimpleNamespace

from . import __version__

def export_svg(env, path, title="Continuous Fly Maze"):
    scale = 42
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{env.width*scale}" height="{env.height*scale+45}" viewBox="0 0 {env.width*scale} {env.height*scale+45}">',
              '<rect width="100%" height="100%" fill="#f8fafc"/>',
              f'<text x="10" y="24" font-family="sans-serif" font-size="15">{html.escape(title)}</text>', '<g transform="translate(0,40)">']
    for y, row in enumerate(env.layout):
        for x, tile in enumerate(row):
            color = "#263442" if tile == "#" else ("#86efac" if tile == "G" else "#fff9de")
            pieces.append(f'<rect x="{x*scale}" y="{y*scale}" width="{scale}" height="{scale}" fill="{color}" stroke="#cbd5e1"/>')
    points = " ".join(f"{x*scale:.3f},{y*scale:.3f}" for x, y in env.path)
    pieces.append(f'<polyline points="{points}" fill="none" stroke="#0284c7" stroke-width="2"/>')
    cx, cy, r = env.fly.x*scale, env.fly.y*scale, env.radius*scale
    pieces.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#ef4444"/>')
    pieces.append(f'<line x1="{cx}" y1="{cy}" x2="{cx+18*math.cos(env.fly.heading)}" y2="{cy+18*math.sin(env.fly.heading)}" stroke="#111827" stroke-width="2"/>')
    pieces.append('</g></svg>')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(pieces))


class Viewer:
    def __init__(self, env, brain=None, *, random_start=False, sample=False, delay=None,
                 identity=None, log_dir=None):
        import tkinter as tk
        self.tk, self.env, self.brain = tk, env, brain
        self.random_start, self.sample = random_start, sample
        self.session = None
        self.last_error = ""
        self.log_dir = Path(log_dir) if log_dir else Path(__file__).resolve().parents[1]/"outputs"/"v2_1_sessions"
        if brain is not None:
            from .runtime import ControlSession
            self.session = ControlSession(env, brain, sample=sample, identity=identity)
        self.delay = max(1, round(env.dt*1000 if delay is None else delay*1000))
        self.root = tk.Tk()
        self.root.title(f"Fly Maze {__version__} - "+(brain.kind if brain else "Manual"))
        self.heading = tk.Label(self.root, font=("Microsoft JhengHei UI", 12, "bold"),
                                text="自動控制：果蠅連接組模型" if brain and brain.kind == "connectome" else
                                     ("自動控制：GRU 比較基準" if brain else "手動操作"))
        self.heading.pack(pady=(10, 0))
        body = tk.Frame(self.root)
        body.pack()
        arena = tk.Frame(body)
        arena.pack(side="left", anchor="n")
        self.cell = min(56, max(20, 800//max(env.width, env.height)))
        self.canvas = tk.Canvas(arena, width=env.width*self.cell, height=env.height*self.cell, bg="#fff9de")
        self.canvas.pack(padx=10, pady=10)
        self.label = tk.Label(arena, justify="left", font=("Consolas", 10))
        self.label.pack(padx=10, pady=6)
        self.brain_panel = tk.Frame(body, padx=12, pady=10)
        self.brain_panel.pack(side="left", anchor="n")
        self.brain_info = tk.Label(self.brain_panel, justify="left", font=("Microsoft JhengHei UI", 10))
        self.brain_info.pack(anchor="w")
        self.activity_canvas = tk.Canvas(self.brain_panel, width=320, height=180, bg="#f1f5f9", highlightthickness=0)
        self.activity_canvas.pack(pady=6)
        self.activity_canvas.bind("<Motion>", self.inspect_node)
        self.node_label = tk.Label(self.brain_panel, text="", justify="left", font=("Consolas", 9), width=43, anchor="w")
        self.node_label.pack(anchor="w")
        self.motor_label = tk.Label(self.brain_panel, text="", justify="left", font=("Consolas", 10), anchor="w")
        self.motor_label.pack(anchor="w", pady=8)
        self.top_label = tk.Label(self.brain_panel, text="", justify="left", font=("Consolas", 9), anchor="w")
        self.top_label.pack(anchor="w")
        controls = tk.Frame(self.root)
        controls.pack(pady=6)
        tk.Button(controls, text="暫停／繼續", command=self.toggle_pause).pack(side="left", padx=4)
        tk.Button(controls, text="重新開始", command=lambda: self.press(SimpleNamespace(keysym="r"))).pack(side="left", padx=4)
        self.next_button = tk.Button(controls, text="下一張地圖", command=lambda: self.press(SimpleNamespace(keysym="n")),
                                     state="normal" if env.difficulty else "disabled")
        self.next_button.pack(side="left", padx=4)
        tk.Button(controls, text="載入模型", command=self.choose_model,
                  state="normal" if self.session else "disabled").pack(side="left", padx=4)
        tk.Button(controls, text="匯出模型紀錄", command=self.export_session,
                  state="normal" if self.session else "disabled").pack(side="left", padx=4)
        self.footer = self.tk.Label(self.root)
        self.footer.pack(pady=6)
        self.keys, self.paused = set(), False
        self.root.bind("<KeyPress>", self.press)
        self.root.bind("<KeyRelease>", lambda event: self.keys.discard(event.keysym.lower()))
        self.root.bind("<FocusOut>", lambda event: self.keys.clear())
        self.reset()
        self.root.after(self.delay, self.tick)

    def reset(self):
        if self.session:
            self.session.reset(self.random_start)
            self.memory = self.session.memory
        else:
            self.env.reset(self.random_start)
            self.memory = None
        self.last_error = ""
        self.keys.clear()
        self.draw()

    def toggle_pause(self):
        self.paused = not self.paused

    def replace_model(self, path):
        from .runtime import load_controller, ControlSession
        from .environment import from_config
        kind = self.brain.kind if self.brain else "connectome"
        brain, checkpoint, identity = load_controller(path, expected_kind=kind)
        env = from_config(checkpoint["environment"], seed=self.env.seed)
        self.brain, self.env = brain, env
        self.session = ControlSession(env, brain, sample=self.sample, identity=identity)
        self.cell = min(56, max(20, 800//max(env.width, env.height)))
        self.canvas.config(width=env.width*self.cell, height=env.height*self.cell)
        self.next_button.config(state="normal" if env.difficulty else "disabled")
        self.paused = False
        self.reset()

    def choose_model(self):
        from tkinter import filedialog, messagebox
        path = filedialog.askopenfilename(parent=self.root, title="載入模型（會使用該模型的地圖並重設記憶）",
                                         initialdir=Path(__file__).resolve().parents[1]/"models",
                                         filetypes=[("PyTorch checkpoint", "*.pt")])
        if path:
            try:
                self.replace_model(path)
            except (OSError, ValueError, RuntimeError) as error:
                messagebox.showerror("模型載入失敗", str(error), parent=self.root)

    def inspect_node(self, event):
        if self.brain is None:
            return
        index = (event.y//10)*32+event.x//10
        if event.x < 0 or event.y < 0 or event.x >= 320 or index >= self.brain.hidden_size:
            return
        node = self.brain.root_ids[index] if self.brain.kind == "connectome" else f"GRU-{index}"
        value = self.memory.flatten()[index].item()
        self.node_label.config(text=f"node {index}: {node}\nabstract state={value:+.5f}")

    def export_session(self):
        if self.session is None:
            return
        from datetime import datetime, timezone, timedelta
        from tkinter import messagebox
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
        target = self.log_dir/f"{self.brain.kind}_{stamp}.json"
        try:
            self.session.export(target)
            self.node_label.config(text=f"Saved: {target.name}")
        except OSError as error:
            messagebox.showerror("匯出失敗", str(error), parent=self.root)

    def draw_brain(self):
        canvas = self.activity_canvas
        canvas.delete("all")
        if self.brain is None:
            self.brain_info.config(text="手動模式：目前沒有模型控制身體\n使用 run_v2_1.py 啟動連接組模型")
            self.motor_label.config(text="Motor source: keyboard")
            self.footer.config(text="Up/W: forward | Left/A, Right/D: steer | Space: pause | R: reset | N: next map")
            return
        brain, session = self.brain, self.session
        if brain.kind == "connectome":
            self.brain_info.config(text=f"FlyWire FAFB v783 連接組子網路\n{brain.hidden_size} 節點 / {int((brain.counts > 0).sum())} 有向連線\n"
                                        "顏色為抽象隱藏狀態，非生物放電")
        else:
            self.brain_info.config(text=f"GRU 比較基準\n{brain.hidden_size} 個隱藏狀態")
        canvas.config(height=max(40, math.ceil(brain.hidden_size/32)*10))
        for index, value in enumerate(self.memory.flatten().tolist()):
            amount = min(1.0, abs(value))
            fade = round(245-170*amount)
            color = f"#ef{fade:02x}{fade:02x}" if value >= 0 else f"#{fade:02x}{fade:02x}ef"
            x, y = (index % 32)*10, (index//32)*10
            canvas.create_rectangle(x, y, x+9, y+9, fill=color, outline="")
        frame = session.latest
        training = session.identity.get("training", {})
        trained = training.get("episodes", "unknown")
        training_line = (f"Guided updates: {training.get('selected_update', '?')}/{training.get('updates', '?')}"
                         if training.get("algorithm") == "recurrent_behavior_cloning_dagger"
                         else f"Training episodes: {trained}")
        self.footer.config(text=f"模型：{session.identity.get('checkpoint', '未標示檔名')} | Space: pause | R: reset")
        if frame is None:
            self.motor_label.config(text=f"Motor source: brain.act\n{training_line}\nWaiting for first brain step")
            self.top_label.config(text="")
        else:
            throttle, steering = frame["action"]
            self.motor_label.config(text=f"Motor source: brain.act\n{training_line}\n"
                                    f"throttle={throttle:.3f} steering={steering:+.3f}\n"
                                    f"speed={frame['velocity']:.3f} cells/s\n"
                                    f"turn={math.degrees(frame['angular_velocity']):+.1f} deg/s\n"
                                    f"state RMS={frame['state_rms']:.4f}")
            self.top_label.config(text="Top |state| nodes:\n"+"\n".join(
                f"{node['root_id']}  {node['state']:+.3f}" for node in frame["top_nodes"]))

    def press(self, event):
        key = event.keysym.lower()
        if key == "space":
            self.paused = not self.paused
        elif key == "r":
            # Reset same map even when the environment was configured to vary maps.
            vary = self.env.vary_maze
            self.env.vary_maze = False
            self.reset()
            self.env.vary_maze = vary
        elif key == "n" and self.env.difficulty:
            vary = self.env.vary_maze
            self.env.vary_maze = True
            self.reset()
            self.env.vary_maze = vary
        else:
            self.keys.add(key)

    def draw(self):
        env, canvas, cell = self.env, self.canvas, self.cell
        canvas.delete("all")
        for y, row in enumerate(env.layout):
            for x, tile in enumerate(row):
                color = "#263442" if tile == "#" else ("#86efac" if tile == "G" else "#fff9de")
                canvas.create_rectangle(x*cell, y*cell, (x+1)*cell, (y+1)*cell, fill=color, outline="#cbd5e1")
                if tile in "SG":
                    canvas.create_text((x+0.5)*cell, (y+0.5)*cell, text=tile)
        # Downsample long traces for interactive drawing; all points remain in env.path.
        points = env.path[::max(1, len(env.path)//3000)]
        if len(points) >= 2:
            canvas.create_line(*(coordinate*cell for point in points for coordinate in point), fill="#0284c7", width=2)
        x, y = env.fly.x*cell, env.fly.y*cell
        for angle, distance, food in env.sense():
            canvas.create_line(x, y, x+math.cos(angle)*distance*cell, y+math.sin(angle)*distance*cell,
                               fill="#22c55e" if food else "#d6a5ed", dash=(2, 4))
        r = env.radius*cell
        canvas.create_oval(x-r, y-r, x+r, y+r, fill="#ef4444", outline="#7f1d1d")
        canvas.create_line(x, y, x+math.cos(env.fly.heading)*20, y+math.sin(env.fly.heading)*20,
                           arrow=self.tk.LAST, fill="#111827", width=3)
        state = "ERROR" if self.last_error else ("SUCCESS" if env.terminated else ("TIME LIMIT" if env.truncated else ("PAUSED" if self.paused else "RUNNING")))
        self.label.config(text=f"{state}  x={env.fly.x:.3f} y={env.fly.y:.3f} angle={math.degrees(env.fly.heading):.1f} deg\n"
                          f"steps={env.steps}/{env.max_steps} simulated time={env.elapsed:.2f}s\n"
                          f"distance={env.path_length:.3f} cells  collisions={env.collisions}  sensors=31"
                          +(f"\n{self.last_error}" if self.last_error else "")
                          +("\n停滯提示：近 3 秒位移很小，請確認載入的模型與訓練結果。"
                            if self.session and self.session.latest and self.session.latest.get("stalled") else ""))
        self.draw_brain()

    def tick(self):
        env = self.env
        if not self.paused and not (env.terminated or env.truncated):
            if self.brain is not None:
                try:
                    self.session.step()
                    self.memory = self.session.memory
                except (ValueError, RuntimeError, FloatingPointError) as error:
                    self.last_error, self.paused = str(error), True
            else:
                throttle = float(bool(self.keys & {"up", "w"}))
                steering = float(bool(self.keys & {"right", "d"}))-float(bool(self.keys & {"left", "a"}))
                action = [throttle, steering]
                # Manual idle does not consume the episode budget.
                if action == [0.0, 0.0]:
                    self.draw()
                    self.root.after(self.delay, self.tick)
                    return
                env.step(action)
        self.draw()
        self.root.after(self.delay, self.tick)

    def run(self):
        self.root.mainloop()
