"""環境：地圖、局部觀察、碰撞、獎勵與回合結束。只依賴標準函式庫。"""

from collections import deque
from pathlib import Path
import random

from fly import ARROWS, DIRECTIONS, FORWARD, Fly
from maze_layouts import DIFFICULTIES, generate_layout

DEFAULT_MAP = [
    "#######",
    "###G###",
    "###.###",
    "#S....#",
    "#######",
]


class Maze:
    # 5x5 區域 x 3 通道 + 朝向 one-hot 4 個值 + 上次動作 3 個值 + 撞牆 1 個值。
    radius = 2
    observation_size = 5 * 5 * 3 + 4 + 3 + 1

    def __init__(self, layout=None, max_steps=100, seed=0, *,
                 observation_mode="world", reward_mode="legacy"):
        if observation_mode not in ("world", "egocentric"):
            raise ValueError("observation_mode 必須是 world 或 egocentric")
        if reward_mode not in ("legacy", "shaped", "sparse"):
            raise ValueError("reward_mode 必須是 legacy、shaped 或 sparse")
        self.observation_mode = observation_mode
        self.reward_mode = reward_mode
        self.layout = list(DEFAULT_MAP if layout is None else layout)
        if not self.layout or not self.layout[0]:
            raise ValueError("地圖不可為空")
        self.height, self.width = len(self.layout), len(self.layout[0])
        if any(len(row) != self.width for row in self.layout):
            raise ValueError("地圖每一列必須等長")
        if any(c not in "#.SG" for row in self.layout for c in row):
            raise ValueError("地圖只接受 #、.、S、G")
        starts = self.find("S")
        goals = self.find("G")
        if len(starts) != 1 or len(goals) != 1:
            raise ValueError("地圖必須各有一個 S 起點與 G 食物")
        if max_steps < 1:
            raise ValueError("max_steps 必須大於 0")
        self.start, self.goal = starts[0], goals[0]
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.start_candidates = self.reachable_cells()
        if self.start not in self.start_candidates:
            raise ValueError("起點無法到達食物")
        # 明確呼叫基底 reset，讓隨機環境能在初始化後才開始換圖。
        Maze.reset(self)

    def find(self, symbol):
        return [(x, y) for y, row in enumerate(self.layout)
                for x, cell in enumerate(row) if cell == symbol]

    def is_wall(self, x, y):
        return (x < 0 or y < 0 or x >= self.width or y >= self.height
                or self.layout[y][x] == "#")

    def reachable_cells(self):
        """從食物搜尋可達位置，只用於驗證地圖與抽取起點，不傳給 AI。"""
        queue = deque([self.goal])
        seen = {self.goal}
        while queue:
            x, y = queue.popleft()
            for dx, dy in DIRECTIONS:
                point = (x + dx, y + dy)
                if point not in seen and not self.is_wall(*point):
                    seen.add(point)
                    queue.append(point)
        return sorted(seen - {self.goal})

    def reset(self, random_start=False):
        start = self.rng.choice(self.start_candidates) if random_start else self.start
        heading = self.rng.randrange(4) if random_start else 0
        self.fly = Fly(*start, heading=heading)
        self.steps = 0
        self.last_action = None
        self.collision = False
        self.terminated = self.truncated = False
        self.path = [start]
        self.visited = {start}
        self.collisions = 0
        self.optimal_steps = self.shortest_action_count(start, heading)
        return self.observe()

    def shortest_action_count(self, start=None, heading=None):
        """含原地轉彎的最短動作數；只供評估，不放進模型觀察。"""
        start = (self.fly.x, self.fly.y) if start is None else start
        heading = self.fly.heading if heading is None else heading
        queue = deque([(*start, heading, 0)])
        seen = {(*start, heading)}
        while queue:
            x, y, direction, distance = queue.popleft()
            if (x, y) == self.goal:
                return distance
            dx, dy = DIRECTIONS[direction]
            candidates = [(x, y, (direction - 1) % 4),
                          (x, y, (direction + 1) % 4)]
            if not self.is_wall(x + dx, y + dy):
                candidates.append((x + dx, y + dy, direction))
            for state in candidates:
                if state not in seen:
                    seen.add(state)
                    queue.append((*state, distance + 1))
        raise ValueError("位置無法到達食物")

    def line_of_sight(self, target_x, target_y):
        """格線射線：可以看見第一面牆，無法穿牆或斜穿封閉牆角。"""
        x, y = self.fly.x, self.fly.y
        nx, ny = abs(target_x - x), abs(target_y - y)
        sx, sy = (1 if target_x > x else -1), (1 if target_y > y else -1)
        ix = iy = 0
        while ix < nx or iy < ny:
            decision = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx
            if decision == 0:
                if self.is_wall(x + sx, y) or self.is_wall(x, y + sy):
                    return False
                x, y, ix, iy = x + sx, y + sy, ix + 1, iy + 1
            elif decision < 0:
                x, ix = x + sx, ix + 1
            else:
                y, iy = y + sy, iy + 1
            if (x, y) == (target_x, target_y):
                return True
            if self.is_wall(x, y):
                return False
        return True

    def local_cells(self):
        """回傳視野中的 (世界 x, 世界 y, 是否可見)，依列排序。"""
        cells = []
        fx, fy = DIRECTIONS[self.fly.heading]
        rx, ry = -fy, fx
        for dy in range(-self.radius, self.radius + 1):
            for dx in range(-self.radius, self.radius + 1):
                if self.observation_mode == "egocentric":
                    x = self.fly.x + dx * rx - dy * fx
                    y = self.fly.y + dx * ry - dy * fy
                    visible = self.line_of_sight(x, y)
                else:
                    x, y = self.fly.x + dx, self.fly.y + dy
                    visible = True
                cells.append((x, y, visible))
        return cells

    def observe(self):
        """75 維局部感知 + 4 維理想羅盤 + 3 維上次動作 + 1 維碰撞。"""
        observation = []
        cells = self.local_cells()
        for channel in range(3):  # 牆壁、可走道路、食物
            for x, y, visible in cells:
                wall = self.is_wall(x, y)
                channels = [wall, not wall, (x, y) == self.goal]
                observation.append(float(visible and channels[channel]))
        observation.extend(float(i == self.fly.heading) for i in range(4))
        observation.extend(float(i == self.last_action) for i in range(3))
        observation.append(float(self.collision))
        return observation

    def step(self, action):
        if self.terminated or self.truncated:
            raise RuntimeError("回合已結束，請先 reset()")
        old_x, old_y = self.fly.x, self.fly.y
        x, y = self.fly.propose(action)
        self.collision = action == FORWARD and self.is_wall(x, y)
        self.collisions += int(self.collision)
        if not self.collision:
            self.fly.x, self.fly.y = x, y
        self.last_action = action
        self.steps += 1
        position = (self.fly.x, self.fly.y)
        first_visit = position not in self.visited
        self.visited.add(position)
        self.path.append(position)
        self.terminated = (self.fly.x, self.fly.y) == self.goal
        self.truncated = self.steps >= self.max_steps and not self.terminated
        reward = -0.01 - (0.1 if self.collision else 0.0)
        if first_visit and self.reward_mode != "sparse":
            reward += 0.05 if self.reward_mode == "legacy" else 0.02
        gx, gy = self.goal
        # 食物在前後兩個局部視野中都可見時，提供「接近食物」的教學提示。
        if (self.reward_mode == "legacy"
                and max(abs(old_x-gx), abs(old_y-gy)) <= self.radius
                and max(abs(self.fly.x-gx), abs(self.fly.y-gy)) <= self.radius):
            before = abs(old_x-gx) + abs(old_y-gy)
            after = abs(self.fly.x-gx) + abs(self.fly.y-gy)
            reward += 0.2 * (before - after)
        if self.terminated:
            reward = 10.0
        info = {"success": self.terminated, "collision": self.collision,
                "steps": self.steps, "collisions": self.collisions,
                "termination_reason": ("food" if self.terminated else
                                       "time_limit" if self.truncated else None)}
        return self.observe(), reward, self.terminated, self.truncated, info

    def config(self):
        return {"layout": self.layout, "max_steps": self.max_steps,
                "observation_mode": self.observation_mode,
                "reward_mode": self.reward_mode}

    def ascii(self):
        rows = [list(row) for row in self.layout]
        for x, y in self.path:
            if rows[y][x] == ".":
                rows[y][x] = "*"
        rows[self.fly.y][self.fly.x] = ARROWS[self.fly.heading]
        return "\n".join("".join(row) for row in rows)


class ProceduralMaze(Maze):
    """固定 seed 序列抽圖；reset 換圖與隨機起點可以獨立設定。"""

    def __init__(self, difficulty="easy", max_steps=None, seed=0, *,
                 vary_maze=False, observation_mode="egocentric", reward_mode="shaped"):
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"未知難度：{difficulty}")
        self.difficulty, self.vary_maze = difficulty, vary_maze
        self.map_rng = random.Random(seed)
        self.map_seed = seed
        super().__init__(generate_layout(difficulty, seed),
                         DIFFICULTIES[difficulty].max_steps if max_steps is None else max_steps,
                         seed, observation_mode=observation_mode, reward_mode=reward_mode)

    def reset(self, random_start=False):
        if self.vary_maze:
            self.map_seed = self.map_rng.randrange(2**63)
            layout = generate_layout(self.difficulty, self.map_seed)
            # 建立並驗證新地圖，再一次替換與地圖相關的資料。
            replacement = Maze(layout, self.max_steps, observation_mode=self.observation_mode,
                               reward_mode=self.reward_mode)
            for name in ("layout", "width", "height", "start", "goal", "start_candidates"):
                setattr(self, name, getattr(replacement, name))
        return super().reset(random_start)

    def config(self):
        return {**super().config(), "difficulty": self.difficulty,
                "vary_maze": self.vary_maze, "map_seed": self.map_seed}


def make_maze(path=None, max_steps=None, seed=0, *, difficulty=None,
              vary_maze=False, observation_mode=None, reward_mode=None):
    if path is not None and difficulty is not None:
        raise ValueError("自訂地圖與 difficulty 請擇一")
    if vary_maze and difficulty is None:
        raise ValueError("vary_maze 需要指定 difficulty")
    if difficulty is not None:
        return ProceduralMaze(difficulty, max_steps, seed, vary_maze=vary_maze,
                              observation_mode=observation_mode or "egocentric",
                              reward_mode=reward_mode or "shaped")
    layout = None if path is None else Path(path).read_text(encoding="utf-8-sig").splitlines()
    return Maze(layout, max_steps=100 if max_steps is None else max_steps, seed=seed,
                observation_mode=observation_mode or "world", reward_mode=reward_mode or "legacy")


def maze_from_config(config, *, seed=0, max_steps=None, path=None,
                     difficulty=None, vary_maze=False):
    """載入舊模型用原本感知；新模型保留訓練時的感知與獎勵設定。"""
    options = {"observation_mode": config.get("observation_mode", "world"),
               "reward_mode": config.get("reward_mode", "legacy")}
    budget = config["max_steps"] if max_steps is None else max_steps
    if vary_maze and difficulty is None:
        difficulty = config.get("difficulty")
    if path is not None or difficulty is not None or vary_maze:
        return make_maze(path, budget, seed, difficulty=difficulty,
                         vary_maze=vary_maze, **options)
    return Maze(config["layout"], budget, seed, **options)
