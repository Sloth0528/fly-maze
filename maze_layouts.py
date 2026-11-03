"""可重現的迷宮產生器：挖通所有房格，再加入少量環路。"""

from collections import deque
from dataclasses import dataclass
import random


@dataclass(frozen=True)
class Difficulty:
    width: int
    height: int
    loop_fraction: float
    max_steps: int


DIFFICULTIES = {
    "easy": Difficulty(9, 9, 0.0, 160),
    "medium": Difficulty(15, 11, 0.05, 450),
    "hard": Difficulty(21, 15, 0.12, 900),
}


def generate_layout(difficulty="easy", seed=0):
    """同一難度與 seed 得到同一地圖；G 位於離 S 最遠的房格之一。"""
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty 必須是 {', '.join(DIFFICULTIES)}")
    spec = DIFFICULTIES[difficulty]
    rng = random.Random(seed)
    grid = [["#"] * spec.width for _ in range(spec.height)]
    start = (1, 1)
    grid[1][1] = "."
    stack = [start]
    while stack:
        x, y = stack[-1]
        choices = [(x + dx, y + dy, dx, dy)
                   for dx, dy in ((2, 0), (0, 2), (-2, 0), (0, -2))
                   if 0 < x + dx < spec.width - 1
                   and 0 < y + dy < spec.height - 1
                   and grid[y + dy][x + dx] == "#"]
        if not choices:
            stack.pop()
            continue
        nx, ny, dx, dy = rng.choice(choices)
        grid[y + dy // 2][x + dx // 2] = grid[ny][nx] = "."
        stack.append((nx, ny))

    # 只移除連接兩條道路的隔牆，維持封閉邊界與連通性。
    walls = [(x, y) for y in range(1, spec.height - 1)
             for x in range(1, spec.width - 1) if grid[y][x] == "#"
             and ((x % 2 == 0 and y % 2 == 1
                   and grid[y][x - 1] == grid[y][x + 1] == ".")
                  or (x % 2 == 1 and y % 2 == 0
                      and grid[y - 1][x] == grid[y + 1][x] == "."))]
    rng.shuffle(walls)
    for x, y in walls[:round(len(walls) * spec.loop_fraction)]:
        grid[y][x] = "."

    distances = {start: 0}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            point = nx, ny = x + dx, y + dy
            if grid[ny][nx] != "#" and point not in distances:
                distances[point] = distances[(x, y)] + 1
                queue.append(point)
    rooms = [p for p in distances if p[0] % 2 and p[1] % 2 and p != start]
    farthest = max(distances[p] for p in rooms)
    gx, gy = rng.choice(sorted(p for p in rooms if distances[p] == farthest))
    grid[1][1], grid[gy][gx] = "S", "G"
    return ["".join(row) for row in grid]
