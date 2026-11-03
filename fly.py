"""果蠅的身體：只管理位置、朝向與動作，不負責學習。"""

from dataclasses import dataclass

FORWARD, LEFT, RIGHT = 0, 1, 2
ACTION_NAMES = ["forward", "left", "right"]
# 地圖座標：向右 x 增加，向下 y 增加；朝向順序：東、南、西、北。
DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
ARROWS = [">", "v", "<", "^"]


@dataclass
class Fly:
    x: int
    y: int
    heading: int = 0

    def propose(self, action: int) -> tuple[int, int]:
        """轉彎會更新朝向；前進只提出位置，由迷宮判斷能否移動。"""
        if action == LEFT:
            self.heading = (self.heading - 1) % 4
        elif action == RIGHT:
            self.heading = (self.heading + 1) % 4
        elif action != FORWARD:
            raise ValueError("action 必須是 0（前進）、1（左轉）或 2（右轉）")
        if action == FORWARD:
            dx, dy = DIRECTIONS[self.heading]
            return self.x + dx, self.y + dy
        return self.x, self.y
