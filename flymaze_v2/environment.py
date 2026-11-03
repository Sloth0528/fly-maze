"""Continuous unicycle body, swept disk collision and occluded egocentric rays."""

from dataclasses import dataclass
import math
from pathlib import Path
import random

from maze import Maze, DEFAULT_MAP
from maze_layouts import DIFFICULTIES, generate_layout
from .geometry import ray_box, segment_box_sq, segment_circle_entry


@dataclass
class Body:
    x: float
    y: float
    heading: float  # radians; positive clockwise in screen coordinates


class ContinuousMaze:
    observation_size = 31  # 13 ranges + 13 food hits + compass 2 + previous action 2 + collision
    action_size = 2
    ray_angles = tuple(-3*math.pi/4 + i*math.pi/8 for i in range(13))

    def __init__(self, layout=None, *, difficulty=None, vary_maze=False, seed=7,
                 max_steps=800, dt=0.1, speed=1.5, turn_rate=math.pi,
                 radius=0.16, sensor_range=4.0, goal_radius=0.25, reward_mode="shaped"):
        if layout is not None and difficulty is not None:
            raise ValueError("layout and difficulty are mutually exclusive")
        if difficulty is not None and difficulty not in DIFFICULTIES:
            raise ValueError("unknown difficulty")
        if vary_maze and difficulty is None:
            raise ValueError("vary_maze requires difficulty")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        for name, value in dict(dt=dt, speed=speed, turn_rate=turn_rate,
                                sensor_range=sensor_range, goal_radius=goal_radius).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(radius) or not 0 < radius < 0.45:
            raise ValueError("radius must be between 0 and 0.45 cell")
        if goal_radius >= 0.5:
            raise ValueError("goal_radius must be smaller than half a cell")
        if reward_mode not in ("shaped", "sparse"):
            raise ValueError("reward_mode must be shaped or sparse")
        self.difficulty, self.vary_maze, self.seed = difficulty, vary_maze, seed
        self.rng, self.map_rng = random.Random(seed), random.Random(seed)
        self.max_steps, self.dt, self.speed, self.turn_rate = max_steps, dt, speed, turn_rate
        self.radius, self.sensor_range, self.goal_radius = radius, sensor_range, goal_radius
        self.reward_mode = reward_mode
        self.map_seed = seed
        self._set_layout(generate_layout(difficulty, seed) if difficulty else (DEFAULT_MAP if layout is None else layout))
        self.reset()

    def _set_layout(self, layout):
        reference = Maze(layout)
        if any(c != "#" for c in reference.layout[0]+reference.layout[-1]) or any(
                row[0] != "#" or row[-1] != "#" for row in reference.layout):
            raise ValueError("continuous maps require a closed wall boundary")
        self.layout = reference.layout
        self.width, self.height = reference.width, reference.height
        self.start, self.goal = reference.start, reference.goal
        self.start_candidates = reference.start_candidates
        self.walls = [(x, y) for y, row in enumerate(self.layout)
                      for x, tile in enumerate(row) if tile == "#"]

    def reset(self, random_start=False):
        if self.vary_maze:
            self.map_seed = self.map_rng.randrange(2**63)
            self._set_layout(generate_layout(self.difficulty, self.map_seed))
        sx, sy = self.rng.choice(self.start_candidates) if random_start else self.start
        self.fly = Body(sx+0.5, sy+0.5, self.rng.uniform(-math.pi, math.pi) if random_start else 0.0)
        self.steps = self.collisions = 0
        self.elapsed = self.path_length = 0.0
        self.collision = self.terminated = self.truncated = False
        self.last_action = [0.0, 0.0]
        self.visited = {(sx, sy)}
        self.path = [(self.fly.x, self.fly.y)]
        self.initial_pose = [self.fly.x, self.fly.y, self.fly.heading]
        return self.observe()

    def clear_segment(self, ax, ay, bx, by):
        r = self.radius
        if min(ax, bx)-r <= 0 or min(ay, by)-r <= 0 or max(ax, bx)+r >= self.width or max(ay, by)+r >= self.height:
            return False
        for x, y in self.walls:
            if x > max(ax, bx)+r or x+1 < min(ax, bx)-r or y > max(ay, by)+r or y+1 < min(ay, by)-r:
                continue
            if segment_box_sq(ax, ay, bx, by, x, y, x+1, y+1) <= r*r+1e-12:
                return False
        return True

    def sense(self):
        readings = []
        x, y, heading = self.fly.x, self.fly.y, self.fly.heading
        gx, gy = self.goal[0]+0.5, self.goal[1]+0.5
        for relative in self.ray_angles:
            angle = heading+relative
            dx, dy = math.cos(angle), math.sin(angle)
            distance = min(self.sensor_range, *(ray_box(x, y, dx, dy, wx, wy, wx+1, wy+1)
                                                for wx, wy in self.walls))
            food = segment_circle_entry(x, y, x+dx*distance, y+dy*distance, gx, gy, self.goal_radius)
            readings.append((angle, distance, food is not None))
        return readings

    def observe(self):
        rays = self.sense()
        return ([distance/self.sensor_range for _, distance, _ in rays]
                + [float(food) for _, _, food in rays]
                + [math.cos(self.fly.heading), math.sin(self.fly.heading)]
                + list(self.last_action) + [float(self.collision)])

    def step(self, action):
        if self.terminated or self.truncated:
            raise RuntimeError("episode ended; call reset()")
        try:
            values = list(action)
        except TypeError as error:
            raise ValueError("action must be [throttle 0..1, steering -1..1]") from error
        if len(values) != 2 or not all(math.isfinite(v) for v in values):
            raise ValueError("action must contain two finite numbers")
        throttle, steering = values
        if not 0 <= throttle <= 1 or not -1 <= steering <= 1:
            raise ValueError("action outside throttle 0..1 / steering -1..1")
        velocity, omega = throttle*self.speed, steering*self.turn_rate
        # Chord error is <= radius/100; inflate collision disk by that error.
        subdivisions = max(1, math.ceil(abs(omega)*self.dt/0.04),
                           math.ceil(velocity*self.dt/(self.radius/2)))
        subdt = self.dt/subdivisions
        self.collision = False
        for _ in range(subdivisions):
            x, y, angle = self.fly.x, self.fly.y, self.fly.heading
            next_angle = angle+omega*subdt
            if abs(omega) < 1e-12:
                nx, ny = x+velocity*subdt*math.cos(angle), y+velocity*subdt*math.sin(angle)
            else:
                nx = x+velocity/omega*(math.sin(next_angle)-math.sin(angle))
                ny = y+velocity/omega*(math.cos(angle)-math.cos(next_angle))
            # Sweep the entire body; a blocked translation still permits turning.
            # Small arc segments use a conservative extra radius for the chord sagitta.
            sagitta = 0.0 if abs(omega) < 1e-12 else abs(velocity/omega)*(1-math.cos(omega*subdt/2))
            original_radius = self.radius
            self.radius += sagitta
            clear = self.clear_segment(x, y, nx, ny)
            self.radius = original_radius
            if not clear:
                self.collision = True
                nx, ny = x, y
            hit = segment_circle_entry(x, y, nx, ny, self.goal[0]+0.5, self.goal[1]+0.5, self.goal_radius)
            fraction = 1.0 if hit is None else hit
            self.fly.x, self.fly.y = x+(nx-x)*fraction, y+(ny-y)*fraction
            self.fly.heading = (angle+omega*subdt*fraction+math.pi) % (2*math.pi)-math.pi
            self.elapsed += subdt*fraction
            self.path_length += velocity*subdt*fraction if clear else 0.0
            self.path.append((self.fly.x, self.fly.y))
            if hit is not None:
                self.terminated = True
                break
        self.steps += 1
        self.collisions += int(self.collision)
        self.last_action = [float(throttle), float(steering)]
        cell = (math.floor(self.fly.x), math.floor(self.fly.y))
        fresh = cell not in self.visited
        self.visited.add(cell)
        self.truncated = self.steps >= self.max_steps and not self.terminated
        reward = -0.01-(0.1 if self.collision else 0.0)
        if fresh and self.reward_mode == "shaped":
            reward += 0.02
        if self.terminated:
            reward += 10.0
        info = {"success": self.terminated, "collision": self.collision,
                "steps": self.steps, "elapsed": self.elapsed, "path_length": self.path_length,
                "collisions": self.collisions,
                "termination_reason": "food" if self.terminated else ("time_limit" if self.truncated else None)}
        return self.observe(), reward, self.terminated, self.truncated, info

    def config(self):
        return {"layout": self.layout, "difficulty": self.difficulty, "vary_maze": self.vary_maze,
                "max_steps": self.max_steps, "dt": self.dt, "speed": self.speed,
                "turn_rate": self.turn_rate, "radius": self.radius, "sensor_range": self.sensor_range,
                "goal_radius": self.goal_radius, "reward_mode": self.reward_mode,
                "sensor_version": "rays13-compass-v1"}


def from_config(config, *, seed=123, path=None, difficulty=None, vary_maze=False, max_steps=None):
    config = dict(config)
    if config.pop("sensor_version", "rays13-compass-v1") != "rays13-compass-v1":
        raise ValueError("unsupported sensor version")
    stored_difficulty = config.pop("difficulty", None)
    config.pop("vary_maze", None)
    layout = config.pop("layout")
    if path is not None and difficulty is not None:
        raise ValueError("maze and difficulty are mutually exclusive")
    if path is not None:
        layout = Path(path).read_text(encoding="utf-8-sig").splitlines()
    if vary_maze and difficulty is None:
        difficulty = stored_difficulty
    if max_steps is not None:
        config["max_steps"] = max_steps
    return ContinuousMaze(None if difficulty else layout, difficulty=difficulty,
                          vary_maze=vary_maze, seed=seed, **config)
