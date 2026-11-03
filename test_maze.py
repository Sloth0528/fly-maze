"""環境契約測試：地圖、感知遮擋、獎勵、動作與回合結束。"""

from collections import deque
import unittest

from fly import DIRECTIONS, FORWARD, LEFT, RIGHT
from maze import Maze, make_maze, maze_from_config
from maze_layouts import DIFFICULTIES, generate_layout


def oracle_actions(env):
    """測試專用全圖搜尋，供驗證可解性；不屬於模型策略。"""
    initial = (env.fly.x, env.fly.y, env.fly.heading)
    queue = deque([(initial, [])])
    seen = {initial}
    while queue:
        (x, y, heading), actions = queue.popleft()
        if (x, y) == env.goal:
            return actions
        dx, dy = DIRECTIONS[heading]
        candidates = [(LEFT, (x, y, (heading - 1) % 4)),
                      (RIGHT, (x, y, (heading + 1) % 4))]
        if not env.is_wall(x + dx, y + dy):
            candidates.append((FORWARD, (x + dx, y + dy, heading)))
        for action, state in candidates:
            if state not in seen:
                seen.add(state)
                queue.append((state, actions + [action]))
    raise AssertionError("Unreachable goal")


class MazeTests(unittest.TestCase):
    def test_generated_maps_solvable_within_budget(self):
        for difficulty in DIFFICULTIES:
            for seed in range(30):
                with self.subTest(difficulty=difficulty, seed=seed):
                    env = make_maze(difficulty=difficulty, seed=seed)
                    floor = {(x, y) for y, row in enumerate(env.layout)
                             for x, tile in enumerate(row) if tile != "#"}
                    self.assertEqual(floor, set(env.start_candidates) | {env.goal})
                    self.assertTrue(all(c == "#" for c in env.layout[0] + env.layout[-1]))
                    self.assertTrue(all(r[0] == r[-1] == "#" for r in env.layout))
                    actions = oracle_actions(env)
                    self.assertEqual(env.optimal_steps, len(actions))
                    self.assertLessEqual(len(actions), env.max_steps)
                    for action in actions:
                        observation, _, terminated, truncated, _ = env.step(action)
                        self.assertEqual(len(observation), 83)
                        self.assertFalse(truncated)
                    self.assertTrue(terminated)

    def test_seed_and_map_sequence(self):
        self.assertEqual(generate_layout("hard", 42), generate_layout("hard", 42))
        self.assertNotEqual(generate_layout("hard", 42), generate_layout("hard", 43))
        a = make_maze(difficulty="easy", seed=8, vary_maze=True)
        b = make_maze(difficulty="easy", seed=8, vary_maze=True)
        layouts = set()
        for _ in range(5):
            self.assertEqual(a.reset(random_start=True), b.reset(random_start=True))
            self.assertEqual(a.layout, b.layout)
            layouts.add(tuple(a.layout))
        self.assertGreater(len(layouts), 1)

    def test_fixed_layout_does_not_change(self):
        env = make_maze(difficulty="medium", seed=12)
        layout = list(env.layout)
        for _ in range(3):
            env.reset(random_start=True)
            self.assertEqual(env.layout, layout)
            self.assertNotEqual((env.fly.x, env.fly.y), env.goal)

    def test_goal_hidden_by_wall(self):
        env = Maze(["#####", "#S#G#", "#...#", "#####"], observation_mode="egocentric")
        observation = env.observe()
        self.assertEqual(sum(observation[50:75]), 0)
        self.assertTrue(env.line_of_sight(2, 1))  # 第一面牆可見。
        self.assertFalse(env.line_of_sight(3, 1))
        self.assertFalse(env.line_of_sight(2, 2))  # 不斜穿封閉牆角。
        target = next(i for i, (x, y, _) in enumerate(env.local_cells()) if (x, y) == env.goal)
        self.assertEqual([observation[c * 25 + target] for c in range(3)], [0, 0, 0])
        legacy = Maze(env.layout)
        self.assertEqual(sum(legacy.observe()[50:75]), 1)

    def test_egocentric_front_follows_heading(self):
        env = Maze(["#######", "#.....#", "#..S.G#", "#.....#", "#######"],
                   observation_mode="egocentric")
        for heading, (dx, dy) in enumerate(DIRECTIONS):
            env.fly.heading = heading
            x, y, visible = env.local_cells()[7]  # 上方正前一格。
            self.assertEqual((x, y), (env.fly.x + dx, env.fly.y + dy))
            self.assertTrue(visible)
            self.assertEqual(env.observe()[75:79], [float(i == heading) for i in range(4)])

    def test_collision_and_action_feedback(self):
        env = Maze(["#####", "#S#G#", "#...#", "#####"], reward_mode="shaped")
        observation, reward, _, _, info = env.step(FORWARD)
        self.assertEqual((env.fly.x, env.fly.y), env.start)
        self.assertAlmostEqual(reward, -.11)
        self.assertTrue(info["collision"])
        self.assertEqual(observation[-4:], [1., 0., 0., 1.])
        env.step(RIGHT)
        self.assertEqual(env.collisions, 1)

    def test_no_repeat_exploration_reward(self):
        env = Maze(["#######", "#S...G#", "#######"], reward_mode="shaped")
        self.assertAlmostEqual(env.step(FORWARD)[1], .01)
        self.assertAlmostEqual(env.step(LEFT)[1], -.01)
        self.assertAlmostEqual(env.step(LEFT)[1], -.01)
        self.assertAlmostEqual(env.step(FORWARD)[1], -.01)
        env.reset()
        self.assertEqual(env.collisions, 0)
        self.assertEqual(len(env.visited), 1)
        self.assertAlmostEqual(env.step(FORWARD)[1], .01)

    def test_sparse_mode_and_no_hidden_goal_hint(self):
        layout = ["#######", "#S...G#", "#######"]
        sparse = Maze(layout, reward_mode="sparse")
        self.assertAlmostEqual(sparse.step(FORWARD)[1], -.01)
        shaped = Maze(layout, reward_mode="shaped")
        shaped.step(FORWARD)
        self.assertAlmostEqual(shaped.step(FORWARD)[1], .01)

    def test_success_on_last_allowed_action(self):
        env = Maze(["####", "#SG#", "####"], max_steps=1)
        _, reward, terminated, truncated, info = env.step(FORWARD)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 10)
        self.assertEqual(info["termination_reason"], "food")
        with self.assertRaises(RuntimeError):
            env.step(FORWARD)

    def test_time_limit(self):
        env = Maze(["#####", "#S.G#", "#####"], max_steps=1)
        _, _, terminated, truncated, info = env.step(LEFT)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["termination_reason"], "time_limit")

    def test_checkpoint_environment_settings(self):
        original = make_maze(difficulty="easy", seed=7, reward_mode="sparse")
        restored = maze_from_config(original.config())
        self.assertEqual(restored.observe(), original.observe())
        self.assertEqual(restored.reward_mode, "sparse")
        legacy = maze_from_config({"layout": original.layout, "max_steps": 80})
        self.assertEqual((legacy.observation_mode, legacy.reward_mode), ("world", "legacy"))

    def test_invalid_configuration(self):
        for kwargs in ({"difficulty": "unknown"}, {"vary_maze": True},
                       {"path": "anything.txt", "difficulty": "easy"},
                       {"observation_mode": "unknown"}, {"reward_mode": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_maze(**kwargs)
        with self.assertRaises(ValueError):
            Maze(["#####", "#S#G#", "#####"])


if __name__ == "__main__":
    unittest.main()
