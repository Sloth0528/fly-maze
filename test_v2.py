"""v2 behavior tests: geometry, sensing, real graph import, training and paired evaluation."""

import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from flymaze_v2.connectome import prepare, load_graph
from flymaze_v2.environment import ContinuousMaze, from_config
from flymaze_v2.experiment import train, evaluate_models, episode
from flymaze_v2.geometry import segment_box_sq
from flymaze_v2.models import GRUPolicy, ConnectomePolicy, build, save, load

OPEN = ["########", "#S.....#", "#......#", "#.....G#", "########"]


class ContinuousTests(unittest.TestCase):
    def test_diagonal_distance_and_time_equal_axial(self):
        straight, diagonal = ContinuousMaze(OPEN), ContinuousMaze(OPEN)
        diagonal.fly.heading = math.pi/4
        straight.step([1, 0])
        diagonal.step([1, 0])
        self.assertAlmostEqual(diagonal.fly.x-1.5, 0.15/math.sqrt(2))
        self.assertAlmostEqual(diagonal.fly.y-1.5, 0.15/math.sqrt(2))
        self.assertAlmostEqual(straight.path_length, diagonal.path_length)
        self.assertAlmostEqual(diagonal.path_length, 0.15)
        self.assertAlmostEqual(straight.elapsed, diagonal.elapsed)

    def test_in_place_turn_and_smooth_arc(self):
        env = ContinuousMaze(OPEN)
        env.step([0, 0.25])
        self.assertAlmostEqual(env.fly.x, 1.5)
        self.assertAlmostEqual(env.fly.heading, math.pi*0.025)
        env = ContinuousMaze(OPEN)
        env.step([1, 1])
        self.assertAlmostEqual(env.fly.x, 1.5+1.5/math.pi*math.sin(math.pi*0.1))
        self.assertAlmostEqual(env.fly.y, 1.5+1.5/math.pi*(1-math.cos(math.pi*0.1)))
        self.assertAlmostEqual(env.path_length, 0.15)

    def test_high_speed_cannot_tunnel_through_wall(self):
        env = ContinuousMaze(["########", "#S.#..G#", "#......#", "########"], speed=50)
        env.step([1, 0])
        self.assertTrue(env.collision)
        self.assertLessEqual(env.fly.x+env.radius, 3)
        self.assertTrue(env.clear_segment(env.fly.x, env.fly.y, env.fly.x, env.fly.y))

    def test_closed_corner_and_body_clearance(self):
        env = ContinuousMaze(OPEN)
        # The outside corner of a wall cannot be crossed by the disk.
        env._set_layout(["########", "#S#....#", "#......#", "#.....G#", "########"])
        self.assertFalse(env.clear_segment(1.8, 1.8, 2.2, 2.2))
        self.assertFalse(env.clear_segment(1.85, 1.5, 1.85, 2.5))
        self.assertTrue(env.clear_segment(1.7, 1.5, 1.7, 2.5))
        self.assertGreater(segment_box_sq(1.7, 2.3, 1.8, 2.2, 2, 1, 3, 2), 0.16**2)

    def test_observation_and_rotation(self):
        env = ContinuousMaze(OPEN)
        obs = env.observe()
        self.assertEqual(len(obs), 31)
        self.assertTrue(all(math.isfinite(x) for x in obs))
        front = env.sense()[6][1]
        env.fly.heading = math.pi/2
        self.assertLess(env.sense()[6][1], front)
        self.assertEqual(env.observe()[26:28], [math.cos(math.pi/2), math.sin(math.pi/2)])

    def test_food_and_walls_are_occluded(self):
        first = ["########", "#S.#..G#", "#..#...#", "#......#", "########"]
        second = ["########", "#S.#...#", "#..#..G#", "#......#", "########"]
        a, b = ContinuousMaze(first), ContinuousMaze(second)
        self.assertEqual(a.observe(), b.observe())
        self.assertFalse(any(hit for _, _, hit in a.sense()))

    def test_food_visible_in_open_corridor(self):
        env = ContinuousMaze(["######", "#S.G.#", "######"])
        self.assertTrue(env.sense()[6][2])

    def test_success_detected_along_sweep_and_last_step(self):
        env = ContinuousMaze(["########", "#S.G...#", "########"], speed=40, max_steps=1)
        _, _, terminated, truncated, info = env.step([1, 0])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "food")
        self.assertAlmostEqual(env.fly.x, 3.25)
        self.assertAlmostEqual(env.elapsed, 1.75/40)
        with self.assertRaises(RuntimeError):
            env.step([0, 0])

    def test_timeout_and_invalid_actions_are_atomic(self):
        env = ContinuousMaze(OPEN, max_steps=1)
        for action in ([float("nan"), 0], [0, float("inf")], [2, 0], [0, -2], [1], 1):
            with self.assertRaises(ValueError):
                env.step(action)
            self.assertEqual(env.steps, 0)
        _, _, terminated, truncated, _ = env.step([0, 0])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertAlmostEqual(env.elapsed, env.dt)

    def test_random_maps_and_starts_reproduce(self):
        a = ContinuousMaze(difficulty="easy", vary_maze=True, seed=41)
        b = ContinuousMaze(difficulty="easy", vary_maze=True, seed=41)
        maps = set()
        for _ in range(4):
            self.assertEqual(a.reset(True), b.reset(True))
            self.assertEqual(a.initial_pose, b.initial_pose)
            self.assertNotEqual((math.floor(a.fly.x), math.floor(a.fly.y)), a.goal)
            self.assertEqual(a.layout, b.layout)
            maps.add(tuple(a.layout))
        self.assertGreater(len(maps), 1)

    def test_configuration_roundtrip_and_invalid_values(self):
        env = ContinuousMaze(OPEN, dt=0.15, radius=0.2, speed=2.3, reward_mode="sparse")
        self.assertEqual(env.config(), from_config(env.config()).config())
        for options in ({"radius": 0}, {"dt": 0}, {"speed": float("inf")}, {"sensor_range": -1},
                        {"max_steps": 0}, {"max_steps": 2.5}, {"vary_maze": True}, {"goal_radius": 0.5}):
            with self.assertRaises(ValueError):
                ContinuousMaze(OPEN, **options)
        with self.assertRaises(ValueError):
            ContinuousMaze(["S.G", "..."])

    def test_default_maze_has_a_physically_executable_route(self):
        # Full-map waypoints are a test oracle, never policy observations.
        env = ContinuousMaze(max_steps=200)
        for tx, ty in ((3.5, 3.5), (3.5, 1.5)):
            for _ in range(100):
                dx, dy = tx-env.fly.x, ty-env.fly.y
                if env.terminated or math.hypot(dx, dy) < 1e-8:
                    break
                delta = (math.atan2(dy, dx)-env.fly.heading+math.pi) % (2*math.pi)-math.pi
                if abs(delta) > 1e-8:
                    env.step([0, max(-1, min(1, delta/(env.turn_rate*env.dt)))])
                else:
                    env.step([min(1, math.hypot(dx, dy)/(env.speed*env.dt)), 0])
        self.assertTrue(env.terminated)
        self.assertEqual(env.collisions, 0)

    def test_sparse_reward_does_not_hint_hidden_goal(self):
        a = ContinuousMaze(OPEN, reward_mode="sparse")
        b = ContinuousMaze(["########", "#S.....#", "#.....G#", "#......#", "########"], reward_mode="sparse")
        self.assertEqual(a.step([1, 0])[1], b.step([1, 0])[1])


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def small_connectome(self):
        # Synthetic three-node unit-test fixture ONLY; production requires measured NPZ data.
        return ConnectomePolicy(np.array([[0, 2, 0], [1, 0, 4], [0, 3, 0]]), ["1", "2", "3"], {"fixture": True})

    def test_measured_import_direction_aggregation_and_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source, output = folder/"fixture.csv", folder/"graph.npz"
            source.write_text("pre_root_id,post_root_id,neuropil,syn_count,nt_type\n"
                              "1,2,EB,3,ACH\n1,2,FB,4,ACH\n2,1,PB,2,GABA\n"
                              "3,1,OTHER,100,ACH\n", encoding="utf-8")
            prepare(output, source=source, nodes=2)
            counts, ids, metadata = load_graph(output)
            self.assertEqual(ids, ["1", "2"])
            self.assertEqual(counts.tolist(), [[0, 2], [7, 0]])
            self.assertEqual(metadata["matrix_axes"], "counts[postsynaptic, presynaptic]")
            with output.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_graph(output)

    def test_bundled_real_graph(self):
        counts, ids, metadata = load_graph("data/flywire_cx_512.npz")
        self.assertEqual(len(ids), 512)
        self.assertEqual(np.count_nonzero(counts), 16212)
        self.assertEqual(int(counts.sum()), 294654)
        self.assertIn("FlyWire FAFB v783", metadata["dataset"])
        brain = build("connectome")
        self.assertFalse(brain.adjacency.requires_grad)
        self.assertEqual(torch.count_nonzero(brain.counts).item(), 16212)
        with self.assertRaises(FileNotFoundError):
            build("connectome", connectome="missing-measured-data.npz")

    def test_bounded_actions_gradients_and_recurrent_connectivity(self):
        observation = ContinuousMaze(OPEN).observe()
        for brain in (GRUPolicy(), self.small_connectome()):
            memory = brain.initial_memory()
            for _ in range(4):
                action, log_prob, value, entropy, memory = brain.act(observation, memory)
                self.assertTrue(0 <= action[0] <= 1 and -1 <= action[1] <= 1)
            loss = -log_prob+value.square()-0.01*entropy
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in brain.parameters() if p.grad is not None))
            self.assertGreater(brain.encoder[0].weight.grad.abs().sum().item() if brain.kind == "gru"
                               else brain.encoder.weight.grad.abs().sum().item(), 0)
        brain = self.small_connectome()
        before = brain(observation, torch.ones(1, 3))[2]
        brain.adjacency.zero_()
        after = brain(observation, torch.ones(1, 3))[2]
        self.assertFalse(torch.allclose(before, after))

    def test_training_updates_weights_and_graph_stays_fixed(self):
        for brain in (GRUPolicy(), self.small_connectome()):
            with tempfile.TemporaryDirectory() as folder:
                before = {key: value.clone() for key, value in brain.state_dict().items()}
                env = ContinuousMaze(OPEN, max_steps=5)
                train(env, brain, Path(folder)/"brain.pt", episodes=2, rollout=3)
                self.assertTrue(any(not torch.equal(before[key], value) for key, value in brain.named_parameters()))
                if brain.kind == "connectome":
                    self.assertTrue(torch.equal(brain.counts, before["counts"]))
                    self.assertTrue(torch.equal(brain.adjacency, before["adjacency"]))

    def test_checkpoint_is_self_contained_and_eval_preserves_weights(self):
        env = ContinuousMaze(OPEN, max_steps=3)
        for brain in (GRUPolicy(), self.small_connectome()):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/"brain.pt"
                save(path, brain, env, {"seed": 7})
                restored, checkpoint = load(path)
                expected = brain(env.observe(), brain.initial_memory())[0]
                actual = restored(env.observe(), restored.initial_memory())[0]
                self.assertTrue(torch.equal(expected, actual))
                before = {key: value.clone() for key, value in restored.state_dict().items()}
                episode(env, restored)
                for key, value in restored.state_dict().items():
                    self.assertTrue(torch.equal(before[key], value))
                with self.assertRaises(FileExistsError):
                    save(path, brain, env, {})

    def test_paired_comparison_and_body_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            a, b = folder/"a.pt", folder/"b.pt"
            env = ContinuousMaze(difficulty="easy", vary_maze=True, max_steps=2)
            save(a, GRUPolicy(), env, {"seed": 7})
            save(b, self.small_connectome(), env, {"seed": 7})
            report = evaluate_models([a, b], episodes=3, seed=10007, random_start=True,
                                     difficulty="easy", vary_maze=True, output=folder/"report.json")
            first, second = report["models"]
            for x, y in zip(first["episodes"], second["episodes"]):
                self.assertEqual(x["initial_pose"], y["initial_pose"])
                self.assertEqual(x["map_seed"], y["map_seed"])
            self.assertIsNone(first["summary"]["mean_sim_seconds_success"])
            self.assertTrue((folder/"report.json").is_file())
            c = folder/"c.pt"
            save(c, GRUPolicy(), ContinuousMaze(OPEN, speed=2), {})
            with self.assertRaisesRegex(ValueError, "mismatch"):
                evaluate_models([a, c], difficulty="easy", max_steps=2, episodes=1)


class ViewerTests(unittest.TestCase):
    def close(self, viewer):
        for callback in viewer.root.tk.call("after", "info"):
            viewer.root.after_cancel(callback)
        viewer.root.destroy()

    def test_manual_window_drawing_and_key_mapping(self):
        from flymaze_v2.viewer import Viewer
        viewer = Viewer(ContinuousMaze(OPEN), delay=10)
        try:
            viewer.root.withdraw()
            viewer.root.update()
            viewer.press(SimpleNamespace(keysym="Up"))
            viewer.press(SimpleNamespace(keysym="Right"))
            viewer.tick()
            self.assertGreater(viewer.env.fly.x, 1.5)
            self.assertGreater(viewer.env.fly.y, 1.5)
            self.assertGreater(viewer.env.fly.heading, 0)
            self.assertIn("sensors=31", viewer.label.cget("text"))
            oval = [item for item in viewer.canvas.find_all() if viewer.canvas.type(item) == "oval"]
            self.assertEqual(len(oval), 1)
            x1, _, x2, _ = viewer.canvas.coords(oval[0])
            self.assertAlmostEqual(x2-x1, 2*viewer.env.radius*viewer.cell)
            viewer.press(SimpleNamespace(keysym="r"))
            self.assertEqual(viewer.env.steps, 0)
        finally:
            self.close(viewer)

    def test_model_windows_play_without_learning(self):
        from flymaze_v2.viewer import Viewer
        for brain in (GRUPolicy(), ModelTests().small_connectome()):
            viewer = Viewer(ContinuousMaze(OPEN, max_steps=2), brain, delay=10)
            try:
                viewer.root.withdraw()
                viewer.root.update()
                before = {key: value.clone() for key, value in brain.state_dict().items()}
                viewer.tick()
                viewer.tick()
                self.assertTrue(viewer.env.truncated)
                self.assertIn("TIME LIMIT", viewer.label.cget("text"))
                self.assertTrue(all(torch.equal(value, before[key]) for key, value in brain.state_dict().items()))
            finally:
                self.close(viewer)


if __name__ == "__main__":
    unittest.main()
