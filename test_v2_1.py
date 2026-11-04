"""v2.1 tests: causal brain/body wiring, telemetry, launch defaults and GUI integration."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from flymaze_v2 import __version__
from flymaze_v2.connectome import load_graph
from flymaze_v2.environment import from_config
from flymaze_v2.runtime import ControlSession, load_controller
from flymaze_v2.viewer import Viewer

ROOT = Path(__file__).resolve().parent
MODEL = ROOT/"models"/"connectome_v2_1.pt"


class BrainLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setup_session(self, max_steps=6):
        brain, checkpoint, identity = load_controller(MODEL)
        env = from_config(checkpoint["environment"], max_steps=max_steps)
        return ControlSession(env, brain, identity=identity)

    def test_bundled_brain_matches_measured_graph_and_version(self):
        session = self.setup_session()
        counts, ids, metadata = load_graph(ROOT/"data"/"flywire_cx_512.npz")
        self.assertEqual(__version__, "2.1.0")
        self.assertEqual(session.identity["checkpoint_version"], "2.1.0")
        self.assertEqual(session.brain.kind, "connectome")
        self.assertTrue(np.array_equal(session.brain.counts.numpy(), counts))
        self.assertEqual(session.brain.root_ids, ids)
        self.assertEqual(session.brain.metadata["source_sha256"], metadata["source_sha256"])
        self.assertEqual(session.identity["training"]["episodes"], 120)

    def test_motor_action_is_exactly_the_brain_output(self):
        session = self.setup_session()
        brain = session.brain
        previous_observation = session.env.observe()
        with torch.inference_mode():
            expected, _, _, _, expected_memory = brain.act(previous_observation, session.memory, greedy=True)
        frame = session.step()
        self.assertEqual(frame["observation"], previous_observation)
        self.assertEqual(frame["action"], expected)
        self.assertEqual(session.env.last_action, expected)
        self.assertTrue(torch.equal(expected_memory, session.memory))
        self.assertEqual(frame["pose_after"], [session.env.fly.x, session.env.fly.y, session.env.fly.heading])

    def test_environment_feedback_and_recurrent_state_reach_next_step(self):
        session = self.setup_session()
        first = session.step()
        expected_observation = session.env.observe()
        first_memory = session.memory.clone()
        with torch.inference_mode():
            expected_action, _, _, _, expected_memory = session.brain.act(expected_observation, first_memory, greedy=True)
        second = session.step()
        self.assertEqual(second["observation"], expected_observation)
        self.assertEqual(second["action"], expected_action)
        self.assertTrue(torch.equal(expected_memory, session.memory))
        self.assertNotEqual(first["observation"], second["observation"])
        self.assertFalse(torch.equal(first_memory, session.memory))

    def test_recorded_activity_is_actual_state_and_does_not_train(self):
        session = self.setup_session()
        before = {key: value.clone() for key, value in session.brain.state_dict().items()}
        frame = session.step()
        for node in frame["top_nodes"]:
            self.assertEqual(node["root_id"], session.brain.root_ids[node["index"]])
            self.assertEqual(node["state"], session.memory.flatten()[node["index"]].item())
        self.assertGreater(frame["state_rms"], 0)
        self.assertFalse(session.memory.requires_grad)
        self.assertTrue(all(torch.equal(value, before[key]) for key, value in session.brain.state_dict().items()))

    def test_measured_recurrent_connections_influence_next_motor_command(self):
        session = self.setup_session()
        session.step()
        with torch.inference_mode():
            real = session.brain.act(session.observation, session.memory, greedy=True)[0]
            session.brain.adjacency.zero_()  # explicit test-only lesion; never part of playback
            lesioned = session.brain.act(session.observation, session.memory, greedy=True)[0]
        self.assertGreater(max(abs(a-b) for a, b in zip(real, lesioned)), 1e-8)

    def test_reset_clears_memory_and_trace(self):
        session = self.setup_session(max_steps=1)
        session.step()
        with self.assertRaises(RuntimeError):
            session.step()
        session.reset()
        self.assertEqual(session.env.steps, 0)
        self.assertEqual(session.frames, [])
        self.assertIsNone(session.latest)
        self.assertEqual(torch.count_nonzero(session.memory).item(), 0)

    def test_export_is_auditable_and_never_overwrites(self):
        session = self.setup_session()
        session.step()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/"trace.json"
            session.export(target)
            report = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(report["command_source"], "brain.act only")
            self.assertEqual(report["graph"]["edges"], 16212)
            self.assertEqual(report["result"]["steps"], 1)
            self.assertEqual(report["frames"][0]["action"], session.env.last_action)
            self.assertEqual(len(report["identity"]["checkpoint_sha256"]), 64)
            with self.assertRaises(FileExistsError):
                session.export(target)

    def test_wrong_brain_or_missing_brain_never_falls_back(self):
        with self.assertRaisesRegex(ValueError, "expected connectome"):
            load_controller(ROOT/"models"/"gru_v2_1.pt")
        with self.assertRaises(FileNotFoundError):
            load_controller(ROOT/"models"/"missing.pt")

    def test_nonfinite_neural_state_stops_before_motor_update(self):
        session = self.setup_session()
        bad = session.memory.clone()
        bad[0, 0] = float("nan")
        with patch.object(session.brain, "act", return_value=([1, 0], None, torch.tensor(0.0), None, bad)):
            with self.assertRaises(FloatingPointError):
                session.step()
        self.assertEqual(session.env.steps, 0)

    def test_launcher_works_from_an_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/"report.json"
            result = subprocess.run([sys.executable, str(ROOT/"run_v2_1.py"), "--headless", "--max-steps", "3",
                                     "--report", str(target)], cwd=folder, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            report = json.loads(target.read_text(encoding="utf-8"))
            episode = report["episodes"][0]
            self.assertEqual(episode["controller"], "connectome")
            self.assertEqual(len(episode["frames"]), 3)
            self.assertEqual(episode["graph"]["nodes"], 512)


class ConnectedViewerTests(unittest.TestCase):
    def test_auto_gui_uses_same_session_and_shows_real_states(self):
        session = BrainLoopTests().setup_session(max_steps=4)
        viewer = Viewer(session.env, session.brain, identity=session.identity, delay=10)
        try:
            viewer.root.withdraw()
            viewer.root.update()
            viewer.press(SimpleNamespace(keysym="Left"))
            viewer.press(SimpleNamespace(keysym="Up"))
            with torch.inference_mode():
                expected = viewer.brain.act(viewer.env.observe(), viewer.memory, greedy=True)[0]
            viewer.tick()
            self.assertEqual(viewer.env.last_action, expected)
            self.assertEqual(len(viewer.activity_canvas.find_all()), 512)
            self.assertIn("brain.act", viewer.motor_label.cget("text"))
            self.assertIn("120", viewer.motor_label.cget("text"))
            self.assertIn("512", viewer.brain_info.cget("text"))
            self.assertEqual(viewer.session.latest["step"], viewer.env.steps)
            viewer.inspect_node(SimpleNamespace(x=1, y=1))
            self.assertIn(viewer.brain.root_ids[0], viewer.node_label.cget("text"))
            before = viewer.env.steps
            viewer.toggle_pause()
            viewer.tick()
            self.assertEqual(viewer.env.steps, before)
        finally:
            for callback in viewer.root.tk.call("after", "info"):
                viewer.root.after_cancel(callback)
            viewer.root.destroy()


if __name__ == "__main__":
    unittest.main()
