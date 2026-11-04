"""Stationary-policy regression tests and transparent training-only guidance."""

import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from flymaze_v2.connectome import sha256
from flymaze_v2.environment import ContinuousMaze, from_config
from flymaze_v2.experiment import episode
from flymaze_v2.guided import collect, supervised_update, train_guided
from flymaze_v2.models import build, load
from flymaze_v2.runtime import ControlSession, default_controller_path, load_controller

ROOT = Path(__file__).resolve().parent


class GuidedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_expert_demonstrations_solve_and_keep_policy_input_local(self):
        data, success = collect(ContinuousMaze(max_steps=100), 8, random.Random(42))
        self.assertEqual(success, 8)
        for observations, actions in data:
            self.assertEqual(observations.shape[1], 31)
            self.assertTrue(((actions[:, 0] >= 0) & (actions[:, 0] <= 1)).all())
            self.assertTrue(((actions[:, 1] >= -1) & (actions[:, 1] <= 1)).all())

    def test_batched_recurrent_forward_matches_individual_streams(self):
        obs = torch.tensor([ContinuousMaze(seed=1).observe(), ContinuousMaze(seed=2).observe()])
        for kind in ("connectome", "gru"):
            brain = build(kind)
            memory = brain.initial_memory().repeat(2, 1)
            mean, _, state = brain(obs, memory)
            for i in range(2):
                expected_mean, _, expected_state = brain(obs[i], memory[i:i+1])
                self.assertTrue(torch.allclose(mean[i:i+1], expected_mean, atol=1e-6))
                self.assertTrue(torch.allclose(state[i:i+1], expected_state, atol=1e-6))

    def test_supervision_updates_motor_head_but_keeps_measured_connections(self):
        brain = build("connectome")
        counts, adjacency = brain.counts.clone(), brain.adjacency.clone()
        actor = brain.actor.weight.detach().clone()
        data, _ = collect(ContinuousMaze(max_steps=100), 3, random.Random(42))
        optimizer = torch.optim.Adam(brain.parameters(), lr=0.001)
        loss = supervised_update(brain, optimizer, data, random.Random(42), batch_size=2)
        self.assertGreaterEqual(loss, 0)
        self.assertFalse(torch.equal(actor, brain.actor.weight))
        self.assertTrue(torch.equal(counts, brain.counts))
        self.assertTrue(torch.equal(adjacency, brain.adjacency))
        self.assertIsNone(brain.critic.weight.grad)

    def test_guided_training_saves_validation_selection_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"guided.pt"
            training = train_guided(ContinuousMaze(max_steps=100), build("gru"), output,
                                    updates=1, demonstrations=2, batch_size=2)
            brain, checkpoint = load(output)
            self.assertEqual(training["selected_update"], 1)
            self.assertEqual(checkpoint["training"]["algorithm"], "recurrent_behavior_cloning_dagger")
            self.assertFalse(checkpoint["training"]["critic_trained"])
            with self.assertRaises(ValueError):
                train_guided(ContinuousMaze(), brain, output, updates=1)


class DefaultModelTests(unittest.TestCase):
    def test_active_model_manifest_is_checked_and_stays_inside_models(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            target = folder/"selected.pt"
            target.write_bytes(b"unit-test checksum fixture, not a neural model")
            manifest = folder/"default_connectome.json"
            manifest.write_text(json.dumps({"format_version": 1, "model": target.name, "sha256": sha256(target)}))
            self.assertEqual(default_controller_path(folder), target.resolve())
            target.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                default_controller_path(folder)

    def test_default_model_now_reaches_food_without_any_teacher_at_runtime(self):
        brain, checkpoint, _ = load_controller(default_controller_path(ROOT/"models"))
        env = from_config(checkpoint["environment"])
        with patch("flymaze_v2.guided.DemonstrationTeacher.action", side_effect=AssertionError("runtime teacher forbidden")):
            result = episode(env, brain)
        self.assertTrue(result["success"])
        self.assertLess(result["steps"], 60)
        self.assertEqual(result["collisions"], 0)
        self.assertEqual(brain.hidden_size, 512)

    def test_stationary_warning_never_changes_motor_action(self):
        brain, checkpoint, _ = load_controller(default_controller_path(ROOT/"models"))
        env = from_config(checkpoint["environment"], max_steps=40)
        session = ControlSession(env, brain)
        memory = brain.initial_memory()
        with patch.object(brain, "act", return_value=([0.0, 1.0], None, torch.tensor(0.0), None, memory)):
            for _ in range(35):
                frame = session.step()
                self.assertEqual(frame["action"], [0.0, 1.0])
        self.assertTrue(frame["stalled"])
        self.assertGreater(session.stalled_steps, 0)
        self.assertEqual(env.path_length, 0)


class GuidedViewerTests(unittest.TestCase):
    def test_model_replacement_clears_old_state_and_shows_loaded_filename(self):
        from flymaze_v2.viewer import Viewer
        old, checkpoint, identity = load_controller(ROOT/"models"/"connectome_v2_1.pt")
        viewer = Viewer(from_config(checkpoint["environment"]), old, identity=identity, delay=10)
        try:
            viewer.root.withdraw()
            viewer.root.update()
            viewer.tick()
            viewer.replace_model(default_controller_path(ROOT/"models"))
            self.assertEqual(viewer.env.steps, 0)
            self.assertEqual(torch.count_nonzero(viewer.memory).item(), 0)
            self.assertIn("connectome_v2_1_retrained.pt", viewer.footer.cget("text"))
            viewer.tick()
            self.assertIn("Guided updates", viewer.motor_label.cget("text"))
            self.assertEqual(len(viewer.activity_canvas.find_all()), 512)
        finally:
            for callback in viewer.root.tk.call("after", "info"):
                viewer.root.after_cancel(callback)
            viewer.root.destroy()


if __name__ == "__main__":
    unittest.main()
