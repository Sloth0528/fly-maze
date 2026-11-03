"""The shared brain/body feedback loop used by GUI and headless v2.1 playback.

Every motor command comes directly from policy.act(observation, recurrent state).
There is no runtime path planner, action override, or fallback random controller.
"""

import json
import math
from pathlib import Path

import torch

from . import __version__
from .connectome import sha256
from .models import load


def load_controller(path, expected_kind="connectome"):
    path = Path(path)
    brain, checkpoint = load(path)
    if brain.kind != expected_kind:
        raise ValueError(f"expected {expected_kind} checkpoint; received {brain.kind}")
    identity = {"checkpoint": path.name, "checkpoint_sha256": sha256(path),
                "checkpoint_version": checkpoint["project_version"],
                "training": checkpoint["training"]}
    return brain, checkpoint, identity


class ControlSession:
    def __init__(self, env, brain, *, sample=False, identity=None, record=True):
        self.env, self.brain = env, brain
        if brain.observation_size != env.observation_size:
            raise ValueError("brain sensory input does not match environment")
        self.brain.eval()
        self.sample, self.identity, self.record = sample, dict(identity or {}), record
        self.memory = brain.initial_memory()
        self.observation = env.observe()
        self.frames, self.latest = [], None
        self.total_reward = 0.0
        self.start_environment = env.config()

    def reset(self, random_start=False):
        self.observation = self.env.reset(random_start)
        self.memory = self.brain.initial_memory()
        self.frames, self.latest, self.total_reward = [], None, 0.0
        self.start_environment = self.env.config()

    def step(self):
        if self.env.terminated or self.env.truncated:
            raise RuntimeError("episode ended; reset the brain/body session")
        observation = list(self.observation)
        pose_before = [self.env.fly.x, self.env.fly.y, self.env.fly.heading]
        with torch.inference_mode():
            action, _, value, _, memory = self.brain.act(observation, self.memory, greedy=not self.sample)
            if memory.shape != (1, self.brain.hidden_size) or not torch.isfinite(memory).all() or not torch.isfinite(value).all():
                raise FloatingPointError("invalid brain state or value; motor update stopped")
            activity = memory.flatten().tolist()
            magnitude = memory.abs().flatten()
            top_indices = torch.topk(magnitude, k=min(5, self.brain.hidden_size)).indices.tolist()
            node_ids = getattr(self.brain, "root_ids", [f"GRU-{i}" for i in range(self.brain.hidden_size)])
            top_nodes = [{"index": i, "root_id": node_ids[i], "state": activity[i]} for i in top_indices]
            value_number = value.item()
            rms = math.sqrt(sum(v*v for v in activity)/len(activity))
        # Environment validates the unmodified policy action before changing the body.
        next_observation, reward, terminated, truncated, info = self.env.step(action)
        self.memory, self.observation = memory, next_observation
        self.total_reward += reward
        frame = {"step": self.env.steps, "observation": observation, "pose_before": pose_before,
                 "action": list(action), "velocity": action[0]*self.env.speed,
                 "angular_velocity": action[1]*self.env.turn_rate,
                 "pose_after": [self.env.fly.x, self.env.fly.y, self.env.fly.heading],
                 "state_rms": rms, "top_nodes": top_nodes, "value": value_number,
                 "reward": reward, "collision": info["collision"], "sim_seconds": self.env.elapsed,
                 "terminated": terminated, "truncated": truncated}
        self.latest = frame
        if self.record:
            self.frames.append(frame)
        return frame

    def report(self):
        brain = self.brain
        graph = None
        if brain.kind == "connectome":
            graph = {"nodes": brain.hidden_size, "edges": torch.count_nonzero(brain.counts).item(),
                     "synapses": int(brain.counts.sum().item()), "provenance": brain.metadata,
                     "model_type": "fixed unsigned connectome-constrained rate network"}
        return {"project_version": __version__, "controller": brain.kind,
                "command_source": "brain.act only", "action_mode": "sample" if self.sample else "deterministic",
                "identity": self.identity, "graph": graph, "environment": self.start_environment,
                "initial_pose": self.env.initial_pose, "map_seed": self.env.map_seed,
                "result": {"success": self.env.terminated, "truncated": self.env.truncated,
                           "steps": self.env.steps, "sim_seconds": self.env.elapsed,
                           "path_length": self.env.path_length, "collisions": self.env.collisions,
                           "reward": self.total_reward}, "frames": list(self.frames),
                "interpretation": "Abstract recurrent activity, not recorded biological spikes. "
                                  "Closed-loop execution does not establish reliable maze navigation."}

    def export(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(self.report(), handle, ensure_ascii=False, indent=2, allow_nan=False)
