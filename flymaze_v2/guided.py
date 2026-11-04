"""Training-only full-map teacher and recurrent imitation/DAgger warm-up.

Playback uses only the learned policy and its original 31-dimensional observation.
The teacher is privileged supervision, not a biological learning model or runtime planner.
"""

from collections import deque
import csv
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from . import __version__
from .environment import from_config
from .experiment import episode, parameter_count, summarize
from .models import save


class DemonstrationTeacher:
    def __init__(self, env):
        self.env = env
        distances = {env.goal: 0}
        queue = deque([env.goal])
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                nx, ny = point = (x+dx, y+dy)
                if (0 <= nx < env.width and 0 <= ny < env.height
                        and env.layout[ny][nx] != "#" and point not in distances):
                    distances[point] = distances[(x, y)]+1
                    queue.append(point)
        self.distances = distances

    def action(self):
        env, body = self.env, self.env.fly
        cell = (math.floor(body.x), math.floor(body.y))
        if cell == env.goal:
            tx, ty = env.goal[0]+0.5, env.goal[1]+0.5
        else:
            candidates = [(cell[0]+dx, cell[1]+dy) for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1))]
            next_cell = min((p for p in candidates if p in self.distances), key=self.distances.get)
            tx, ty = next_cell[0]+0.5, next_cell[1]+0.5
            if not env.clear_segment(body.x, body.y, tx, ty):
                tx, ty = cell[0]+0.5, cell[1]+0.5
        delta = (math.atan2(ty-body.y, tx-body.x)-body.heading+math.pi) % (2*math.pi)-math.pi
        steering = max(-1.0, min(1.0, delta/(env.turn_rate*env.dt)))
        throttle = 0.95*max(0.0, 1.0-abs(delta)/0.35)
        throttle *= min(1.0, math.hypot(tx-body.x, ty-body.y)/(env.speed*env.dt))
        return [throttle, steering]


def collect(env, count, rng, *, brain=None, teacher_probability=1.0):
    trajectories = []
    successes = 0
    for index in range(count):
        observation = env.reset(random_start=index != 0)
        # Start perturbations are training-only domain randomization, not extra observations.
        if index != 0:
            nx, ny = env.fly.x+rng.uniform(-0.10, 0.10), env.fly.y+rng.uniform(-0.10, 0.10)
            if env.clear_segment(nx, ny, nx, ny):
                env.fly.x, env.fly.y = nx, ny
                observation = env.observe()
        teacher = DemonstrationTeacher(env)
        memory = brain.initial_memory() if brain is not None else None
        observations, targets = [], []
        while not (env.terminated or env.truncated):
            target = teacher.action()
            observations.append(observation)
            targets.append(target)
            action = target
            if brain is not None:
                with torch.inference_mode():
                    policy_action, _, _, _, memory = brain.act(observation, memory, greedy=True)
                if rng.random() >= teacher_probability:
                    action = policy_action
            observation, _, _, _, _ = env.step(action)
        successes += int(env.terminated)
        trajectories.append((np.asarray(observations, dtype=np.float32), np.asarray(targets, dtype=np.float32)))
    return trajectories, successes


def supervised_update(brain, optimizer, trajectories, rng, batch_size=8):
    chosen = rng.choices(trajectories, k=batch_size)
    length = max(len(item[0]) for item in chosen)
    observations = np.zeros((length, batch_size, brain.observation_size), dtype=np.float32)
    targets = np.zeros((length, batch_size, 2), dtype=np.float32)
    mask = np.zeros((length, batch_size), dtype=np.float32)
    for batch, (obs, actions) in enumerate(chosen):
        observations[:len(obs), batch], targets[:len(obs), batch] = obs, actions
        mask[:len(obs), batch] = 1
    observations, targets, mask = (torch.from_numpy(value) for value in (observations, targets, mask))
    memory = brain.log_std.new_zeros(batch_size, brain.hidden_size)
    losses = []
    for step in range(length):
        mean, _, memory = brain(observations[step], memory)
        predicted = torch.stack(((mean[:, 0].tanh()+1)/2, mean[:, 1].tanh()), dim=1)
        error = (predicted-targets[step]).square()*predicted.new_tensor([1.0, 2.0])
        losses.append((error.sum(dim=1)*mask[step]).sum())
    loss = torch.stack(losses).sum()/mask.sum()
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite guided loss")
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return loss.item()


def validate(config, brain, *, seed=20021, episodes=12):
    env = from_config(config, seed=seed)
    brain.eval()
    results = [episode(env, brain, random_start=True) for _ in range(episodes)]
    fixed = from_config(config, seed=seed)
    fixed_result = episode(fixed, brain, random_start=False)
    return summarize(results), fixed_result


def train_guided(env, brain, output, *, seed=42, updates=300, demonstrations=48, batch_size=8,
                 lr=0.001, validate_every=25, validation_seed=20021):
    output = Path(output)
    log_path = output.with_suffix(".csv")
    if output.suffix != ".pt" or output.exists() or log_path.exists():
        raise ValueError("choose unused .pt and .csv output names")
    if updates < 1 or demonstrations < 1 or batch_size < 1 or validate_every < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("invalid guided training settings")
    if env.vary_maze:
        raise ValueError("guided v2.1 warm-up currently supports a fixed map only")
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    config = env.config()
    trajectories, expert_success = collect(env, demonstrations, rng)
    if expert_success < demonstrations:
        raise ValueError(f"teacher could not solve all demonstrations ({expert_success}/{demonstrations}); check geometry/budget")
    optimizer = torch.optim.Adam(brain.parameters(), lr=lr)
    best_key, best_state, best_summary, best_update = None, None, None, 0
    start = time.perf_counter()
    rows = []
    with log_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["update", "loss", "validation_success", "fixed_success", "validation_path_length", "trajectories"])
        writer.writeheader()
        for update in range(1, updates+1):
            brain.train()
            loss = supervised_update(brain, optimizer, trajectories, rng, batch_size)
            if update % validate_every == 0 or update == updates:
                summary, fixed = validate(config, brain, seed=validation_seed)
                key = (int(fixed["success"]), summary["success_rate"],
                       -(summary["mean_sim_seconds_success"] or env.max_steps*env.dt))
                if best_key is None or key > best_key:
                    best_key, best_update = key, update
                    best_state = {name: value.detach().clone() for name, value in brain.state_dict().items()}
                    best_summary = {"random_start": summary, "fixed_start": fixed}
                row = {"update": update, "loss": loss, "validation_success": summary["success_rate"],
                       "fixed_success": int(fixed["success"]), "validation_path_length": summary["mean_path_length_all"],
                       "trajectories": len(trajectories)}
                rows.append(row)
                writer.writerow(row)
                handle.flush()
                print(f"guided {brain.kind}: update={update}/{updates} loss={loss:.5f} "
                      f"validation={summary['successes']}/{summary['episodes']} fixed={fixed['success']}", flush=True)
                if update < updates:
                    extra, _ = collect(env, 8, rng, brain=brain, teacher_probability=0.25 if update < updates/2 else 0.0)
                    trajectories.extend(extra)
                    # Keep expert examples and the most recent corrective trajectories.
                    trajectories = trajectories[:demonstrations]+trajectories[demonstrations:][-128:]
    brain.load_state_dict(best_state)
    brain.eval()
    training = {"algorithm": "recurrent_behavior_cloning_dagger", "seed": seed, "updates": updates,
                "selected_update": best_update, "episodes": len(trajectories), "demonstrations": demonstrations,
                "teacher": "training-only full-map BFS waypoints; no teacher/planner during evaluation or playback",
                "validation_seed": validation_seed, "validation": best_summary,
                "trainable_parameters": parameter_count(brain), "lr": lr, "batch_size": batch_size,
                "wall_seconds": time.perf_counter()-start, "version": __version__,
                "critic_trained": False, "stochastic_policy_trained": False,
                "scope": "fixed-map guided navigation; not evidence of unseen-maze generalization"}
    save(output, brain, env, training)
    print(f"Saved best validated model (update {best_update}): {output}", flush=True)
    return training
