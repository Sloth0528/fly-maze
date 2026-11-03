"""Shared recurrent Actor-Critic training and paired, no-update evaluation."""

import csv
import json
import math
from pathlib import Path
import time

import torch
from torch import nn

from . import __version__
from .environment import from_config
from .models import save, load


def parameter_count(brain):
    return sum(p.numel() for p in brain.parameters() if p.requires_grad)


def update(brain, optimizer, rewards, log_probs, values, entropies, bootstrap, gamma):
    targets = []
    total = bootstrap
    for reward in reversed(rewards):
        total = reward+gamma*total
        targets.append(total)
    targets = torch.tensor(list(reversed(targets)), dtype=torch.float32)
    advantage = targets-torch.stack(values)
    loss = (-(torch.stack(log_probs)*advantage.detach()).mean()
            + 0.5*advantage.square().mean()-0.01*torch.stack(entropies).mean())
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite training loss")
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(brain.parameters(), 0.5, error_if_nonfinite=True)
    optimizer.step()
    return loss.item()


def train(env, brain, output, *, episodes=1000, seed=7, lr=0.0003,
          gamma=0.99, rollout=64, random_start=False):
    output = Path(output)
    log_path = output.with_suffix(".csv")
    if output.suffix != ".pt" or output.exists() or log_path.exists():
        raise ValueError("choose an unused .pt output (and unused companion .csv)")
    if episodes < 1 or rollout < 1 or not math.isfinite(lr) or lr <= 0 or not 0 <= gamma <= 1:
        raise ValueError("invalid training settings")
    output.parent.mkdir(parents=True, exist_ok=True)
    brain.train()
    optimizer = torch.optim.Adam(brain.parameters(), lr=lr)
    recent, started = [], time.perf_counter()
    fields = ["episode", "reward", "steps", "success", "collisions", "elapsed", "path_length", "loss"]
    with log_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for episode in range(1, episodes+1):
            observation, memory = env.reset(random_start), brain.initial_memory()
            done, total_reward, losses = False, 0.0, []
            while not done:
                rewards, log_probs, values, entropies = [], [], [], []
                for _ in range(rollout):
                    action, log_prob, value, entropy, memory = brain.act(observation, memory)
                    observation, reward, terminated, truncated, _ = env.step(action)
                    rewards.append(reward)
                    log_probs.append(log_prob)
                    values.append(value)
                    entropies.append(entropy)
                    total_reward += reward
                    done = terminated or truncated
                    if done:
                        break
                with torch.no_grad():
                    bootstrap = 0.0 if env.terminated else brain(observation, memory)[1].item()
                losses.append(update(brain, optimizer, rewards, log_probs, values, entropies, bootstrap, gamma))
                memory = memory.detach()  # truncated BPTT; same schedule for both brains
            row = {"episode": episode, "reward": total_reward, "steps": env.steps,
                   "success": int(env.terminated), "collisions": env.collisions,
                   "elapsed": env.elapsed, "path_length": env.path_length,
                   "loss": sum(losses)/len(losses)}
            writer.writerow(row)
            handle.flush()
            recent = (recent+[row["success"]])[-100:]
            if episode == 1 or episode % 100 == 0 or episode == episodes:
                print(f"{brain.kind}: episode={episode}/{episodes} train_success={sum(recent)/len(recent):.1%} "
                      f"reward={total_reward:.3f} loss={row['loss']:.4f}", flush=True)
    training = {"seed": seed, "episodes": episodes, "lr": lr, "gamma": gamma, "rollout": rollout,
                "random_start": random_start, "algorithm": "recurrent_actor_critic_tbptt",
                "trainable_parameters": parameter_count(brain), "wall_seconds": time.perf_counter()-started,
                "entropy": "base Gaussian", "version": __version__}
    save(output, brain, env, training)
    print(f"Saved {output} ({training['trainable_parameters']} trainable parameters)", flush=True)
    return training


def episode(env, brain, *, random_start=False, greedy=True):
    observation, memory = env.reset(random_start), brain.initial_memory()
    reward_sum = 0.0
    with torch.inference_mode():
        while not (env.terminated or env.truncated):
            action, _, _, _, memory = brain.act(observation, memory, greedy)
            observation, reward, _, _, _ = env.step(action)
            reward_sum += reward
    return {"success": env.terminated, "steps": env.steps, "sim_seconds": env.elapsed,
            "path_length": env.path_length, "collisions": env.collisions, "reward": reward_sum,
            "initial_pose": env.initial_pose, "map_seed": env.map_seed}


def summarize(results):
    n = len(results)
    successes = [r for r in results if r["success"]]
    rate = len(successes)/n
    # Wilson interval describes episode-level binomial uncertainty, not training-seed uncertainty.
    z = 1.96
    denominator = 1+z*z/n
    middle = (rate+z*z/(2*n))/denominator
    half = z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/denominator
    return {"episodes": n, "successes": len(successes), "success_rate": rate,
            "success_wilson95": [max(0.0, middle-half), min(1.0, middle+half)],
            "mean_steps_all": sum(r["steps"] for r in results)/n,
            "mean_sim_seconds_all": sum(r["sim_seconds"] for r in results)/n,
            "mean_path_length_all": sum(r["path_length"] for r in results)/n,
            "mean_collisions_all": sum(r["collisions"] for r in results)/n,
            "mean_sim_seconds_success": (sum(r["sim_seconds"] for r in successes)/len(successes)) if successes else None,
            "mean_path_length_success": (sum(r["path_length"] for r in successes)/len(successes)) if successes else None}


def evaluate_models(paths, *, episodes=20, seed=10007, random_start=False, sample=False,
                    path=None, difficulty=None, vary_maze=False, max_steps=None, output=None, trace=None):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    for target in (output, trace):
        if target is not None and Path(target).exists():
            raise FileExistsError(target)
    loaded = [(str(model), *load(model)) for model in paths]
    # A comparison rejects different body/sensor/reward settings, rather than silently changing them.
    reference = loaded[0][2]["environment"]
    for _, _, checkpoint in loaded[1:]:
        for key, value in reference.items():
            if key in ("layout", "difficulty", "vary_maze") and (path is not None or difficulty is not None):
                continue
            if key == "max_steps" and max_steps is not None:
                continue
            if checkpoint["environment"].get(key) != value:
                raise ValueError(f"comparison environment mismatch: {key}; provide a shared test map for differing layouts")
    report = {"project_version": __version__, "test_seed": seed, "random_start": random_start,
              "action_mode": "sample" if sample else "deterministic",
              "vary_maze": vary_maze, "models": [],
              "interpretation": "Short runs validate execution only. No claim of stable navigation or biological fidelity. "
                                "Parameter counts are reported, not matched. Wilson intervals assume independent episodes; "
                                "fixed-map/fixed-start deterministic repetitions are not independent trials."}
    for model, brain, checkpoint in loaded:
        env = from_config(reference, seed=seed, path=path, difficulty=difficulty,
                          vary_maze=vary_maze, max_steps=max_steps)
        started = time.perf_counter()
        results = []
        for i in range(episodes):
            torch.manual_seed(seed+i)
            results.append(episode(env, brain, random_start=random_start, greedy=not sample))
        result = {"model": model, "kind": brain.kind, "trainable_parameters": parameter_count(brain),
                  "training": checkpoint["training"], "environment": env.config(),
                  "summary": summarize(results), "wall_seconds": time.perf_counter()-started,
                  "episodes": results}
        report["models"].append(result)
        summary = result["summary"]
        print(f"{model}: success={summary['successes']}/{episodes} ({summary['success_rate']:.1%}), "
              f"time(all)={summary['mean_sim_seconds_all']:.2f}s, "
              f"collisions={summary['mean_collisions_all']:.2f}, params={parameter_count(brain)}", flush=True)
        if trace is not None and model == loaded[0][0]:
            from .viewer import export_svg
            export_svg(env, trace, title=f"{brain.kind}: {'SUCCESS' if env.terminated else 'TIME LIMIT'}")
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return report
