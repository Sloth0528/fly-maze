"""Identical continuous action heads for GRU and connectome-constrained rate RNN."""

from pathlib import Path
import torch
from torch import nn
from torch.distributions import Normal

from . import __version__
from .connectome import load_graph


class Policy(nn.Module):
    def initial_memory(self):
        return self.log_std.new_zeros(1, self.hidden_size)

    def act(self, observation, memory, greedy=False):
        mean, value, memory = self(observation, memory)
        distribution = Normal(mean, self.log_std.clamp(-3, 0.5).exp())
        z = mean if greedy else distribution.sample()
        bounded = torch.tanh(z)
        log_prob = (distribution.log_prob(z)-torch.log(1-bounded.square()+1e-6)).sum()
        # Exploration regularizer is base-Gaussian entropy, not squashed entropy.
        entropy = distribution.entropy().sum()
        action = [(bounded[0, 0].item()+1)/2, bounded[0, 1].item()]
        return action, log_prob, value.squeeze(), entropy, memory

    def heads(self, memory):
        return self.actor(memory), self.critic(memory).squeeze(), memory

    def observation_tensor(self, observation):
        value = torch.as_tensor(observation, dtype=torch.float32, device=self.log_std.device)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[1] != self.observation_size:
            raise ValueError("observation size does not match checkpoint")
        return value

    def init_heads(self):
        self.actor = nn.Linear(self.hidden_size, 2)
        self.critic = nn.Linear(self.hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)


class GRUPolicy(Policy):
    kind = "gru"

    def __init__(self, observation_size=31, hidden_size=64):
        super().__init__()
        self.observation_size, self.hidden_size = observation_size, hidden_size
        self.encoder = nn.Sequential(nn.Linear(observation_size, hidden_size), nn.Tanh())
        self.memory = nn.GRUCell(hidden_size, hidden_size)
        self.init_heads()

    def forward(self, observation, memory):
        memory = self.memory(self.encoder(self.observation_tensor(observation)), memory)
        return self.heads(memory)

    def config(self):
        return {"kind": self.kind, "observation_size": self.observation_size, "hidden_size": self.hidden_size}


class ConnectomePolicy(Policy):
    kind = "connectome"

    def __init__(self, counts, root_ids, metadata, observation_size=31, leak=0.3, gain=0.8):
        super().__init__()
        counts = torch.as_tensor(counts, dtype=torch.float32)
        if (counts.ndim != 2 or counts.shape[0] != counts.shape[1]
                or counts.shape[0] != len(root_ids) or not torch.isfinite(counts).all()
                or (counts < 0).any() or counts.sum() <= 0):
            raise ValueError("invalid connectome counts")
        if not 0 < leak <= 1 or not 0 <= gain < 1:
            raise ValueError("invalid rate dynamics")
        self.observation_size, self.hidden_size = observation_size, counts.shape[0]
        self.root_ids, self.metadata = list(root_ids), dict(metadata)
        self.leak, self.gain = leak, gain
        self.register_buffer("counts", counts)
        # Row normalization limits recurrent gain, preserving direction and absent edges.
        self.register_buffer("adjacency", counts/counts.sum(dim=1, keepdim=True).clamp_min(1))
        self.encoder = nn.Linear(observation_size, self.hidden_size)
        self.init_heads()

    def forward(self, observation, memory):
        drive = self.encoder(self.observation_tensor(observation))
        memory = (1-self.leak)*memory + self.leak*torch.tanh(drive+self.gain*(memory @ self.adjacency.T))
        return self.heads(memory)

    def config(self):
        return {"kind": self.kind, "observation_size": self.observation_size, "hidden_size": self.hidden_size,
                "leak": self.leak, "gain": self.gain, "root_ids": self.root_ids, "provenance": self.metadata}


def build(kind, *, hidden_size=64, connectome="data/flywire_cx_512.npz"):
    if kind == "gru":
        return GRUPolicy(hidden_size=hidden_size)
    if kind == "connectome":
        counts, ids, metadata = load_graph(connectome)
        return ConnectomePolicy(counts, ids, metadata)
    raise ValueError("unknown brain kind")


def save(path, brain, env, training):
    with Path(path).open("xb") as handle:
        torch.save({"format_version": 2, "project_version": __version__,
                    "brain": brain.config(), "environment": env.config(),
                    "training": training, "state_dict": brain.state_dict()}, handle)


def load(path):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2:
        raise ValueError("use v1 scripts for format_version=1 checkpoints")
    config = checkpoint["brain"]
    if config["kind"] == "gru":
        brain = GRUPolicy(config["observation_size"], config["hidden_size"])
    elif config["kind"] == "connectome":
        brain = ConnectomePolicy(checkpoint["state_dict"]["counts"], config["root_ids"],
                                 config["provenance"], config["observation_size"], config["leak"], config["gain"])
        if not torch.equal(brain.adjacency, checkpoint["state_dict"]["adjacency"]):
            raise ValueError("checkpoint adjacency does not match its measured counts")
    else:
        raise ValueError("unknown checkpoint brain kind")
    brain.load_state_dict(checkpoint["state_dict"])
    brain.eval()
    return brain, checkpoint
