"""簡化的大腦：感覺層 → GRU 記憶 → 動作與價值輸出。"""

from pathlib import Path
import torch
from torch import nn
from torch.distributions import Categorical


class Brain(nn.Module):
    def __init__(self, observation_size=83, hidden_size=64):
        super().__init__()
        self.observation_size = observation_size
        self.hidden_size = hidden_size
        self.sense = nn.Sequential(nn.Linear(observation_size, hidden_size), nn.Tanh())
        self.memory = nn.GRUCell(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, 3)
        self.critic = nn.Linear(hidden_size, 1)

    def initial_memory(self):
        # 每回合從空記憶開始，不能把上一回合路徑帶進下一回合。
        return self.actor.weight.new_zeros(1, self.hidden_size)

    def forward(self, observation, memory):
        observation = torch.as_tensor(observation, dtype=torch.float32,
                                      device=self.actor.weight.device).reshape(1, -1)
        memory = self.memory(self.sense(observation), memory)
        return self.actor(memory), self.critic(memory).squeeze(), memory

    def act(self, observation, memory, greedy=False):
        logits, value, memory = self(observation, memory)
        distribution = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if greedy else distribution.sample()
        return (action.item(), distribution.log_prob(action).squeeze(), value,
                distribution.entropy().squeeze(), memory,
                distribution.probs.detach().flatten().tolist())


def load_brain(path):
    # 僅讀取 tensor 與基本資料型別，避免反序列化任意 Python 物件。
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("不支援此模型格式")
    config = checkpoint["config"]
    brain = Brain(config["observation_size"], config["hidden_size"])
    brain.load_state_dict(checkpoint["state_dict"])
    brain.eval()
    return brain, checkpoint
