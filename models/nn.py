import json

import torch
from torch import nn
from torch.nn import functional as F


class QNet(nn.Module):
    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()

        self.reward_function = {
            "damageEnemyStatue": 4,
            "damageEnemyUnit": 2,
            "killEnemyStatue": 4,
            "killEnemyUnit": 2,
            "healFriendlyStatue": 1,
            "healTeammate1": 2,
            "healTeammate2": 2,
            "timeSpentHomeBase": 0,
            "timeSpentHomeTerritory": 0,
            "timeSpentAwayTerritory": 0,
            "timeSpentAwayBase": 0,
            "damageTaken": -1,
            "friendlyFire": -1,
            "healEnemy": -1,
            "fallDamageTaken": -10,
            "statueDamageTaken": 0,
            "manualBonus": 0,
            "victory": 100,
            "loss": -100,
            "tie": 0,
            "teamSpirit": 0.5,
            "timeScaling": 0.8,
        }

        self.dropout_rate = dropout_rate

        self.fc1 = nn.Linear(69, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = nn.Dropout(self.dropout_rate)(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # return torch.tanh(x)
        return x

    def save_parameters(self, weights_path: str, reward_function_path: str):
        torch.save(self.state_dict(), weights_path)

        with open(reward_function_path, 'w') as f:
            json.dump(self.reward_function, f)

    def load_parameters(self, weights_path: str, reward_function_path: str):
        self.load_state_dict(torch.load(weights_path))

        with open(reward_function_path, 'r') as f:
            self.reward_function = json.load(f)
