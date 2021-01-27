import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Qnet(nn.Module):
    def __init__(self, nobservations=64, output_size=13, emb_size=4):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(nobservations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        move = torch.tanh(x[0:1])  # MoveX
        rotate = torch.tanh(x[1:2])  # Rotate
        chase_focus = torch.sigmoid(x[2:3]) # # ChaseFocus
        casts = torch.sigmoid(x[3:6])
        focuses = torch.sigmoid(x[6:13])
        # print("focuses", focuses.shape)
        seq = [move, rotate, chase_focus, casts, focuses]
        return torch.cat(seq, dim=0)

    def sample_action(self, obs, epsilon):
        # obs: 6, 64
        out = self.forward(obs)
        print("out:", out.shape)
        coin = random.random()
        if coin < epsilon:
            move = random.uniform(-1, 1)
            rotate = random.uniform(-1, 1)
            chase_focus = random.random()
            cast = random.sample([0, 1, 2], 1)[0]
            focus = random.sample([0, 1, 2, 3, 4, 5, 6, 7], 1)[0]
            return [move, rotate, chase_focus, cast, focus]
        else:
            move = out[0].detach().cpu().item()
            rotate = out[1].detach().cpu().item()
            chase_focus = out[2].detach().cpu().item()
            casts = out[3:6].detach().cpu().numpy()
            focuses = out[6:13].detach().cpu().numpy()
            return [move, rotate, chase_focus, casts.argmax(), focuses.argmax()]
  