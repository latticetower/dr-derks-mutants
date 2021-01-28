import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Qnet(nn.Module):
    def __init__(self, n_observations=64, n_actions=13, emb_size=4):
        super(Qnet, self).__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.net = self.create_mlp(n_observations + n_actions, [128, 128], 1)

    def forward(self, observations, actions):
        # print(x.shape)
        # x: (?, 64)
        # actions: (?, k, 13), 1 for each row
        # collected = torch.einsum('bk,bij->bjki', observations, actions)  # ?, 13, 64, k
        assert len(observations.shape) == 2
        b, k, d = actions.shape
        observations = observations.unsqueeze(1).expand(b, k, -1)
        collected = torch.cat([observations, actions], dim=-1)
        #collected = torch.einsum("bi,bkj->bkij", observations, actions)
        #assert len(collected.shape) == 4
        b, k, oa = collected.shape
        collected = collected.reshape(b*k, -1)
        rewards = self.net(collected) # after: (?, 1) - policies
        # move = torch.tanh(x.index_select(-1, torch.tensor([0])))  # MoveX
        # rotate = torch.tanh(x.index_select(-1, torch.tensor([1])))  # Rotate
        # chase_focus = torch.sigmoid(x.index_select(-1, torch.tensor([2]))) # # ChaseFocus
        # casts = torch.sigmoid(x.index_select(-1, torch.tensor([3, 4, 5])))
        # focuses = torch.sigmoid(x.index_select(-1, torch.tensor([6, 7, 8, 9, 10, 11, 12])))
        # print("focuses", focuses.shape)
        #seq = [move, rotate, chase_focus, casts, focuses]
        #return torch.cat(seq, dim=-1)
        return rewards.reshape(-1, k) # (?, k)

    @staticmethod
    def random_actions(batch_size=6, k=1, for_env=False):
        # deprecated!
        # obs: 6, 64
        move = np.random.uniform(-1, 1, (batch_size * k, 1)) #np.ones((batch_size * k, 1))
        rotate = np.random.uniform(-1, 1, (batch_size * k, 1))
        chase_focus = np.random.random(((batch_size * k, 1)))
        cast = np.random.randint(0, 3, (batch_size*k, ))
        focus = np.random.randint(0, 7, (batch_size*k, ))
        if for_env:
            cast = np.expand_dims(cast, -1)
            focus = np.expand_dims(focus, -1)
        else:
            chase_focus = chase_focus/2.0 - 0.5
            cast = torch.eye(3)[cast]
            focus = torch.eye(7)[focus]
        result = np.concatenate([move, rotate, chase_focus, cast, focus], axis=-1)
        return result.reshape(batch_size, k, -1)

    def sample_actions(self, obs, epsilon, k=1):
        # deprecated!
        # obs: 6, 64
        assert len(obs.shape) == 2
        coin = random.random()
        if coin < epsilon:
            size = obs.shape[0]
            move = np.random.uniform(-1, 1, (size, 1, k))
            rotate = np.random.uniform(-1, 1, (size, 1, k))
            chase_focus = np.random.random(((size, 1, k)))
            cast = np.random.randint(0, 3, (size, 1, k))
            focus = np.random.randint(0, 7, (size, 1, k))
            result = np.concatenate([move, rotate, chase_focus, cast, focus], axis=1)
            # return result
        else:
            out = self.forward(obs)
            out = out.detach().cpu()
            move_rotate_cf = out.index_select(-1, torch.tensor([0, 1, 2])).numpy()
            casts = out.index_select(
                -1, torch.tensor([3, 4, 5])
            ).argmax(-1, keepdim=True).numpy()
            focuses = out.index_select(
                -1, torch.tensor([6, 7, 8, 9, 10, 11, 12])
            ).argmax(-1, keepdim=True).numpy()
            # print(move_rotate_cf.shape, casts.shape, focuses.shape)
            result = np.concatenate([
                move_rotate_cf,
                casts,
                focuses
            ], axis=-1)
        return result

    def create_mlp(self, input_size, hidden, output_size):
        input_dim = input_size
        layers = []
        for h in hidden:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, output_size))
        return nn.Sequential(*layers)
  