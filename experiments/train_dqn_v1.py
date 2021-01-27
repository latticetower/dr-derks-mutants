"""Basic DQN training
"""
import collections
import random
from gym_derk.envs import DerkEnv
from gym_derk import ObservationKeys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.qnet import Qnet
from utils.replay_buffer import ReplayBuffer


NPZ_FILENAME = "weights/model_v2.npz"
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 10 #50000
batch_size    = 4 #32
REWARD_FUNCTION = {
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


def train(q, q_target, memory, optimizer):
    for i in range(10):
        state, action, r, s_prime, done_mask = memory.sample(batch_size)
        print(state)
        exit(1)
        q_out = q(state)
        q_a = q_out.gather(1, action)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main(env):
    # env = gym.make('CartPole-v1') 
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer(10)

    print_interval = 2 #20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(20):#10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        observations = env.reset()
        done = False
        q.eval()
        while not done:
            observations = torch.from_numpy(observations).float()
            actions = [ q.sample_action(obs, epsilon) for obs in observations]
            # print("shape", actions)
            # print(actions)
            next_observations, r, done, info = env.step(actions)
            # print(next_observations)
            # exit(1)
            # print("s_prime", s_prime)
            # print(r)
            # print(done)
            # print(info)
            done_mask = 0.0 if done else 1.0
            memory.put((observations, actions, r/100.0, next_observations, done_mask))
            observations = next_observations
            score += r
            if done:
                break
        if memory.size() > 5:  # 2000:
            q.train()
            train(q, q_target, memory, optimizer)
            exit(1)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {}, {}, n_buffer : {}, eps : {}%".format(
                n_epi, score, print_interval, memory.size(), epsilon))
            score = 0.0
    # env.close()

if __name__ == '__main__':
    env = DerkEnv(
        mode="train",
        turbo_mode=True,
        home_team=[
            {'primaryColor': '#3AA8C1'},
            {'primaryColor': '#BD559C', 'slots': ['Talons', None, None]},
            {'primaryColor': '#832A0D', 'rewardFunction': {'healTeammate1': 1}}
        ],
        away_team=[
            {'primaryColor': '#2D5DA1'},
            {'primaryColor': '#D05340', 'slots': ['Talons', None, None]},
            {'primaryColor': '#FBE870', 'rewardFunction': {'healTeammate1': 1}}
        ],
        session_args = {
            "reward_function": REWARD_FUNCTION
        }
    )
    main(env)
    env.close()