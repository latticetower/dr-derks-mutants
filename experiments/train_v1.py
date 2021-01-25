from gym_derk.envs import DerkEnv
from gym_derk import ObservationKeys
import numpy as np
import gym
import math
import os.path

from models.network_v1 import Network


SEED = 137
np.random.seed(SEED)

NPZ_FILENAME = "weights/model_v1.npz"
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


env = DerkEnv(
    mode="train",
    turbo_mode=True,
    home_team=[
        {'primaryColor': '#ff00ff'},
        {'primaryColor': '#00ff00', 'slots': ['Talons', None, None]},
        {'primaryColor': '#ff0000', 'rewardFunction': {'healTeammate1': 1}}
    ],
    away_team=[
        {'primaryColor': '#c0c0c0'},
        {'primaryColor': 'navy', 'slots': ['Talons', None, None]},
        {'primaryColor': 'red', 'rewardFunction': {'healTeammate1': 1}}
    ],
    session_args = {
        "reward_function": REWARD_FUNCTION
    }
   
)


if os.path.exists(NPZ_FILENAME):
    with np.load(NPZ_FILENAME) as data:
        weights = np.asarray(data['weights']).copy()
        biases = np.asarray(data['biases']).copy()
else:
    weights = None
    biases = None


networks = [Network(weights, biases) for i in range(env.n_agents)]

for e in range(20):
    observation_n = env.reset()
    while True:
        action_n = [networks[i].forward(observation_n[i]) for i in range(env.n_agents)]
        observation_n, reward_n, done_n, info = env.step(action_n)
        if all(done_n):
            print("Episode finished")
            break
    if env.mode == 'train':
        reward_n = env.total_reward
        print(reward_n)
        top_network_i = np.argmax(reward_n)
        top_network = networks[top_network_i].clone()
        for network in networks:
            network.copy_and_mutate(top_network)
        print(f'Round {e} top reward', reward_n[top_network_i])
        np.savez_compressed(
            NPZ_FILENAME,
            weights=top_network.weights,
            biases=top_network.biases
        )
env.close()