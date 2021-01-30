"""
Run a derk game between bots
"""

import asyncio
import importlib
from argparse import ArgumentParser
from models.nn import QNet

from gym_derk import DerkSession, DerkAgentServer, DerkAppInstance


async def run_player(env: DerkSession, DerkPlayerClass):
    """
    Runs a DerkPlayer
    """

    estimator = QNet()

    player = DerkPlayerClass(env.n_agents, estimator, epsilon=1.0)

    obs = await env.reset()
    player.signal_env_reset(obs)
    ordi = await env.step()

    while not env.done:
        actions = player.take_action(ordi)
        ordi = await env.step(actions)


async def main(p1, p2, n, turbo):
    """
    Runs the game in n arenas between p1 and p2
    """

    agent_p1 = DerkAgentServer(run_player, args={"DerkPlayerClass": p1}, port=8788)
    agent_p2 = DerkAgentServer(run_player, args={"DerkPlayerClass": p2}, port=8789)

    await agent_p1.start()
    await agent_p2.start()

    app = DerkAppInstance()
    await app.start()

    await app.run_session(
        n_arenas=n,
        turbo_mode=turbo,
        agent_hosts=[
            {"uri": agent_p1.uri, "regions": [{"sides": "home"}]},
            {"uri": agent_p2.uri, "regions": [{"sides": "away"}]},
        ],
        reward_function={
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
        },
    )
    await app.print_team_stats()


def parse_arguments():
    p = ArgumentParser()

    p.add_argument(
        "-p1",
        help="Path to player1 module relative to agent. Defaults to `bot`",
        type=str,
        default="bot",
    )
    p.add_argument(
        "-p2",
        help="Path to player2 module relative to agent. Defaults to `bot`",
        type=str,
        default="bot",
    )
    p.add_argument(
        "-n",
        help="Number of arenas to run, Defaults to 2",
        type=int,
        default=2,
    )
    p.add_argument(
        "--fast",
        help="To enable turbo mode or not. Defaults to False",
        action="store_true",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    player1 = importlib.import_module(f"agent.{args.p1}").DerkAgent
    player2 = importlib.import_module(f"agent.{args.p2}").DerkAgent

    asyncio.get_event_loop().run_until_complete(
        main(player1, player2, args.n, args.fast)
    )
