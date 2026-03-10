import dill
dill.extend(False)
import sao
from datetime import datetime
from gym import register
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
dill.extend(True)

ENV_NAME = 'IssueActionEnv-{}-{}-v0'
ISSUE_NAMES = [
    'Laptop',
    'ItexvsCypress',
    'IS_BT_Acquisition',
    'Grocery',
    'thompson',
    'Car',
    'EnergySmall_A'
]
AGENT_LIST = [
    'Boulware',
    'Linear',
    'Conceder',
    'TitForTat1',
    'TitForTat2',
    "AgentK",
    "HardHeaded",
    "Atlas3",
    "AgentGG",
]
SAVE_PATH = "./results/{}/".format(datetime.now().strftime('%Y%m%d-%H%M%S')[2:])


def register_neg_env(issue, agent):
    env_name = ENV_NAME.format(issue, agent)
    register(
        id=env_name,
        entry_point='envs.env:IssueActionEnv',
        kwargs={'domain': issue, 'opponent': agent, 'is_first': True},
    )
    return env_name


def run_rl(args):
    issue, agent = args
    env_name = register_neg_env(issue, agent)
    f_name = env_name.split('-', maxsplit=1)[1]
    env = make_vec_env(env_name, n_envs=4)

    ppo = PPO("MlpPolicy",
              env,
              policy_kwargs={"net_arch": [64, dict(vf=[64, 64], pi=[64, 64])]},
              verbose=1,
              device="cpu",
              tensorboard_log=SAVE_PATH)
    model = ppo.policy

    print(model)
    dummy_input = torch.from_numpy(np.array([model.observation_space.sample().astype(np.float32)])).clone()
    # writer = SummaryWriter()
    # writer.add_graph(model, dummy_input)
    # writer.close()

    # input_names = ["input"]
    # output_names = ["output"]
    # torch.onnx.export(model, dummy_input, "./test_model.onnx", input_names=input_names, output_names=output_names)
    env.close()
    del model


def main():
    run_rl((ISSUE_NAMES[0], AGENT_LIST[0]))


if __name__ == '__main__':
    main()
