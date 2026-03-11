import dill
dill.extend(False)
import gym
import sao
import sys
import os
import argparse
from multiprocessing import Pool
from datetime import datetime
from gym import register

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
dill.extend(True)

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
ENV_LIST = [
    ('IssueActionEnv-{}-{}-v0', 'envs.env:IssueActionEnv'),
    ('AOPEnv-{}-{}-v0', 'envs.env:AOPEnv'),
]
SAVE_PATH = "./results/{}/".format(datetime.now().strftime('%Y%m%d-%H%M%S')[2:])


def register_neg_env(issue, agent, env):
    env_name = env[0].format(issue, agent)
    register(
        id=env_name,
        entry_point=env[1],
        kwargs={'domain': issue, 'opponent': agent, 'is_first': True},
    )
    return env_name


def run_rl(args):
    issue, agent, e_tuple, save_path = args
    env_name = register_neg_env(issue, agent, e_tuple)
    f_name = env_name.split('-', maxsplit=1)[1]
    env = make_vec_env(env_name, n_envs=4)

    model = PPO("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log=save_path)
    model.learn(total_timesteps=500000, tb_log_name=f_name)
    model.save(save_path + f_name)

    # Use a separate environement for evaluation
    eval_env = gym.make(env_name)
    eval_env.test = True
    # Random Agent, before training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    with open(save_path + "result.csv", "a") as f:
        f.write("{},{},{},{}\n".format(*env_name.split('-')[1:3], mean_reward, std_reward))

    env.close()
    eval_env.close()
    del model


def main_issue(agents, issues):
    save_path = SAVE_PATH + 'MiPN/'
    os.makedirs(save_path)
    with open(save_path + "result.csv", "w") as f:
        f.write("domain,opponent,mean,std\n")

    p = Pool(len(agents))
    for issue in issues:
        p.map(run_rl, [(issue, agent, ENV_LIST[0], save_path) for agent in agents])


def main_aop(agents, issues):
    save_path = SAVE_PATH + 'VeNAS/'
    os.makedirs(save_path)
    with open(save_path + "result.csv", "w") as f:
        f.write("domain,opponent,mean,std\n")

    p = Pool(len(agents))
    for issue in issues:
        p.map(run_rl, [(issue, agent, ENV_LIST[1], save_path) for agent in agents])


def main():
    # 変更箇所
    # IssueとAgentを指定して実行 -> --agents, --issue
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', '-a', required=True, nargs='*', type=str)
    parser.add_argument('--issue', '-i', required=True, nargs='*', type=str)
    args = parser.parse_args()
    #print(args)
    
    main_issue(args.agents, args.issue) # MiPN
    main_aop(args.agents, args.issue) # VeNAS


if __name__ == '__main__':
    main()

