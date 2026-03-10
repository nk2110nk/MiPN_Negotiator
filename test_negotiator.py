import gc
import os
import csv
from multiprocessing import Pool
from itertools import product

import sao
from negmas import UtilityFunction, Issue
from sao.my_sao import MySAOMechanism
from sao.my_negotiators import *
from envs.rl_negotiator import TestRLNegotiator
from matplotlib import pyplot as plt

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
# LOAD_PATH = "./results/211025-105809/"  # issue
LOAD_PATH = "./results/211019-140000/"  # venas
PLOT = True


def a(x):
    return 'T' if x else 'F'


def run_session(path, save_path,  opponent, domain, util1, util2, det, noise):
    session = MySAOMechanism(issues=domain, n_steps=80, avoid_ultimatum=False)
    my_agent = TestRLNegotiator(domain, path, deterministic=det, mode='venas')
    opponent = get_opponent(opponent, add_noise=noise)

    # 先攻想定
    session.add(my_agent, ufun=util1)
    session.add(opponent, ufun=util2)

    result = session.run()

    # 結果を描画
    if PLOT:
        my_agent.name = "Our Agent"
        session.plot(path=save_path + path.split('/')[-1].rsplit('-', maxsplit=1)[0] + f'-d{a(det)}-n{a(noise)}.png')
        # plt.show()
        plt.clf()
        plt.close()

    session.reset()
    del my_agent, util1._ami, util2._ami, session, opponent
    gc.collect()

    if result['agreement'] is not None:
        my_util, opp_util = util1(result['agreement']), util2(result['agreement'])
    else:
        my_util, opp_util = 0, 0

    return [
        my_util,
        opp_util,
        my_util + opp_util,
        my_util * opp_util,
        result['agreement'],
        result['step']
    ]
    # return {
    #     'my_util': my_util,
    #     'opp_util': opp_util,
    #     'social': my_util + opp_util,
    #     'nash': my_util * opp_util,
    #     'agreement': result['agreement'],
    #     'step': result['step']
    # }


def test_negotiator(config):
    issue, agent, det, noise, save_path = config
    results = [['my_util', 'opp_util', 'social', 'nash', 'agreement', 'step']]
    domain, _ = Issue.from_genius('./domain/' + issue + '/domain.xml')
    util1, _ = UtilityFunction.from_genius('./domain/' + issue + '/utility1.xml')
    util2, _ = UtilityFunction.from_genius('./domain/' + issue + '/utility2.xml')
    for _ in range(1 if PLOT else 100):
        results.append(run_session(f'{LOAD_PATH}{issue}-{agent}-v0.zip', save_path, agent, domain, util1, util2, det, noise))

    if not PLOT:
        with open(f'{save_path}{issue}-{agent}-d{a(det)}-n{a(noise)}.tsv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(results)


def get_opponent(opponent, add_noise=False):
    if opponent == 'Boulware':
        opponent = TimeBasedNegotiator(name='Boulware', aspiration_type=10.0, add_noise=add_noise)
    elif opponent == 'Linear':
        opponent = TimeBasedNegotiator(name='Linear', aspiration_type=1.0, add_noise=add_noise)
    elif opponent == 'Conceder':
        opponent = TimeBasedNegotiator(name='Conceder', aspiration_type=0.2, add_noise=add_noise)
    elif opponent == 'TitForTat1':
        opponent = AverageTitForTatNegotiator(name='TitForTat1', gamma=1, add_noise=add_noise)
    elif opponent == 'TitForTat2':
        opponent = AverageTitForTatNegotiator(name='TitForTat2', gamma=2, add_noise=add_noise)
    elif opponent == 'AgentK':
        opponent = AgentK(add_noise=add_noise)
    elif opponent == 'HardHeaded':
        opponent = HardHeaded(add_noise=add_noise)
    elif opponent == 'CUHKAgent':
        opponent = CUHKAgent(add_noise=add_noise)
    elif opponent == 'Atlas3':
        opponent = Atlas3(add_noise=add_noise)
    elif opponent == 'AgentGG':
        opponent = AgentGG(add_noise=add_noise)
    else:
        opponent = TimeBasedNegotiator(name='Linear', aspiration_type=1.0, add_noise=add_noise)
    return opponent


def main():
    # try:
    #     os.makedirs(LOAD_PATH + 'img' if PLOT else 'csv')
    # except FileExistsError:
    #     pass

    p = Pool(len(AGENT_LIST))
    for det, noise in product([True, False], [False]):
        save_path = LOAD_PATH + ('img' if PLOT else 'csv') + f'/det={det}_noise={noise}/'
        try:
            os.makedirs(save_path)
        except FileExistsError:
            exit(0)
        for issue in ISSUE_NAMES:
            p.map(test_negotiator, [(issue, agent, det, noise, save_path) for agent in AGENT_LIST])


if __name__ == '__main__':
    main()
