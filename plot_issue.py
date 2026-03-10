import gc
import os
import csv
from itertools import product

import sao
from negmas import UtilityFunction, Issue
from sao.my_sao import MySAOMechanism
from sao.my_negotiators import *
from envs.rl_negotiator import TestRLNegotiator
from matplotlib import pyplot as plt

# ISSUE_NAMES = [
#     'Laptop',
#     'ItexvsCypress',
#     'IS_BT_Acquisition',
#     'Grocery',
#     'thompson',
#     'Car',
#     'EnergySmall_A'
# ]
ISSUE_NAMES = [
    'thompson_11',
    'thompson_12',
    'thompson_21',
    'thompson_13',
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

MODEL_NUM = 0
MODEL = ['Issue', 'VeNAS'][MODEL_NUM]
# LOAD_PATH = ["./results/211025-105809/", "./results/211019-140000/"][MODEL_NUM]
LOAD_PATH = ["./results/mipn_weight/", "./results/venas_weight/"][MODEL_NUM]


def a(x):
    return 'T' if x else 'F'


def run_session(path, save_path, opp_name, domain, util1, util2, det, noise):
    session = MySAOMechanism(issues=domain, n_steps=80, avoid_ultimatum=False)
    my_agent = TestRLNegotiator(domain, path, deterministic=det, mode=MODEL.lower())
    opponent = get_opponent(opp_name, add_noise=noise)

    # 先攻想定
    session.add(my_agent, ufun=util1)
    session.add(opponent, ufun=util2)

    result = session.run()

    # 結果を描画
    history = [h['current_offer'] for h in session.history
               if h['agreement'] is None and h['current_proposer'] is not None and 'RLAgent' in h['current_proposer']]


    count_dict = {issue.name: {value: 0 for value in issue.values} for issue in domain}
    for bid in history:
        for k, v in bid.items():
            count_dict[k][v] += 1
    # print(util1.issue_utilities['Price'].mapping.values())
    weights = {issue.name: [max(util1.issue_utilities[issue.name].mapping.values()),
                            max(util2.issue_utilities[issue.name].mapping.values())
                            ] for issue in domain}
    evals = {issue.name: {value: (util1.issue_utilities[issue.name].mapping[value] / weights[issue.name][0],
                                   util2.issue_utilities[issue.name].mapping[value] / weights[issue.name][1]
                                   ) for value in issue.values} for issue in domain}

    n_row, n_col = 1, 1
    n_issue = len(weights)
    while n_row * n_col < n_issue:
        n_row += 1
        if n_row * n_col >= n_issue:
            break
        n_col += 1

    # 図を用意
    fig = plt.figure(figsize=(4.8 * n_row, 4.8 * n_col / 0.96), constrained_layout=True)
    for i, (issue_name, values) in enumerate(count_dict.items(), start=1):
        ax = fig.add_subplot(n_col, n_row, i)
        for value, n in values.items():
            x, y = evals[issue_name][value]
            ax.scatter(x, y, color='skyblue', s=n * 50 if n != 0 else 1)
            ax.text(x, y, f'{n}')
        if result.agreement is not None:
            x, y = evals[issue_name][result.agreement[issue_name]]
            ax.scatter(x, y, color='magenta', marker='x')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(f'Our Agent: {weights[issue_name][0]:.4f}')
        ax.set_ylabel(f'{opp_name}: {weights[issue_name][1]:.4f}')
        ax.set_title(issue_name)
        ax.set_aspect('equal')
    fig.suptitle(f'{MODEL}-' + path.split('/')[-1].rsplit('-', maxsplit=1)[0], fontweight="bold")
    plt.savefig(save_path + path.split('/')[-1].rsplit('-', maxsplit=1)[0] + f'-d{a(det)}-n{a(noise)}.png')
    # plt.show()
    plt.clf()
    plt.close()

    session.reset()
    del my_agent, util1._ami, util2._ami, session, opponent
    gc.collect()

    return


def test_negotiator(issue, agent, det, noise, save_path):
    # results = [['my_util', 'opp_util', 'social', 'nash', 'agreement', 'step']]
    domain, _ = Issue.from_genius('./domain/' + issue + '/domain.xml')
    util1, _ = UtilityFunction.from_genius('./domain/' + issue + '/utility1.xml')
    util2, _ = UtilityFunction.from_genius('./domain/' + issue + '/utility2.xml')

    # results.append(run_session(f'{LOAD_PATH}{issue}-{agent}-v0.zip', save_path, agent, domain, util1, util2, det, noise))
    run_session(f'{LOAD_PATH}{issue}-{agent}-v0.zip', save_path, agent, domain, util1, util2, det, noise)

    # with open(f'{save_path}{issue}-{agent}-d{a(det)}-n{a(noise)}.tsv', 'w') as f:
    #     writer = csv.writer(f, delimiter='\t')
    #     writer.writerows(results)


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
    for det, noise in product([True, False], [False]):
        save_path = LOAD_PATH + f'issue/det={det}_noise={noise}/'
        os.makedirs(save_path, exist_ok=True)
        for issue in ISSUE_NAMES:
            for agent in AGENT_LIST:
                test_negotiator(issue, agent, det, noise, save_path)


if __name__ == '__main__':
    main()
