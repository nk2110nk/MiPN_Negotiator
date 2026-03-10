import csv

import sao
from negmas import UtilityFunction, Issue
from sao.my_sao import MySAOMechanism
from sao.my_negotiators import *

ISSUE_NAMES = [
    'Laptop',
    'ItexvsCypress',
    'IS_BT_Acquisition',
    'Grocery',
    'thompson',
    'Car',
    'EnergySmall_A'
]


def main():
    for issue in ISSUE_NAMES:
        results = [['bid', 'util1', 'util2']]
        domain, _ = Issue.from_genius('./domain/' + issue + '/domain.xml')
        util1, _ = UtilityFunction.from_genius('./domain/' + issue + '/utility1.xml')
        util2, _ = UtilityFunction.from_genius('./domain/' + issue + '/utility2.xml')

        session = MySAOMechanism(issues=domain, n_steps=80, avoid_ultimatum=False)
        agent1 = TimeBasedNegotiator()
        agent2 = TimeBasedNegotiator()
        session.add(agent1, ufun=util1)
        session.add(agent2, ufun=util2)

        # session.run()
        frontier, frontier_outcome = session.pareto_frontier(sort_by_welfare=True)
        # print(frontier, frontier_outcome)
        results += [[bid, *util] for bid, util in zip(frontier_outcome, frontier)]

        with open('./domain/' + issue + '/pareto.tsv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(results)

        session.reset()


if __name__ == '__main__':
    main()
