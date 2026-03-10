import csv
from ast import literal_eval

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
LOAD_PATH = "../results/det=False_noise=True/{}-{}-dF-nT.tsv"


def l1_dist(x, y, pareto):
    tmp = []
    for p in pareto:
        tmp.append(abs(x - p[0]) + abs(y - p[1]))
    return min(tmp)


def l2_dist(x, y, pareto):
    tmp = []
    for p in pareto:
        tmp.append(((x - p[0]) ** 2 + (y - p[1]) ** 2) ** 0.5)
    return min(tmp)


def ham_dist(bid, pareto):
    if bid is None:
        return len(pareto[0])
    tmp = []
    for p in pareto:
        tmp.append(sum([0 if p[key] == value else 1 for key, value in bid.items()]))
    return min(tmp)


def main():
    for issue in ISSUE_NAMES:
        with open('../domain/' + issue + '/pareto.tsv', 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            pareto = [row for row in reader]
        pareto_util = [[float(util1), float(util2)] for _, util1, util2 in pareto[1:]]
        pareto_outcome = [literal_eval(bid) for bid, *_ in pareto[1:]]
        nash = [u1 * u2 for u1, u2 in pareto_util]
        nash_util = [pareto_util[nash.index(max(nash))]]
        nash_outcome = [pareto_outcome[nash.index(max(nash))]]
        for agent in AGENT_LIST:
            with open(LOAD_PATH.format(issue, agent), 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                lines = [[r if r != '' else 'None' for r in row] for row in reader]
            lines[1:] = [[float(mu), float(ou), so, na, literal_eval(ag), st] for mu, ou, so, na, ag, st in lines[1:]]

            lines[0].append('pareto_l1')
            lines[0].append('pareto_l2')
            lines[0].append('pareto_hamming')
            for lst in lines[1:]:
                lst.append(l1_dist(lst[0], lst[1], pareto_util))
                lst.append(l2_dist(lst[0], lst[1], pareto_util))
                lst.append(ham_dist(lst[4], pareto_outcome))

            lines[0].append('nash_l1')
            lines[0].append('nash_l2')
            lines[0].append('nash_hamming')
            for lst in lines[1:]:
                lst.append(l1_dist(lst[0], lst[1], nash_util))
                lst.append(l2_dist(lst[0], lst[1], nash_util))
                lst.append(ham_dist(lst[4], nash_outcome))

            with open(LOAD_PATH.format(issue, agent)[:-10] + '.tsv', 'w') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(lines)


if __name__ == '__main__':
    main()
