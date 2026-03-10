from negmas import UtilityFunction, Issue
from negmas.negotiators import Negotiator
from sao.my_sao import MySAOMechanism
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

ISSUE_NAMES = [
    'Laptop',
    'ItexvsCypress',
    'IS_BT_Acquisition',
    'Grocery',
    'thompson',
    'Car',
    'EnergySmall_A'
]


def plot_util(issue, geniusize=True):
    issues, _ = Issue.from_genius('../domain/' + issue + '/domain.xml')
    util1, _ = UtilityFunction.from_genius('../domain/' + issue + '/utility1.xml', geniusize_utility=geniusize)
    util2, _ = UtilityFunction.from_genius('../domain/' + issue + '/utility2.xml', geniusize_utility=geniusize)
    session = MySAOMechanism(issues=issues)
    agent1 = Negotiator(name=issue+'_1')
    agent2 = Negotiator(name=issue+'_2')
    session.add(agent1, ufun=util1)
    session.add(agent2, ufun=util2)

    ufuns = session._get_ufuns()
    outcomes = session.outcomes
    utils = [tuple(f(o) for f in ufuns) for o in outcomes]

    frontier, frontier_outcome = session.pareto_frontier(sort_by_welfare=True)
    frontier_indices = [
        i for i, _ in enumerate(frontier)
        if _[0] is not None and _[0] > float("-inf") and _[1] is not None and _[1] > float("-inf")
    ]
    frontier = [frontier[i] for i in frontier_indices]
    max_welfare = frontier[0]
    frontier = sorted(frontier, key=lambda x: x[0])

    fig_util = plt.figure(figsize=(5, 5), dpi=300)
    plt.scatter([_[0] for _ in utils], [_[1] for _ in utils], label="outcomes", color="#50D1DE", marker=".", s=5)
    f1, f2 = [_[0] for _ in frontier], [_[1] for _ in frontier]
    plt.plot(f1, f2, linewidth=1.0, label="frontier", color="#F3AD45", marker="o", markersize=5, zorder=1)
    # plt.scatter([max_welfare[0]], [max_welfare[1]], color="#E86998", label=f"Max. Welfare", zorder=2)
    for ax in fig_util.get_axes():
        ax.xaxis.set_minor_locator(tick.MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(tick.MultipleLocator(0.05))
        ax.set_xlabel(agent1.name)
        ax.set_ylabel(agent2.name)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(color='gray', which='both', alpha=0.1, linestyle='--')
    # pp = PdfPages(issue + '.pdf')
    # pp.savefig(bbox_inches="tight", pad_inches=0.01)
    # pp.close()
    plt.savefig(issue + '.png', bbox_inches="tight", pad_inches=0.01)
    plt.show()


def main():
    for i in ISSUE_NAMES:
        plot_util(i, True)


if __name__ == '__main__':
    main()
