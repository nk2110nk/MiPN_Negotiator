import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from itertools import product


INS = [
    'Laptop',
    'ItexvsCypress',
    'IS_BT_Acquisition',
    'Grocery',
    'thompson'
]


def main():
    df = pd.read_csv('AgentK.csv')
    for i2, i3 in product([INS[1], INS[2]], [INS[3], INS[4]]):
        plt.boxplot(df.loc[:, [INS[0], i2, i3]], labels=["$10^1$", "$10^2$", "$10^3$"], whis=(0, 100))
        plt.xlabel('Domain Size')
        plt.ylabel('Individual Utility')
        # plt.title(f'{i2}-{i3}')
        plt.tight_layout()
        plt.savefig(f'box_plot/{i2}-{i3}.png', bbox_inches="tight", pad_inches=0.03)
        plt.show()

    df2 = pd.read_csv('AgentK_2.csv')
    plt.boxplot([df2['10'].dropna(), df2['100'], df2['1000']], labels=["$10^1$", "$10^2$", "$10^3$"], whis=(0, 100))
    plt.xlabel('Domain Size')
    plt.ylabel('Individual Utility')
    # plt.title('All')
    plt.tight_layout()
    plt.savefig('box_plot/All.png', bbox_inches="tight", pad_inches=0.03)
    plt.show()


if __name__ == '__main__':
    main()
