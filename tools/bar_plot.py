from matplotlib import pyplot as plt
import pandas as pd


def main():
    labels = ["", "", "", "", ""]
    # ANACじゃないほう
    df_s = pd.read_csv('VeNAS_simple.csv', index_col=0)
    print(df_s)

    df_s.plot.bar()
    plt.tight_layout()
    # plt.xlim(0, 1.05)
    plt.legend(loc='lower left')
    # plt.savefig('venas.png', bbox_inches="tight", pad_inches=0.03)
    plt.show()

    # ANAC
    df_a = pd.read_csv('VeNAS_anac.csv', index_col=0)
    print(df_a)

    df_a.plot.bar()
    plt.tight_layout()
    # plt.xlim(0, 1.05)
    plt.legend(loc='lower left')
    # plt.savefig('venas.png', bbox_inches="tight", pad_inches=0.03)
    plt.show()


if __name__ == '__main__':
    main()
