import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats, linalg
import argparse as ap


# TODO: use OS module to properly load and save
# TODO: maybe just have one file and it scrapes the tournament if necessary?


class GolfModel:

    def __init__(self, tourn_name, tourn_year, functions):
        self.BASE_DATA_PATH = 'data/current_data_files/golf/'
        self.BASE_SAVE_PATH = '../actual_project/figures/'
        self.SAVE_FORMAT = 'eps'
        self.tournament_name = tourn_name
        self.year = tourn_year

        self.title_dict = {
            'us_open': 'US Open',
            'players_championship': 'Players Championship',
            'masters': 'Masters'
        }
        self.full_data_path = self.BASE_DATA_PATH + self.tournament_name + "_" + self.year + "_made_cut" + ".csv"
        self.full_save_path = self.BASE_SAVE_PATH + self.tournament_name + "_" + self.year + "_"
        self.df = pd.read_csv(self.full_data_path)
        self.functions = functions

    def fit(self, data_type='scores'):
        df = self.df
        df.dropna(axis=0, inplace=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        self.df = df[df['Type'] == data_type]
        if 'h2h' in self.functions:
            self._perform_hole_to_hole_analysis()
        DEBUG = 2

    # TODO: create function: _perform_hole_to_hole analysis()
    # TODO: create function: _perform_hole_independence_test()
    # TODO: create function: _perform_round_independence_test()
    # TODO: create function: _perform_covariance_likelihood()

    def _perform_hole_to_hole_analysis(self):
        score_dict = {6: 'bogey', 5: 'bogey', 4: 'bogey', 3: 'bogey', 2: 'bogey', 1: 'bogey', 0: 'par', -1: 'birdie',
                      -2: 'birdie', -3: 'birdie'}

        df = self.df.drop(['In', 'Out'], axis=1)
        pars = df.iloc[0, 2:20]
        df = df.iloc[1:, 2:20].apply(lambda row: row - pars, axis=1)
        df = df.reset_index(drop=True)

        df = df.replace(score_dict)
        df['padding'] = 'null'

        df = df.stack().T
        df = df.reset_index(0).reset_index(drop=True)

        df = df.drop(['level_0'], axis=1)
        df.columns = ['first_hole']
        df['second_hole'] = df.shift(-1)

        df = df[df.first_hole != 'null']
        samples_df = df[df.second_hole != 'null']

        critical_value, test_statistic, p_val = self._perform_chi_squared(samples_df)

        print()
        print("#############################################################")
        print()
        print("Null Hypothesis: ")

    @staticmethod
    def _perform_chi_squared(samples):
        full_table = pd.crosstab(samples.first_hole, samples.second_hole, margins=True)
        num_cols = full_table.shape[0] - 1
        n = full_table.iloc[num_cols, num_cols]

        observed = full_table.iloc[0:num_cols, 0:num_cols]
        expected = np.outer(full_table["All"][0:num_cols],
                            full_table.loc["All"][0:num_cols]) / n

        expected = pd.DataFrame(expected)
        expected.columns = ['birdie', 'bogey', 'par']
        expected.index = ['birdie', 'bogey', 'par']

        # TODO: set up degrees of freedom to be dynamic
        chi_squared_stat = (((observed - expected) ** 2) / expected).sum().sum()
        crit = stats.chi2.ppf(q=0.95,  # Find the critical value for 95% confidence*
                              df=4)

        p_value = 1 - stats.chi2.cdf(x=chi_squared_stat,  # Find the p-value
                                     df=4)

        return crit, chi_squared_stat, p_value
        DEBUG = 1024


def parse_command_line():
    # TODO: add a 'function' argument that represents which functions to initiate
    parser = ap.ArgumentParser()
    parser.add_argument('tournament', nargs=1, help='name of tournament')
    parser.add_argument('year', nargs=1, help='year of tournament')
    parser.add_argument('analyses', nargs='?', const=1, help='analysis to perform')
    args = parser.parse_args()

    tournament_name = args.tournament[0]
    year = args.year[0]
    functions = args.analyses

    return tournament_name, year, functions


def main():
    tournament_name, year, functions = parse_command_line()
    tournament_1 = GolfModel(tournament_name, year, functions)
    tournament_1.fit(data_type='scores')

    DEBUG = 12


if __name__ == '__main__':
    main()
