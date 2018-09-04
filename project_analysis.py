import numpy as np
import pandas as pd
import os
from pathlib2 import Path
from scipy import stats, linalg
import argparse as ap

import matplotlib.pyplot as plt
import seaborn as sns


# TODO: create new file that aggregates tournaments and years of tournaments
# TODO: set up argparse flag for whether to save plots


class GolfModel:

    def __init__(self, tourn_name, tourn_year, functions):
        self.BASE_DATA_PATH = './data/golf/'
        self.BASE_SAVE_PATH = '../actual_project/figures/'
        self.PICKLE_PATH = './jupyter_notebooks/data_and_pickles/'
        self.SAVE_FORMAT = 'eps'

        self.tournament_name = tourn_name
        self.year = tourn_year

        self.players_champ_slug = "players_championship"
        self.us_open_slug = "us_open"
        self.masters_slug = "masters"
        self.open_slug = "open"
        self.pga_champ_slug = "pga_championship"

        self.title_dict = {
            'us_open': 'US Open',
            'players_championship': 'Players Championship',
            'masters': 'Masters',
            'open': 'Open Championship',
            'pga_championship': 'PGA Championship'
        }

        if not os.path.exists(self.BASE_DATA_PATH):
            # create directory
            os.makedirs(self.BASE_DATA_PATH)
            print("The necessary data directory does not exist. Creating directory now.")

        self.full_data_path = self.BASE_DATA_PATH + self.tournament_name + "_" + self.year + "_made_cut" + ".csv"
        self.full_save_path = self.BASE_SAVE_PATH + self.tournament_name + "_" + self.year + "_"
        self.functions = functions

        if not os.path.exists(self.PICKLE_PATH):
            # create directory
            os.makedirs(self.PICKLE_PATH)
            print("The necessary pickle directory does not exist. Creating directory now.")

        # check for file
        csv_file = Path(self.full_data_path)
        if csv_file.is_file():
            # import csv file to dataframe
            self.df = pd.read_csv(self.full_data_path)
        else:
            print("\nYou do not have the proper csv file.\n")
            print(f"Attempting to scrape {self.year} {self.title_dict[self.tournament_name]}....................")
            execution_string = f"python3 golf_scraper.py {self.tournament_name} {self.year}"
            os.system(execution_string)
            self.df = pd.read_csv(self.full_data_path)

    def fit(self, data_type='scores'):
        df = self.df
        df.dropna(axis=0, inplace=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        self.df = df[df['Type'] == data_type]
        self._print_title()
        if 'hole_2_hole' in self.functions:
            self._perform_hole_to_hole_analysis()

        if 'hole_independence' in self.functions:
            self._perform_hole_independence_test()

        if 'round_independence' in self.functions:
            self._perform_round_independence_test()

        elif len(self.functions) == 0:
            self._perform_hole_to_hole_analysis()
            self._perform_hole_independence_test()
            self._perform_round_independence_test()

        return

    def _perform_hole_to_hole_analysis(self):
        score_dict = {11: 'bogey', 10: 'bogey', 9: 'bogey', 8: 'bogey', 7: 'bogey', 6: 'bogey', 5: 'bogey', 4: 'bogey', 3: 'bogey',
                      2: 'bogey', 1: 'bogey', 0: 'par', -1: 'birdie',
                      -2: 'birdie', -3: 'birdie', -4: 'birdie', -5: 'birdie', -6: 'birdie'}

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
        hypothesis_result = self._p_value_decision(p_val)

        print()
        # print("#########################################################################")
        print("NULL HYPOTHESIS: The result on a previous hole has no effect on the result \n of the current hole")
        # print("#########################################################################")
        # print()
        print(f"Critical value = {critical_value: .4f}")
        print(f"Chi Squared Test Statistic = {test_statistic: .4f}")
        print(f"P-Value = {p_val}")
        # print()
        # TODO: make the decision statement dependant on the p_value

        print(f"DECISION: {hypothesis_result}")
        # print()
        # print("#########################################################################")
        # print()
        print()

    def _perform_hole_independence_test(self):
        null_hypothesis = 'The performances of each of the 18 holes are indpendent from one another'
        df = self.df
        df = df.drop(['In', 'Out'], axis=1)
        # TODO: double check that the par scores are not being included here
        df = df.drop('Round', axis=1)
        hole_average_df = df.groupby('Player').mean()
        hole_average_df = hole_average_df.drop(['Total'], axis=1)

        covariance_matrix = hole_average_df.cov()

        print("Covariance Matrix")
        print(covariance_matrix.round(decimals=4).to_latex())
        cov_path = self.PICKLE_PATH + 'hole_covariance_matrix_' + self.tournament_name + '_' + self.year
        covariance_matrix.to_pickle(cov_path)
        self._perform_sphericity_test(covariance_matrix, hole_average_df.shape[0],
                                      null_hypothesis)

        self._create_correlation_heatmap(hole_average_df.corr(), 'Holes')
        # TODO: plot bar chart of variance per hole

        self._create_bar_plot(covariance_matrix.columns, covariance_matrix.values.diagonal(), measure='Variance',
                              between='Hole',
                              limit=0, ylim=(5, 10))

    def _perform_round_independence_test(self):
        null_hypothesis = "The performance of the 4 rounds are independent"
        df = self.df
        df = df[['Player', 'Round', 'Total']]

        df = df.pivot(index='Player', columns='Round', values='Total')
        df.columns = ['Round_1', 'Round_2', 'Round_3', 'Round_4']

        covariance_matrix = df.cov()
        # cov_path = self.PICKLE_PATH + 'round_covariance_matrix_' + self.tournament_name + '_' + self.year
        # covariance_matrix.to_pickle(cov_path)

        # print("Covariance Matrix")
        # print(covariance_matrix.to_latex())
        print("Correlation Matrix")
        print(df.corr().to_latex())

        self._perform_sphericity_test(covariance_matrix, df.shape[0], null_hypothesis)
        self._create_correlation_heatmap(df.corr(), 'Rounds')

        self._create_bar_plot(covariance_matrix.columns, covariance_matrix.values.diagonal(), measure='Variance',
                              between='Round',
                              limit=0, ylim=(5, 10))

    def _print_title(self):
        print()
        print()
        print("             The following results are for the                    ".upper())
        print(f"                      {self.year}   {self.title_dict[self.tournament_name].upper()}              ")
        print()

        return

    def _create_correlation_heatmap(self, corr_mat, corr_type):
        title = f"Correlation between {corr_type} of {self.year} {self.title_dict[self.tournament_name]}"
        sns.heatmap(corr_mat, cmap='coolwarm', annot=False)
        plt.title(title)
        save_path = self.BASE_SAVE_PATH + title.replace(' ', '_') + '.eps'
        plt.title(title)
        # plt.tight_layout()
        plt.savefig(save_path, format='eps', dpi=1000)
        # plt.show()

        DEBUG = 12

    def _create_bar_plot(self, x, y, measure, between, limit=0, ylim=(0, 0)):
        title = f"{measure} of each {between} for {self.year} {self.title_dict[self.tournament_name]}"
        sns.barplot(x=x, y=y, palette='coolwarm')
        plt.title(title)
        if limit == 1:
            a = ylim[0]
            b = ylim[1]
            plt.ylim(a, b)
        save_path = self.BASE_SAVE_PATH + title.replace(' ', '_') + '.eps'
        plt.savefig(save_path, format='eps', dpi=500)
        plt.tight_layout()
        # plt.show()

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

    @staticmethod
    def _perform_sphericity_test(cov, n, null):
        p = cov.shape[0]
        det = linalg.det(cov)
        trace = np.trace(cov)
        LR = (det / (trace / p) ** p) ** (n / 2)

        log_LR = -2 * np.log(LR)
        u = ((p ** p) * det) / (trace ** p)
        eig_ps = (2 * p ** 2 + p + 2) / (6 * p)
        u_prime = -((n - 1) - eig_ps) * np.log(u)

        degrees_freedom = (.5 * p) * (p + 1) - 1
        critical_value = stats.chi2.ppf(q=.95, df=degrees_freedom)

        print()
        print(f"NULL HYPOTHESIS: {null}")
        if LR < 0.0000:
            print(f"Likelihood Ratio = {LR}")
        else:
            print(f"Likelihood Ratio = {LR}")
        print(f"Critical value (chi squared at .95 on {degrees_freedom}"
              f" degrees of freedom = {critical_value: .4f}")
        print(f"-2ln(LR) = {log_LR: .4f}")
        print(f"U-Prime = {u_prime: .4f}")
        # print()
        if u_prime > critical_value:
            print("DECISION: We have evidence to reject the null hypothesis")
        else:
            print("DECISION: We fail to reject the null hypothesis")
        print()

        return

    @staticmethod
    def _p_value_decision(pval):
        decision = ""
        if pval > 0.1:
            decision = "There is no evidence against the null hypothesis"
        elif 0.1 > pval >= 0.05:
            decision = "There is marginal evidence against the null hypothesis"
        elif 0.05 > pval >= 0.1:
            decision = "There is significant evidence against the null hypothesis"
        elif pval < 0.01:
            decision = "There is very significant evidence against the null hypothesis"

        return decision


def parse_command_line():
    # TODO: add a 'function' argument that represents which functions to initiate
    parser = ap.ArgumentParser()
    parser.add_argument('tournament', nargs=1, help='name of tournament')
    parser.add_argument('year', nargs=1, help='year of tournament')
    parser.add_argument('analyses', nargs='*', help='analysis to perform')
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
