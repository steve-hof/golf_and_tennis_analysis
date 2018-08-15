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
    def __init__(self, tourn_name, tourn_year):
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

    def fit(self, data_type='scores'):
        df = self.df
        df.dropna(axis=0, inplace=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        self.df = df[df['Type'] == data_type]

    # TODO: create function: _perform_hole_to_hole analysis()
    # TODO: create function: _perform_hole_independence_test()
    # TODO: create function: _perform_round_independence_test()
    # TODO: create function: _perform_covariance_likelihood()


def parse_command_line():
    # TODO: add a 'function' argument that represents which functions to initiate
    parser = ap.ArgumentParser()
    parser.add_argument('tournament', nargs=1, help='name of tournament')
    parser.add_argument('year', nargs=1, help='year of tournament')
    parser.add_argument('analyses', nargs='?', help='analysis to perform')
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