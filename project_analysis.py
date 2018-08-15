import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats, linalg
import argparse as ap

# TODO: use OS module to properly load and save


class GolfModel:
    def __init__(self, tourn_name, tourn_year, type='scores'):
        self.BASE_DATA_PATH = 'data/current_data_files/golf/'
        self.BASE_SAVE_PATH = '../actual_project/figures/'
        self.SAVE_FORMAT = 'eps'
        self.tournament_name = tourn_name
        self.year = tourn_year
        self.type = type
        
        self.title_dict = {
            'us_open': 'US Open',
            'players_championship': 'Players Championship',
            'masters': 'Masters'
        }
        self.full_data_path = self.BASE_DATA_PATH + self.tournament_name + "_" + self.year + "_made_cut" + ".csv"
        self.full_save_path = self.BASE_SAVE_PATH + self.tournament_name + "_" + self.year + "_"

    def fit(self):
        df = pd.read_csv(self.full_data_path)


def parse_command_line():
    # TODO: add a 'function' argument that represents which functions to initiate
    parser = ap.ArgumentParser()
    parser.add_argument('tournament', nargs=1, help='name of tournament')
    parser.add_argument('year', nargs=1, help='year of tournament')
    args = parser.parse_args()

    tournament_name = args.tournament[0]
    year = args.year[0]

    return tournament_name, year


def main():
    tournament_name, year = parse_command_line()
    tournament_1 = GolfModel(tournament_name, year, type='scores')
    DEBUG = 12


if __name__ == '__main__':
    main()