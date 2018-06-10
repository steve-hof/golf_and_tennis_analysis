#!/usr/bin/env
"""
Script for performing data analysis on the 2017 US Open Golf Tournament
"""

import pandas as pd
import numpy as np
import os


def feet_to_yards(element):
    if type(element) is str:
        return element
    elif type(element) is float and element > 100:
        in_yards = element / 36
        return round(in_yards, 1)
    else:
        return element


def main():
    player_df = pd.read_csv('data/current_data_files/us_open_statistics_2017.csv')
    tournament_df = pd.read_csv('data/current_data_files/'
                                'us_open_tournament_info_2017.csv')
    # player_df = player_df.loc[:, ~player_df.columns.str.contains('^Unnamed')]
    # # change driving distance from inches into yards
    # player_df = player_df.applymap(feet_to_yards)
    #
    # # add totals column
    # hole_nums = list(range(1, 19))
    # sum_cols = [str(i) for i in hole_nums]
    # player_df['totals'] = player_df[sum_cols].sum(axis=1)
    # player_df['totals'] = player_df['totals'].map(lambda num: round(num, 1))
    #
    # player_df.to_csv('data/current_data_files/us_open_statistics_2017_'
    #                  'submitted_june_6.csv')
    fill = 12


if __name__ == '__main__':
    main()
