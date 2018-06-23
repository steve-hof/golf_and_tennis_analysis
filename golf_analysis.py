#!/usr/bin/env python3
"""
Script for performing data analysis on the 2017 US Open Golf Tournament
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np
import os
import argparse as ap

BASE_DATA_PATH = 'data/current_data_files/golf/'
BASE_SAVE_PATH = '../current_report_folder/figure_final/'
SAVE_FORMAT = 'eps'

parser = ap.ArgumentParser()
parser.add_argument('tournament', nargs=1, help='name of tournament')
parser.add_argument('year', nargs=1, help='year of tournament')
args = parser.parse_args()

TOURNAMENT = args.tournament[0]
YEAR = args.year[0]
PREVIOUS_YEAR = str(int(YEAR) - 1)

ALL_PLAYERS_FILEPATH = BASE_DATA_PATH + TOURNAMENT + "_" + YEAR + ".csv"
MADE_CUT_FILEPATH = BASE_DATA_PATH + TOURNAMENT + "_" + YEAR + "_made_cut" + ".csv"
BASE_SAVE_PATH_PLUS = BASE_SAVE_PATH + TOURNAMENT + "_" + YEAR + "_"

PREV_YEAR_MADE_CUT_FILEPATH = BASE_DATA_PATH + TOURNAMENT + "_" + PREVIOUS_YEAR + ".csv"

title_dict = {
    'us_open': 'US Open',
    'players_championship': 'Players Championship',
    'masters': 'Masters'
}


def plot_violin(df):
    sns.violinplot(data=df)
    plt.ylim(57, 90)
    title_string = "" + YEAR + " " + title_dict[TOURNAMENT] + " Round vs Round Violin Plot"
    plt.title(title_string)
    plt.plot([0, 72], [50, 72], linewidth=2)
    fig_save_path = BASE_SAVE_PATH_PLUS + "round_by_round_violin." + SAVE_FORMAT

    # plt.show()
    plt.savefig(fig_save_path, format=SAVE_FORMAT, dpi=500)


def plot_putts_regression(df):

    DEBUG = 'Yankees Win!!!'


def plot_heatmap(df):
    sns.heatmap(df.corr(), cmap='coolwarm',annot=True)
    title_string = "Correlation between rounds " + YEAR + " " + title_dict[TOURNAMENT]
    plt.title(title_string)
    fig_save_path = BASE_SAVE_PATH_PLUS + "heatmap." + SAVE_FORMAT
    plt.savefig(fig_save_path, format=SAVE_FORMAT, dpi=500)


def get_round_df(df, type):
    round_1_totals = df['Total'][(df['Round'] == 1) & (df['Type'] == type)]
    round_2_totals = df['Total'][(df['Round'] == 2) & (df['Type'] == type)]
    round_3_totals = df['Total'][(df['Round'] == 3) & (df['Type'] == type)]
    round_4_totals = df['Total'][(df['Round'] == 4) & (df['Type'] == type)]

    round_1_totals = round_1_totals.reset_index(drop=True)
    round_2_totals = round_2_totals.reset_index(drop=True)
    round_3_totals = round_3_totals.reset_index(drop=True)
    round_4_totals = round_4_totals.reset_index(drop=True)

    rounds = {'Round_1': round_1_totals, 'Round_2': round_2_totals, 'Round_3': round_3_totals,
              'Round_4': round_4_totals}

    round_df = pd.DataFrame.from_dict(rounds, orient='columns')
    round_df.dropna(axis=0, inplace=True)

    totals = round_df.sum(axis=1)

    return round_df, totals


def plot_pairplot(df):
    scaler = StandardScaler()
    df.dropna(axis=0, inplace=True)

    scaled_df = pd.DataFrame(scaler.fit_transform(df))
    scaled_df.columns = df.columns

    sns.pairplot(scaled_df)
    fig_save_path_scaling = BASE_SAVE_PATH_PLUS + "pairplot_scaling." + SAVE_FORMAT
    plt.savefig(fig_save_path_scaling, format=SAVE_FORMAT, dpi=500)
    plt.show()

    sns.pairplot(df)
    fig_save_path_no_scaling = BASE_SAVE_PATH_PLUS + "pairplot_no_scaling." + SAVE_FORMAT
    plt.savefig(fig_save_path_no_scaling, format=SAVE_FORMAT, dpi=500)
    plt.show()

    print("DEBUG")


def expand_latex_c(num_columns):
    base_string = '|'
    add_c_string = 'c|'
    for i in range(num_columns - 1):
        add_c_string += add_c_string

    return base_string + add_c_string


def print_latex_table(df, yr='UNK', caption='BLANK', label='BLANK'):
    num_rows = df.shape[0]
    num_cols = df.shape[1]
    col_list = df.columns
    c_string = expand_latex_c(num_cols)
    full_caption = caption + " " + yr

    print()
    print("\\begin{table*}[h]")
    print(f"\\caption{{{caption}}}")
    print(f"\\label{{{label}}}")
    print("\\centering")
    print(f"\\begin{{tabular}}{{{c_string}}}")
    print("\\hline")
    for j, col in enumerate(col_list):
        if j < num_cols - 1:
            print(f"{col} & ", end='')
        else:
            print(f"{col_list[-1]}\\\\")

    for index, row in df.iterrows():
        print("\\hline")
        for i in range(num_cols - 1):
            print(f"{row[col_list[i]]:.1f} & ", end='')

        print(f"{row[col_list[-1]]:.1f}\\\\")
    print("\\end{tabular}")
    print("\\end{table*}")
    print()


def get_descriptive_stats(main_df, prev_df):
    #######################################################
    #         Get descriptive stats for main year         #
    #######################################################

    type_list = ['scores', 'putts', 'drive_dist']
    rounds_scores_df, total_scores_series = get_round_df(main_df, 'scores')
    rounds_putts_df, total_putts_series = get_round_df(main_df, 'putts')
    rounds_drives_df, total_drives_series = get_round_df(main_df, 'drive_dist')

    main_score_means = rounds_scores_df.mean()
    main_putt_means = rounds_putts_df.mean()
    main_drive_means = rounds_drives_df.mean()

    main_score_summary_df = pd.DataFrame([rounds_scores_df.mean(), rounds_scores_df.var()])
    main_score_summary_df = main_score_summary_df.transpose()
    main_score_summary_df.index = ['Round 1', 'Round 2', 'Round 3', 'Round 4']
    main_score_summary_df.index.name = 'Round'
    main_score_summary_df.columns = ['Mean', 'Variance']

    print_latex_table(main_score_summary_df, caption='Mean and Variance per Round', yr=YEAR)

    #######################################################
    #      Get descriptive stats for previous year        #
    #######################################################


def main():
    ###################################################
    ####          Load and clean csv files         ####
    ###################################################
    full_df = pd.read_csv(ALL_PLAYERS_FILEPATH)
    df = pd.read_csv(MADE_CUT_FILEPATH)
    prev_year_df = pd.read_csv(PREV_YEAR_MADE_CUT_FILEPATH)
    
    # clean and ensure no unnamed columns
    full_df.dropna(axis=0, inplace=True)
    full_df = full_df.loc[:, ~full_df.columns.str.contains('^Unnamed')]

    df.dropna(axis=0, inplace=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    prev_year_df.dropna(axis=0, inplace=True)
    prev_year_df = prev_year_df.loc[:, ~prev_year_df.columns.str.contains('^Unnamed')]

    rounds_scores_df, total_scores_series = get_round_df(df, 'scores')
    rounds_putts_df, total_putts_series = get_round_df(df, 'putts')
    rounds_drives_df, total_drives_series = get_round_df(df, 'drive_dist')

    totals_df = pd.concat([total_scores_series, total_putts_series, total_drives_series], axis=1)
    totals_df.columns = ['scores', 'putts', 'drive_dist']

    main_year_stats = get_descriptive_stats(df, prev_year_df)
    # plot_pairplot(totals_df)
    # plot_violin(rounds_scores_df)
    # plot_heatmap(rounds_scores_df)

    DEBUG = 12


if __name__ == '__main__':
    main()
