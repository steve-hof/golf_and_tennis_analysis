#!/usr/bin/env python3
"""
Script for performing data analysis on the 2017 US Open Golf Tournament
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats, linalg
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression

import math

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
    sns.heatmap(df.corr(), cmap='coolwarm', annot=True)
    title_string = "Correlation between rounds " + YEAR + " " + title_dict[TOURNAMENT]
    plt.title(title_string)
    fig_save_path = BASE_SAVE_PATH_PLUS + "heatmap." + SAVE_FORMAT
    plt.savefig(fig_save_path, format=SAVE_FORMAT, dpi=500)


def get_made_cut_X_y_df(df):
    made_round_4_df = df[df['Round'] == 4]
    made_cut_player_list = made_round_4_df['Player'].unique().tolist()

    drives = df[['Player', 'Total']][(df['Round'] == 1) & (df['Type'] == 'drive_dist')]
    drives2 = df[['Player', 'Total']][(df['Round'] == 2) & (df['Type'] == 'drive_dist')]
    prelim_drives = pd.concat([drives, drives2])

    putts = df[['Player', 'Total']][(df['Round'] == 1) & (df['Type'] == 'putts')]
    putts2 = df[['Player', 'Total']][(df['Round'] == 2) & (df['Type'] == 'putts')]
    prelim_putts = pd.concat([putts, putts2])

    reg_df = pd.merge(prelim_putts, prelim_drives, on='Player', how='outer')

    columns = ['Player', 'Putts', 'Drive_Distance']
    reg_df.columns = columns

    scaler = MinMaxScaler()
    reg_df[['Putts', 'Drive_Distance']] = scaler.fit_transform(reg_df[['Putts', 'Drive_Distance']])
    reg_df['Made_Cut'] = np.where(reg_df['Player'].isin(made_cut_player_list), 1, 0)

    pd.to_pickle(reg_df, '../forClustering')
    y = np.where(reg_df['Player'].isin(made_cut_player_list), 1, 0)
    X = reg_df.drop(['Player'], axis=1)

    return X, y


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


def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def expand_latex_c(num_columns):
    base_string = '|'
    add_c_string = 'c|'
    full_string = "|"
    for i in range(num_columns - 1):
        full_string += add_c_string

    return full_string + add_c_string


def print_latex_table(df_start, yr='UNK', caption='BLANK', label='BLANK', index_name="UNK"):
    # Turn index into a column and re-order for display
    df = df_start
    df[index_name] = df.index
    cols = df.columns.tolist()
    cols.insert(0, cols.pop())
    df = df[cols]

    num_cols = df.shape[1]
    col_list = df.columns
    c_string = expand_latex_c(num_cols)
    df.round(3)
    print(f"Figure \\ref{{{label}}} shows {caption}")
    print("\n\\begin{table*}[h]")
    print(f"\\caption{{{caption}}}")
    print(f"\\label{{{label}}}")
    print("\\centering")
    print(f"\\begin{{tabular}}{{{c_string}}}\n\\hline")
    for j, col in enumerate(col_list):
        if j < num_cols - 1:
            print(f"{col} & ", end='')
        else:
            print(f"{col_list[-1]}\\\\")

    for index, row in df.iterrows():
        print("\\hline")
        for i in range(num_cols - 1):
            if not is_numeric(str(row[col_list[i]])):
                print(f"{row[col_list[i]]} & ", end="")
            else:
                print(f"{row[col_list[i]]:.3f} & ", end='')
        if not is_numeric(str(row[col_list[-1]])):
            print(f"{row[col_list[-1]]}")
        else:
            print(f"{row[col_list[-1]]:.3f}\\\\")
    print("\\hline\n\\end{tabular}\n\\end{table*}\n")


def get_descriptive_stats(main_df, prev_df):
    ##########################################################################
    # Get descriptive stats for scores, putts and driving distance per year  #
    ##########################################################################
    df_list = [main_df, prev_df]

    for index, df in enumerate(df_list):
        rounds_scores_df, total_scores_series = get_round_df(df, 'scores')
        rounds_putts_df, total_putts_series = get_round_df(df, 'putts')
        rounds_drives_df, total_drives_series = get_round_df(df, 'drive_dist')

        type_list = ['scores', 'putts', 'drive_dist']

        for kind in type_list:
            if kind == 'scores':
                summary_df = pd.DataFrame([rounds_scores_df.mean(), rounds_scores_df.var()])
            elif kind == 'putts':
                summary_df = pd.DataFrame([rounds_putts_df.mean(), rounds_putts_df.var()])
            elif kind == 'drive_dist':
                summary_df = pd.DataFrame([rounds_drives_df.mean(), rounds_drives_df.var()])

            summary_df = summary_df.transpose()
            summary_df.index = ['Round 1', 'Round 2', 'Round 3', 'Round 4']
            summary_df.index.name = 'Round'
            summary_df.columns = ['Mean', 'Variance']
            if index == 0:
                curr_year = YEAR
            else:
                curr_year = PREVIOUS_YEAR

            if kind == 'scores':
                curr_label = 'round_score_stat_' + curr_year
                cap = 'Scores per Round for ' + title_dict[TOURNAMENT] + ' ' + curr_year
            elif kind == 'putts':
                curr_label = 'round_putt_stat_' + curr_year
                cap = 'Putts per Round for ' + title_dict[TOURNAMENT] + ' ' + curr_year
            else:
                curr_label = 'round_drive_stat_' + curr_year
                cap = 'Driving Distance (in yards) per Round for ' + title_dict[TOURNAMENT] + ' ' + curr_year

            print_latex_table(summary_df, caption=cap, label=curr_label, yr=curr_year, index_name='Round')


def test_year_significance(main_df, prev_df):
    df_list = [main_df, prev_df]

    round_scores_main_df, _ = get_round_df(main_df, 'scores')
    round_scores_prev_df, _ = get_round_df(prev_df, 'scores')

    least_num_rows = min(round_scores_main_df.shape[0], round_scores_prev_df.shape[0])

    # 2017 had an extra 2 players
    round_scores_prev_df.drop(round_scores_prev_df.tail(2).index, inplace=True)

    main_list = []
    for col in round_scores_main_df:
        main_list.extend(round_scores_main_df[col])

    prev_list = []
    for col in round_scores_prev_df:
        prev_list.extend(round_scores_prev_df[col])

    t_test = stats.ttest_ind(main_list, prev_list)
    print("DEBUG")


def test_round_significance(main_df, prev_df):
    # Round 1 vs each other round

    df_list = [main_df, prev_df]

    for index, df in enumerate(df_list):
        rounds_scores_df, total_scores_series = get_round_df(df, 'scores')
        rounds_putts_df, total_putts_series = get_round_df(df, 'putts')
        rounds_drives_df, total_drives_series = get_round_df(df, 'drive_dist')

        type_list = ['scores', 'putts', 'drive_dist']

        for kind in type_list:
            sig = []
            pval = []
            if kind == 'scores':
                base_list = rounds_scores_df['Round_1'].values
                compare_list = rounds_scores_df.columns[1:]
                for round in compare_list:
                    sig.append(stats.ttest_ind(base_list, rounds_scores_df[round].values)[0])
                    pval.append(stats.ttest_ind(base_list, rounds_scores_df[round].values)[1])

                summary_df = pd.DataFrame([sig, pval])

            elif kind == 'putts':
                base_list = rounds_putts_df['Round_1'].values
                compare_list = rounds_putts_df.columns[1:]
                for round in compare_list:
                    sig.append(stats.ttest_ind(base_list, rounds_putts_df[round].values)[0])
                    pval.append(stats.ttest_ind(base_list, rounds_putts_df[round].values)[1])

                summary_df = pd.DataFrame([sig, pval])

            elif kind == 'drive_dist':
                base_list = rounds_drives_df['Round_1'].values
                compare_list = rounds_drives_df.columns[1:]
                for round in compare_list:
                    sig.append(stats.ttest_ind(base_list, rounds_drives_df[round].values)[0])
                    pval.append(stats.ttest_ind(base_list, rounds_drives_df[round].values)[1])

                summary_df = pd.DataFrame([sig, pval])

            # index_list = ['Round 2', 'Round 3', 'Round 4']
            # col_list = ['Test Statistic', 'P Value']

            summary_df = summary_df.transpose()
            summary_df.index = ['Round 2', 'Round 3', 'Round 4']
            summary_df.index.name = 'Round'
            summary_df.columns = ['Test Statistic', 'P Value']

            # summary_df = pd.DataFrame(index=index_list, columns=col_list)
            # summary_df[col_list[0]] = sig
            # summary_df[col_list[1]] = pval

            if index == 0:
                curr_year = YEAR
            else:
                curr_year = PREVIOUS_YEAR

            if kind == 'scores':
                curr_label = 'sig_test_scores_' + curr_year
                cap = "" + YEAR + " " + title_dict[
                    TOURNAMENT] + ' Significance Testing for Scores (Round 1 vs Other Rounds)'
            elif kind == 'putts':
                curr_label = 'sig_test_putts_' + curr_year
                cap = "" + YEAR + " " + title_dict[
                    TOURNAMENT] + ' Significance Testing for Putts (Round 1 vs Other Rounds)'
            else:
                curr_label = 'round_drive_stat_' + curr_year
                cap = "" + YEAR + " " + title_dict[
                    TOURNAMENT] + ' Significance Testing for Driving Distance (Round 1 vs Other Rounds)'

            print_latex_table(summary_df, caption=cap, label=curr_label, yr=curr_year, index_name='Round')

    DEBUG = 256


def plot_made_cut_regression(df):
    df.dropna(axis=0, inplace=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    X, y = get_made_cut_X_y_df(df)

    DEBUG = 12


def hole_to_hole(frame):
    score_dict = {6: 'bogey', 5: 'bogey', 4: 'bogey', 3: 'bogey', 2: 'bogey', 1: 'bogey', 0: 'par', -1: 'birdie',
                  -2: 'birdie', -3: 'birdie'}

    scores_by_hole_df = frame.drop(['In', 'Out'], axis=1)
    pars = scores_by_hole_df.iloc[0, 2:20]
    scores_by_hole_df = scores_by_hole_df[scores_by_hole_df['Type'] == 'scores']
    relative_score_df = scores_by_hole_df.iloc[1:, 2:20].apply(lambda row: row - pars, axis=1)
    relative_score_df = relative_score_df.reset_index(drop=True)

    relative_score_df = relative_score_df.replace(score_dict)
    relative_score_df['padding'] = 'null'

    df = relative_score_df.stack().T
    df = df.reset_index(0).reset_index(drop=True)

    df = df.drop(['level_0'], axis=1)
    df.columns = ['first_hole']
    df['second_hole'] = df.shift(-1)

    df = df[df.first_hole != 'null']
    df = df[df.second_hole != 'null']

    score_table = pd.crosstab(df.first_hole, df.second_hole, margins=True)
    index_list = ['birdie', 'par', 'bogey', 'All']
    col_list = ['birdie', 'par', 'bogey', 'All']
    score_table = score_table[col_list]
    score_table = score_table.loc[index_list]
    score_table.index = ['birdie', 'par', 'bogey', 'col_totals']
    score_table.columns = ['birdie', 'par', 'bogey', 'row_totals']

    observed = score_table.iloc[0:3, 0:3]

    n = score_table.iloc[3, 3]
    expected = np.outer(score_table["row_totals"][0:3],
                        score_table.loc["col_totals"][0:3]) / n

    expected = pd.DataFrame(expected)
    expected.columns = ['birdie', 'par', 'bogey']
    expected.index = ['birdie', 'par', 'bogey']

    # TODO: Give description of what is being measured in printouts below
    print()
    print("#############################################################")
    print()
    chi_squared_stat = (((observed - expected) ** 2) / expected).sum().sum()
    print("Null Hypothesis: The result on a previous hole has no effect on the result of the current hole")
    print()
    print(f"Chi Squared Test Statistic = {chi_squared_stat}")

    crit = stats.chi2.ppf(q=0.95,  # Find the critical value for 95% confidence*
                          df=4)  # *

    print(f"Critical value = {crit}")

    p_value = 1 - stats.chi2.cdf(x=chi_squared_stat,  # Find the p-value
                                 df=4)
    print(f"P value = {p_value}")
    print()
    # print("#############################################################")
    DEBUG = 28


# def gather_sequential_data(df):
#     scores_by_hole_df = df.drop(['In', 'Out'], axis=1)
#     pars = scores_by_hole_df.iloc[0, 2:20]
#     scores_by_hole_df = scores_by_hole_df[scores_by_hole_df['Type'] == 'scores']
#     relative_score_df = scores_by_hole_df.iloc[1:, 2:20].apply(lambda row: row - pars, axis=1)
#     relative_score_df = relative_score_df.reset_index(drop=True)
#     observed = np.zeros((3, 3))
#     score_values = relative_score_df.values
#
#     for x in range(score_values.shape[0] - 1):
#         for y in range(score_values.shape[1] - 2):
#             i = x + 1
#             j = y + 1
#             if score_values[x, y] < 0 and score_values[i, j] < 0:
#                 observed[0, 0] = observed[0, 0] + 1
#             elif score_values[x, y] < 0 and score_values[i, j] == 0:
#                 observed[0, 1] = observed[0, 1] + 1
#             elif score_values[x, y] < 0 and score_values[i, j] > 0:
#                 observed[0, 2] = observed[0, 2] + 1
#
#             elif score_values[x, y] == 0 and score_values[i, j] < 0:
#                 observed[1, 0] = observed[1, 0] + 1
#             elif score_values[x, y] == 0 and score_values[i, j] == 0:
#                 observed[1, 1] = observed[1, 1] + 1
#             elif score_values[x, y] == 0 and score_values[i, j] > 0:
#                 observed[1, 2] = observed[1, 2] + 1
#
#             elif score_values[x, y] > 0 and score_values[i, j] < 0:
#                 observed[2, 0] = observed[2, 0] + 1
#             elif score_values[x, y] > 0 and score_values[i, j] == 0:
#                 observed[2, 1] = observed[2, 1] + 1
#             elif score_values[x, y] > 0 and score_values[i, j] > 0:
#                 observed[2, 2] = observed[2, 2] + 1
#
#     contingency_table = pd.DataFrame(data=observed, index=['Bog_plus', 'Par', 'Bird-'],
#                                      columns=['Bog_plus', 'Par', 'Bird-'])
#     contingency_table['Row_Totals'] = contingency_table.sum(axis=1)
#     contingency_table.loc['Col_Totals'] = contingency_table.sum(axis=0)
#
#     total = contingency_table.iloc[3, 3]
#     expected = np.outer(contingency_table['Row_Totals'][0:3], contingency_table.loc['Col_Totals'][0:3]) / total
#     expected = pd.DataFrame(data=expected, index=['Bog_plus', 'Par', 'Bird-'],
#                             columns=['Bog_plus', 'Par', 'Bird-'])
#
#     chi_squared_stat = (((observed - expected) ** 2) / expected).sum().sum()
#     print(f"Chi Squared Test Statistic = {chi_squared_stat}")
#
#     crit = stats.chi2.ppf(q=0.95,  # Find the critical value for 95% confidence*
#                           df=4)  # *
#
#     print(f"Critical value = {crit}")
#
#     p_value = 1 - stats.chi2.cdf(x=chi_squared_stat,  # Find the p-value
#                                  df=4)
#     print(f"P value = {p_value}")
#
#     DEBUG = 2


def hole_independence(df):
    scores_by_hole_df = df.drop(['In', 'Out'], axis=1)
    pars = scores_by_hole_df.iloc[0, 2:20]
    scores_by_hole_df = scores_by_hole_df[scores_by_hole_df['Type'] == 'scores']
    scores_by_hole_df = scores_by_hole_df.drop('Round', axis=1)
    hole_average_df = scores_by_hole_df.groupby('Player').mean()
    hole_average_df = hole_average_df.drop(['Total'], axis=1)

    covariance_matrix = hole_average_df.cov()

    LR, logLR, u, u_prime, p = sphericity(covariance_matrix, hole_average_df.shape[0])

    degrees_freedom = (.5 * p) * (p + 1) - 1
    crit_val = stats.chi2.ppf(q=.95, df=degrees_freedom)

    # hole_average_df.loc['Sample_Variance'] = hole_average_df.apply(lambda col: stats.tvar(col), axis=0)
    # hole_average_df.loc['Estimated_Variance (population)'] = hole_average_df.drop(['Sample_Variance'], axis=0).apply(
    #     lambda col: np.var(col.values))

    DEBUG = 2


def sphericity(covariance_matrix, n):
    p = covariance_matrix.shape[0]
    det = linalg.det(covariance_matrix)
    trace = np.trace(covariance_matrix)
    LR = (det / (trace / p)**p) ** (n / 2)

    log_LR = -2 * np.log(LR)

    u = ((p ** p) * det) / (trace ** p)
    eig_ps = (2 * p**2 + p + 2) / (6 * p)
    u_prime = -((n - 1) - eig_ps) * np.log(u)

    return LR, log_LR, u, u_prime, p


def two_way_anova_practice():
    sydney = np.array([75, 70, 50, 65, 80, 65])
    brisbane = np.array([75, 70, 55, 60, 65, 65])
    melbourne = np.array([90, 70, 75, 85, 80, 65])


    DEBUG = 6


def practice_question():
    height = np.array([69, 74, 68, 70, 72, 67, 66, 70, 76, 68, 72, 79, 74, 67, 66, 71, 74, 75, 75, 76])
    weight = np.array([153, 175, 155, 135, 172, 150, 115, 137, 200, 130, 140, 265, 185, 112, 140, 150, 165, 185, 210, 220])

    height_mean = np.mean(height)
    weight_mean = np.mean(weight)

    dot_prod = np.dot(height, weight)
    S_xy = (dot_prod - height.size * height_mean * weight_mean) / (height.size - 1)
    DEBUG = 2


def pq_2():
    y1 = np.array([51, 27, 37, 42, 27, 43, 41, 38, 36, 26, 29])
    y2 = np.array([36, 20, 22, 36, 18, 32, 22, 21, 23, 31, 20])
    y3 = np.array([50, 26, 41, 32, 33, 43, 36, 31, 27, 31, 25])
    y4 = np.array([35, 17, 37, 34, 14, 35, 25, 20, 25, 32, 26])
    y5 = np.array([42, 27, 30, 27, 29, 40, 38, 16, 28, 36, 25])

    word_df = pd.DataFrame({'y1': y1, 'y2': y2, 'y3': y3, 'y4': y4, 'y5': y5})
    # y_means = word_df.mean(axis=0)
    S = word_df.cov()

    LR, logLR, u, u_prime, p = sphericity(S, word_df.shape[0])

    degrees_freedom = (.5 * p) * (p + 1) - 1

    crit_val = stats.chi2.ppf(q=.95, df=degrees_freedom)

    DEGUG = 27


def round_independence(frame):
    # TODO: add titles to all plots
    frame = frame[frame['Type'] == 'scores']
    df = frame[['Player', 'Round', 'Total']]

    scores_per_round_df = df.pivot(index='Player', columns='Round', values='Total')
    scores_per_round_df.columns = ['Round_1', 'Round_2', 'Round_3', 'Round_4']
    covariance_matrix = scores_per_round_df.cov()

    LR, logLR, u, u_prime, p = sphericity(covariance_matrix, scores_per_round_df.shape[0])
    degrees_freedom = (.5 * p) * (p + 1) - 1
    crit_val = stats.chi2.ppf(q=.95, df=degrees_freedom)

    eigval, eigvec = linalg.eig(covariance_matrix)
    starting_round = scores_per_round_df.Round_1.values
    starting_round = np.append(starting_round, scores_per_round_df.Round_2.values)
    starting_round = np.append(starting_round, scores_per_round_df.Round_3.values)

    next_round = scores_per_round_df.Round_2.values
    next_round = np.append(next_round, scores_per_round_df.Round_3.values)
    next_round = np.append(next_round, scores_per_round_df.Round_4.values)

    round_to_round_df = pd.DataFrame({'Initial_Round': starting_round, 'Next_Round': next_round})
    sns.scatterplot(x='Initial_Round', y='Next_Round', data=round_to_round_df)  # , hue='size', size='size')
    plt.title("Scatter Plot showing Previous Round Score vs Next")
    plt.show()

    correlation_matrix = scores_per_round_df.corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True)
    plt.title("Correlation Matrix of Rounds (2018)")
    plt.show()

    # sns.jointplot(x='Round_1', y='Round_2', data=scores_per_round_df, kind='scatter', cmap='coolwarm')

    sns.pairplot(scores_per_round_df, palette='coolwarm')
    plt.title("Scatter Plots showing Relationships between round scores")
    # g = sns.PairGrid(scores_per_round_df)
    # g.map_diag(plt.hist)
    # g.map_upper(plt.scatter)
    # g.map_lower(sns.kdeplot)

    plt.show()

    DEBUG = 2


def sum_square(V, mean):
    ssq = 0
    for v in V:
        ssq += (v - mean) ** 2
    return ssq

    DEBUG = 28


def anova(*args):
    N = 0
    num_columns = len(args)

    # Calculate degrees of freedom
    df_col = num_columns - 1
    df_err = N - num_columns
    df_tot = N - 1

    # Calculate N and sum of all data points
    total = 0
    for arr in args:
        for i in arr:
            total += i
            N += 1
    correction_mean = total ** 2 / N

    # Calculate sum of squares total (SS_Total)
    SS_Total = 0
    for arr in args:
        SS_Total += np.dot(arr, arr)
    SS_Total = SS_Total - correction_mean

    # Calculate Treatment Sum of Squares (SST) (a treatment is a column)
    SS_Treatment = 0
    for arr in args:
        SS_Treatment += ((np.sum(arr)) ** 2 / arr.size)
    SS_Treatment = SS_Treatment - correction_mean

    # Calculate Error Sum of Squares (SSE)
    SS_Error = SS_Total - SS_Treatment

    # Calculate Mean Squared of Treatment, Mean Squared Error, and F Ratio
    MST = SS_Treatment / df_col
    MSE = SS_Error / df_err
    F = MST / MSE

    DEBUG = 28
    return MST, MSE, F


def mucking_about():
    year_1_scores = np.array([82, 93, 61, 74, 69, 70, 53])
    year_2_scores = np.array([71, 62, 85, 94, 78, 66, 71])
    year_3_scores = np.array([65, 73, 87, 91, 56, 78, 87])

    one = np.array([6.9, 5.4, 5.8, 4.6, 4.0])
    two = np.array([8.3, 6.8, 7.8, 9.2, 6.5])
    three = np.array([8.0, 10.5, 8.1, 6.9, 9.3])

    MSE, MST, F_ratio = anova(one, two, three)

    DEBUG = 12


def main():
    ###################################################
    ####          Load and clean csv files         ####
    ###################################################

    # We're concerned mostly with df and prev_year_df, full is just here
    # in case we need it for something I haven't thought of yet
    # mucking_about()
    # two_way_anova_practice()
    # pq_2()
    # practice_question()
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

    # main_year_stats = get_descriptive_stats(df, prev_year_df)
    # plot_pairplot(totals_df)
    # plot_violin(rounds_scores_df)
    # plot_heatmap(rounds_scores_df)
    # test_round_significance(df, prev_year_df)
    # test_year_significance(df, prev_year_df)

    hole_to_hole(df)
    hole_independence(df)
    round_independence(df)
    # plot_made_cut_regression(full_df)
    DEBUG = 12


if __name__ == '__main__':
    main()
