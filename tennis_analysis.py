#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.stats import logistic
from matplotlib.ticker import NullFormatter
from sklearn import linear_model
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
import collections
import math


def compare_elos(row):  # winner_elo, loser_elo):
    # elo_diff = max(winner_elo, loser_elo) - min(winner_elo, loser_elo)
    # result = max(winner_elo, loser_elo) == winner_elo
    elo_diff = max(row['winner_elo'], row['loser_elo']) - min(row['winner_elo'], row['loser_elo'])
    result = max(row['winner_elo'], row['loser_elo']) == row['winner_elo']

    last = (elo_diff, result)
    return last


def plot_logit(source_list):
    X, bool_y = zip(*source_list)
    X = np.asarray(X)
    X = X.reshape(-1, 1)
    y = [1 if x is True else 0 for x in bool_y]
    y = np.asarray(y)

    logit_no_int = LogisticRegression(C=1, fit_intercept=False)
    logit_no_int.fit(X, y)
    logit = LogisticRegression(C=1, fit_intercept=True)
    logit.fit(X, y)

    m_no_int = logit_no_int.coef_[0, 0]
    m = logit.coef_[0, 0]
    b = logit.intercept_

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, .85, .85])
    logit_curve_no_int = lambda x: 1/(1 + math.e**(-(m_no_int*x)))
    logit_curve = lambda x: 1/(1 + math.e**(-(m*x+b)))
    x_vals = np.linspace(0, 600, 100)
    y_vals_no_int = logit_curve_no_int(x_vals)
    y_vals = logit_curve(x_vals)

    ax.plot(x_vals, y_vals_no_int, c='b', label='Intercept Removed')
    ax.plot(x_vals, y_vals, c='r', label='With Intercept')

    ax.scatter(X, y, c='black', s=2)
    ax.set_xlabel("Difference in Elo Rating")
    ax.set_ylabel("Probability of Higher Rank Winning")
    ax.legend(loc=5)
    # ax.axis('tight')
    ax.set_title("Logistic Function with & without Intercept")
    fig.savefig("../current_report_folder/figures/logistic_with_and_without_intercept.eps", format='eps', dpi=300)
    plt.show()


def main():
    df = pd.read_csv('data/current_data_files/aus_open_with_elos_full.csv')

    # drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['winner_elo'] = df['winner_elo'].replace(0.00000, np.nan)
    df['loser_elo'] = df['loser_elo'].replace(0.00000, np.nan)
    df = df[np.isfinite(df['winner_elo'])]
    df = df[np.isfinite(df['loser_elo'])]
    elo_tuples = df.apply(compare_elos, axis=1)


    plot_logit(elo_tuples)
    fill = 12


if __name__ == '__main__':
    main()
