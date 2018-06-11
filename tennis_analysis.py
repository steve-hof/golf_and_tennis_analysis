#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats import logistic
from matplotlib.ticker import NullFormatter
from sklearn import linear_model
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
import collections


def compare_elos(row):  # winner_elo, loser_elo):
    # elo_diff = max(winner_elo, loser_elo) - min(winner_elo, loser_elo)
    # result = max(winner_elo, loser_elo) == winner_elo
    elo_diff = max(row['winner_elo'], row['loser_elo']) - min(row['winner_elo'], row['loser_elo'])
    result = max(row['winner_elo'], row['loser_elo']) == row['winner_elo']

    last = (elo_diff, result)
    return result


def plot_logit(source_list):
    fig, ax = plt.subplots(1, 1)
    X, bool_y = zip(*source_list)
    X = np.asarray(X)
    X = X.reshape(-1, 1)
    y = [1 if x is True else 0 for x in bool_y]
    y = np.asarray(y)

    clf = linear_model.LogisticRegression(C=1e2)  # C=1e5)
    clf.fit(X, y)
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.scatter(X, y, color='blue', zorder=20)
    # plt.scatter(X, clf.predict_proba(X)[:, 1])

    def model(x):
        return 1 / (1 + np.exp(-x))

    loss = model(X * clf.coef_ + clf.intercept_)
    plt.plot(X, loss, color='red', linewidth=3)
    plt.show()
    fill = 12


def main():
    df = pd.read_csv('data/current_data_files/aus_open_with_elos_full.csv')

    # drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.fillna(value=0, inplace=True)

    # give all un-ranked players and elo of one less than the lowest elo
    min_winner_elo = np.min(df['winner_elo'].values[np.nonzero(df['winner_elo'].values)])
    min_loser_elo = np.min(df['loser_elo'].values[np.nonzero(df['loser_elo'].values)])
    min_elo = min(min_winner_elo, min_loser_elo)

    df['winner_elo'] = df['winner_elo'].apply(lambda rank: min_elo - 1 if rank == 0 else rank)
    df['loser_elo'] = df['loser_elo'].apply(lambda rank: min_elo - 1 if rank == 0 else rank)
    df['elo_diff'] = df['winner_elo'] - df['loser_elo']

    check = df['elo_diff'] > 0
    num = collections.Counter(check)
    elo_tuples = df.apply(compare_elos, axis=1)


    # plot_logit(elo_tuples)
    fill = 12


if __name__ == '__main__':
    main()
