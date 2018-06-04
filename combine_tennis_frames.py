#!/usr/bin/env python3

import pandas as pd


def main():
    match_df = pd.DataFrame.from_csv("data/current_data_files/aus_open_2018_match_results.csv")
    elo_df = pd.DataFrame.from_csv("data/current_data_files/aus_open_pre_elo.csv")
    match_df2 = pd.DataFrame.from_csv("data/current_data_files/aus_open_with_elos_2.csv")
    # names = list(elo_df.player)
    # elos = list(elo_df.elo_score)
    # elo_dict = dict(zip(names, elos))
    # match_df['winner_elo'] = match_df['winner_name'].map(elo_dict)
    # match_df['loser_elo'] = match_df['loser_name'].map(elo_dict)
    # cols = ['tourney_year_id', 'tourney_order', 'tourney_slug',
    #                'tourney_url_suffix', 'tourney_round_name', 'round_order',
    #                'match_order', 'winner_name', 'winner_elo', 'winner_player_id', 'winner_slug',
    #                'loser_name', 'loser_elo', 'loser_player_id', 'loser_slug', 'winner_seed',
    #                'loser_seed', 'match_score_tiebreaks', 'winner_sets_won',
    #                'loser_sets_won', 'winner_games_won', 'loser_games_won',
    #                'winner_tiebreaks_won', 'loser_tiebreaks_won', 'match_id',
    #                'match_stats_url_suffix']
    # match_df = match_df[cols]
    elo_dict_2 = dict(zip(match_df2['winner_name'], match_df2['winner_elo']))
    match_df2['loser_elo'] = match_df2['loser_name'].map(elo_dict_2)
    match_df2.to_csv("data/current_data_files/aus_open_with_elos3.csv")
    fill = 12


if __name__ == '__main__':
    main()