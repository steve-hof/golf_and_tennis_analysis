#!/usr/bin/env python3

import pandas as pd

match_scores_csv_path = 'data/extra_tennis_csv/2_match_scores/'
base_save_path = 'data/current_data_files'

# Load 2018 Match Scores
full_path = match_scores_csv_path + 'match_scores_2018-2018.csv'
cols = ["tourney_year_id", "tourney_order", "tourney_slug", "tourney_url_suffix",
        "tourney_round_name", "round_order", "match_order", "winner_name",
        "winner_player_id", "winner_slug", "loser_name", "loser_player_id", "loser_slug",
        "winner_seed", "loser_seed", "match_score_tiebreaks", "winner_sets_won",
        "loser_sets_won", "winner_games_won", "loser_games_won", "winner_tiebreaks_won",
        "loser_tiebreaks_won", "match_id", "match_stats_url_suffix"]

match_scores_df = pd.read_csv(full_path, header=None)
match_scores_df.columns = cols

match_scores_df.to_csv('data/current_data_files/atp_tennis_match_results_2018.csv')

aus_open_df = match_scores_df[match_scores_df['tourney_slug'] == 'australian-open']
aus_open_df.reset_index(inplace=True)

aus_open_df = aus_open_df.drop(['index'], axis=1)
aus_open_save_path = base_save_path + 'aus_open_2018_match_results.csv'
aus_open_df.to_csv(aus_open_save_path)
print(aus_open_df.to_string())
fill = 12