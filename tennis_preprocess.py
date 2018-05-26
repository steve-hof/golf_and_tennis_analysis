#!/usr/bin/env python3

import pandas as pd

match_scores_csv_path = 'csv/2_match_scores/'

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

match_scores_df.to_csv('atp_tennis_match_results_2018_may17.csv')
fill = 12