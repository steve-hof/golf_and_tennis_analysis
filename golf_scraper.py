#!/usr/bin/env python3

import numpy as np
import pandas as pd
import requests
import urllib.request
import json
import time
import argparse as ap
from decimal import Decimal

BASE_PATH = "data/current_data_files/golf/"

parser = ap.ArgumentParser()
parser.add_argument('tournament', nargs=1, help='name of tournament')
parser.add_argument('year', nargs=1, help='YEAR of tournament')
args = parser.parse_args()

TOURNAMENT = args.tournament[0]
YEAR = args.year[0]

tournament_id_dict = {
    'us_open': '026', 
    'masters': '014',
    'players_championship': '011'
}

players_champ_slug = "players_championship"
us_open_slug = "us_open"
masters_slug = "masters"

CSV_SAVE_SLUG = BASE_PATH + TOURNAMENT + "_" + YEAR + ".csv"
JUST_THE_CUT_SAVE_SLUG = BASE_PATH + TOURNAMENT + "_" + YEAR + "_made_cut" + ".csv"
PICKLE_SAVE_SLUG = BASE_PATH + TOURNAMENT + "_" + YEAR + "_pickle"


# Build dataframe containing pars and yards for each hole
def get_tournament_info(json_data):
    pars_list = []
    yards_1 = []
    yards_2 = []
    yards_3 = []
    yards_4 = []
    i = 1
    for rnd in json_data['trn']['rnds']:
        for course in rnd['courses']:
            for hole in course['holes']:
                if i == 1:
                    pars_list.append(hole['par'])
                    yards_1.append(hole['tee']['yards'])
                elif i == 2:
                    yards_2.append(hole['tee']['yards'])
                elif i == 3:
                    yards_3.append(hole['tee']['yards'])
                elif i == 4:
                    yards_4.append(hole['tee']['yards'])
        i += 1

    index = ['par', 'r1 yards', 'r2 yards', 'r3 yards', 'r4 yards']
    df = pd.DataFrame([pars_list, yards_1, yards_2, yards_3, yards_4], index=index, columns=list(range(1, 19)))

    return df


# Returns dict mapping player ids to names
# Also returns course_info_df
def get_player_field_ids(json_url):
    data = {}
    player_id_dict = {}
    with urllib.request.urlopen(json_url) as url:
        data = json.loads(url.read().decode())
    field_list = data['trn']['field']
    for player in field_list:
        ident = player['id']
        last_name = player['name']['last']
        first_name = player['name']['first']
        full_name = first_name + " " + last_name

        player_id_dict[ident] = full_name

    course_info_df = get_tournament_info(data)
    return player_id_dict, course_info_df


def build_player_df(json_url, player_id, player_name):
    data = {}
    with urllib.request.urlopen(json_url) as url:
        data = json.loads(url.read().decode())

    rounds_played = len(data['p']['rnds'])
    r1_dict = {}
    r2_dict = {}
    r3_dict = {}
    r4_dict = {}
    player_scores = []
    player_drive_distance = []
    player_num_putts = []
    total_rounds = 0
    full_player_info_list = []
    for a_round in data['p']['rnds']:
        total_rounds += 1
        round_num = a_round['n']
        # print(f"Round {round_num}")
        for hole in a_round['holes']:
            # print(f"Hole {hole['n']}:")
            score = hole['sc']
            # print(f"Score = {score}")
            player_scores.append(score)
            DUPLICATE = False
            for shot in hole['shots']:
                if shot['n'] == '1' and DUPLICATE is False:
                    DUPLICATE = True
                    drive_dist = shot['dist']
                    drive_dist = float(drive_dist) / 36
                    player_drive_distance.append(drive_dist)
                elif shot['n'] == score:
                    if not shot['putt']:
                        putts = '0'
                    else:
                        putts = shot['putt']
                    player_num_putts.append(putts)

        if round_num == '1':
            r1_dict['scores'] = player_scores
            r1_dict['drive_dist'] = player_drive_distance
            r1_dict['putts'] = player_num_putts
            full_player_info_list.append(r1_dict)
        elif round_num == '2':
            r2_dict['scores'] = player_scores
            r2_dict['drive_dist'] = player_drive_distance
            r2_dict['putts'] = player_num_putts
            full_player_info_list.append(r2_dict)
        elif round_num == '3':
            r3_dict['scores'] = player_scores
            r3_dict['drive_dist'] = player_drive_distance
            r3_dict['putts'] = player_num_putts
            full_player_info_list.append(r3_dict)
        elif round_num == '4':
            r4_dict['scores'] = player_scores
            r4_dict['drive_dist'] = player_drive_distance
            r4_dict['putts'] = player_num_putts
            full_player_info_list.append(r4_dict)
        player_scores = []
        player_drive_distance = []
        player_num_putts = []

        # print()
        print(f"scraping {player_name}")

    hole_columns = list(range(1, 19))
    player_name_index = [player_name, player_name, player_name]
    rounds_1 = ['1', '1', '1']
    rounds_2 = ['2', '2', '2']
    rounds_3 = ['3', '3', '3']
    rounds_4 = ['4', '4', '4']
    rounds_5 = ['5', '5', '5']
    desc = ['scores', 'drive_dist', 'putts']

    df_list = []
    for i, r in enumerate(full_player_info_list):
        if i + 1 == 1:
            hier_index = list(zip(player_name_index, rounds_1, desc))
            hier_index = pd.MultiIndex.from_tuples(hier_index)
            score_list = list(r.values())
            df_list.append(pd.DataFrame(score_list, index=hier_index, columns=hole_columns))
        if i + 1 == 2:
            hier_index = list(zip(player_name_index, rounds_2, desc))
            hier_index = pd.MultiIndex.from_tuples(hier_index)
            score_list = list(r.values())
            df_list.append(pd.DataFrame(score_list, index=hier_index, columns=hole_columns))
        if i + 1 == 3:
            hier_index = list(zip(player_name_index, rounds_3, desc))
            hier_index = pd.MultiIndex.from_tuples(hier_index)
            score_list = list(r.values())
            df_list.append(pd.DataFrame(score_list, index=hier_index, columns=hole_columns))
        if i + 1 == 4:
            hier_index = list(zip(player_name_index, rounds_4, desc))
            hier_index = pd.MultiIndex.from_tuples(hier_index)
            score_list = list(r.values())
            df_list.append(pd.DataFrame(score_list, index=hier_index, columns=hole_columns))

    df = df_list[0]
    for i in range(1, min(4, rounds_played)):
        df = df.append(df_list[i])

    time.sleep(.5)
    return df


def get_par_info(json_url):
    with urllib.request.urlopen(json_url) as url:
        data = json.loads(url.read().decode())

    pars_list = []
    yards_1 = []
    yards_2 = []
    yards_3 = []
    yards_4 = []
    i = 1
    for round in data['trn']['rnds']:
        for course in round['courses']:
            for hole in course['holes']:
                if i == 1:
                    pars_list.append(hole['par'])
                    yards_1.append(hole['tee']['yards'])
                elif i == 2:
                    yards_2.append(hole['tee']['yards'])
                elif i == 3:
                    yards_3.append(hole['tee']['yards'])
                elif i == 4:
                    yards_4.append(hole['tee']['yards'])
        i += 1

    return list(map(int, pars_list))


def golf_round(series):
    rounded = series.astype(float).round(0)
    return rounded


def filter_cut(full_df):
    par_row = full_df.loc[0]
    made_round_4_df = full_df[full_df['Round'] == 4]
    made_cut_player_list = made_round_4_df['Player'].unique().tolist()

    df = full_df[full_df['Player'].isin(made_cut_player_list)]

    df.loc[-1] = par_row
    df.index = df.index + 1
    df.sort_index(inplace=True)
    return df


def main():
    base_url_begin = "https://statdata.pgatour.com/r/"
    url = base_url_begin + tournament_id_dict[TOURNAMENT] + '/' + YEAR + '/setup.json'
    tournament_player_ids_dict, course_df = get_player_field_ids(url)

    for i, p_id in enumerate(tournament_player_ids_dict.keys()):
        if i == 0:
            player_name = tournament_player_ids_dict[p_id]
            player_json_url = base_url_begin + tournament_id_dict[TOURNAMENT] + '/' + YEAR + '/scorecards/' + \
                              p_id + '.json'
            base_df = build_player_df(player_json_url, p_id, player_name)
        else:
            player_name = tournament_player_ids_dict[p_id]
            player_json_url = base_url_begin + tournament_id_dict[TOURNAMENT] + '/' + YEAR + '/scorecards/' + \
                              p_id + '.json'
            new_df = build_player_df(player_json_url, p_id, player_name)
            base_df = base_df.append(new_df)

    base_df.to_csv('working_file.csv')
    base_df = pd.read_csv('working_file.csv')
    cols = ['Player', 'Round', 'Type', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9',
            'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16', 'H17', 'H18']
    base_df.columns = cols

    # base_df.to_csv(CSV_SAVE_SLUG)
    # base_df.to_pickle('working_pickle')

    # base_df = pd.read_pickle('working_pickle')

    # Get par info
    pars_list = ['Par', '0', 'Par']
    look = get_par_info(url)
    pars_list.extend(look)

    # add par to dataframe
    # base_df.loc[0] = pars_list  # adding a row
    # base_df.reset_index(drop=True, inplace=True)
    # final_df = base_df

    base_df.loc[-1] = pars_list
    base_df.index = base_df.index + 1
    base_df.sort_index(inplace=True)
    final_df = base_df

    # df.loc[-1] = ['45', 'Dean', 'male']  # adding a row
    # df.index = df.index + 1  # shifting index
    # df.sort_index(inplace=True)

    all_holes = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9',
                 'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16', 'H17', 'H18']
    final_df['Total'] = final_df[all_holes].sum(axis=1)
    final_df['Total'] = final_df['Total'].map(lambda num: round(num, 1))

    front_nine = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9']
    final_df['Out'] = final_df[front_nine].sum(axis=1)
    final_df['Out'] = final_df['Out'].map(lambda num: round(num, 1))

    back_nine = ['H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16', 'H17', 'H18']
    final_df['In'] = final_df[back_nine].sum(axis=1)
    final_df['In'] = final_df['In'].map(lambda num: round(num, 1))

    # Reorder columns
    cols = ['Player', 'Round', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9',
            'Out', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16', 'H17', 'H18', 'In', 'Total', 'Type']

    final_df = final_df[cols]
    round_cols = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9',
                  'Out', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16', 'H17', 'H18', 'In', 'Total']
    final_df[round_cols] = final_df[round_cols].apply(golf_round)
    final_df.to_csv(CSV_SAVE_SLUG)
    final_df.to_pickle(PICKLE_SAVE_SLUG)

    final_df = pd.read_pickle(PICKLE_SAVE_SLUG)

    # Create dataframe containing only golfers that made the cut
    weekend_df = filter_cut(final_df)
    weekend_df.to_csv(JUST_THE_CUT_SAVE_SLUG, index=False)


if __name__ == '__main__':
    main()
