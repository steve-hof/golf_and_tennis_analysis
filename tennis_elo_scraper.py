#!/usr/bin/env python3

import pandas as pd
import requests
from bs4 import BeautifulSoup

rank_list = []
player_list = []
elo_list = []
position_plus_minus_list = []
elo_plus_minus_list = []

pre_aus_open_2018_url = 'https://tenniseloranking.blogspot.com/2018/01/mens-elo-rankings-15012018.html'
page_request = requests.get(pre_aus_open_2018_url)
soup = BeautifulSoup(page_request.text, "lxml")
# find table in soup
table = soup.find_all('table')[1]

tbody = table.find('tbody')
trows = tbody.find_all('tr')

for row in trows:
    rank_list.append(row.find_all('td')[0].text)
    player_list.append(row.find_all('td')[1].text)
    elo_list.append(row.find_all('td')[2].text)
    position_plus_minus_list.append(row.find_all('td')[3].text)
    elo_plus_minus_list.append(row.find_all('td')[4].text)

elo_dict = {
    'rank': rank_list[1:],
    'player': player_list[1:],
    'elo_score': elo_list[1:],
    'pos_plus_min': position_plus_minus_list[1:],
    'elo_plus_min': elo_plus_minus_list[1:]
}
df = pd.DataFrame.from_dict(elo_dict)
cols = ['rank', 'player', 'elo_score', 'pos_plus_min', 'elo_plus_min']
df = df[cols]

df.to_csv("data/current_data_files/aus_open_pre_elo.csv")
print(df.to_string())
fill = 12