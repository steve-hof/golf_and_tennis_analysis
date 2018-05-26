#!/usr/bin/env python3

"""
A script to scrape golf statistics for the Masters Tournament

by Steve Hof May 18, 2018
"""

import sys
import re
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEnginePage
import requests
from bs4 import BeautifulSoup


# Use PyQt5 module to pretend we're a browser
# That way we can see the javascript code we want to see
class Client(QWebEnginePage):
    def __init__(self, url):
        self.app = QApplication(sys.argv)
        QWebEnginePage.__init__(self)
        self.html = ''
        self.loadFinished.connect(self._on_load_finished)
        self.load(QUrl(url))
        self.app.exec_()

    def _on_load_finished(self):
        self.html = self.toHtml(self.Callable)
        print('Load finished')

    def Callable(self, html_str):
        self.html = html_str
        self.app.quit()


def get_player_urls(tournament_url):
    client_response = Client(tournament_url)
    soup = BeautifulSoup(client_response.html, 'html.parser')

    # Work our way through the web of nested elements
    body = soup.body
    div = body.find('div', class_="wrap alternative-font overlay-default")
    container = div.find('div', class_="container")
    page_container = container.find('div', class_="page-container")
    parsys = page_container.find('div', class_="parsys mainParsys")
    field_section = parsys.find('div', class_="field section")
    field_body = field_section.find('div', class_="field-body")
    ul = field_body.find('ul', class_="ul-inline field-body-cols")
    li = ul.find('li', class_="column-left")
    field_module = li.find('div', class_="field-module")
    field_container = field_module.find('div', class_="field-container")
    # players_list = field_container.find('div', class_="players-list players-list-main clearfix clear")

    # list of <img> tags contain all player ids
    player_img_list = field_container.find_all('img', class_="player-img")

    # go through each one and extract player-ids
    id_list = []
    for player in player_img_list:
        player_id = player['id']
        match = re.search('player-(\d+)', player_id)
        if match:
            id_list.append(match.group(1))
        else:
            print("no player id")

    # go through and extract player names
    player_name_list = []
    player_span_list = field_container.find_all('span', class_="player-name")
    for name in player_span_list:
        p_name = str(name.string).lower()

        switched = '-'.join(reversed(p_name.split(', ')))
        player_name_list.append(switched)

    # Dictionary mapping player names to id's will probably come in handy
    player_id_dict = dict(zip(player_name_list, id_list))

    # Create list of urls for all players in tournament
    url_list = []
    base_url = "https://www.pgatour.com/players/player."
    for name, id in player_id_dict.items():
        player_string = "" + id + "." + name + ".html"
        url = base_url + player_string
        url_list.append(url)

    return url_list


def main():
    masters_url = "https://www.pgatour.com/tournaments/masters-tournament/field.html"
    pebble_url = "https://www.pgatour.com/tournaments/at-t-pebble-beach-pro-am/field.html"
    masters_players_urls = get_player_urls(masters_url)
    pebble_players_urls = get_player_urls(pebble_url)



if __name__ == '__main__':
    main()
