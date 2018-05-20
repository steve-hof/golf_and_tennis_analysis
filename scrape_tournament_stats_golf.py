#!/usr/bin/env python3
"""
Use player IDS from scrape_golf_players.py to gather stats for a particular tournament

by Steve Hof May 20, 2018
"""

import sys
import re
import pandas as pd
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


def get_tourney_stats(player_url):
    client_response = Client(player_url)
    soup = BeautifulSoup(client_response.html, 'html.parser')

    # Work our way through the web of nested elements
    body = soup.body
    wrap = body.find('div', class_="wrap")
    container = wrap.find('div', class_="container")
    clearfix_module_player = container.find('div', class_="clearfix module-player-navigation")
    tabbable = clearfix_module_player.find('div', class_="tabbable")
    tab_content = tabbable.find('div', class_="tab-content tab-content-players-overview")
    player_performance = tab_content.find('div', class_="performance player-performance")
    performance_section = player_performance.find('section', id="performance")
    tabbable_performance_section = performance_section.find('div', class_="tabbable player-section-performance")
    tab_content = tabbable_performance_section.find('div', class_="tab-content")
    tab_pane_active = tab_content.find('div', id="performanceTournament")
    player_perform_module = tab_pane_active.find('div', class_="player-performance-tournament-"
                                                               "module module hide-tour-year-filter")
    tour_filter_block = player_perform_module.find('div', class_="tour-year-filter-block")

    # trigger some necessary javascript
    tour_filter_container = tour_filter_block.find('div', class_="tour-year-filter-container")
    performance_head_clearfix = tour_filter_container.find('div', class_="performance-head clearfix")

    render = tour_filter_block.find('div', class_="render")

    # see = render.descendents
    # data_react = render.find_all('div')
    # career_table = data_react.find_all('div')[1]
    fill = 12


def main():
    # pgatour.com page scraping
    url = "https://www.pgatour.com/players/player.28237.rory-mcilroy.html"
    # pga_tour_response = Client(url)
    # tournament_stats = get_tourney_stats(url)

    # golfstats.com page scraping (way easier)
    years = ['2011', '2010']
    base_url = "https://www.golfstats.com/search/?yr="
    end_url = "&tournament=Masters&player=&tour=Majors&submit=go"
    for year in years:
        url = base_url + year + end_url
        the_response = Client(url)
        df = pd.read_html(the_response.html)[0]
        cols = ['Player', 'empty1', 'Placing', 'R1 Score', 'R2 Score', 'R3 Score',
                'R4 Score', 'R1 Placing', 'R2 Placing', 'R3 Placing', 'Vs Par',
                'Total Score', 'Money Earned', 'empty2', 'empty3']
        df.columns = cols
        df.drop(['empty1', 'empty2', 'empty3'], axis=1, inplace=True)
        save_path = 'masters_results_' + year + '.csv'
        df.to_csv(save_path, index=False)
    fill = 2


if __name__ == '__main__':
    main()

