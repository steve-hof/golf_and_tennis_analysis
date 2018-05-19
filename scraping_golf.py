#!/usr/bin/env python3

"""
A script to scrape golf statistics for the Masters Tournament

by Steve Hof May 18, 2018
"""

import sys
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


def main():
    url = "https://www.pgatour.com/tournaments/masters-tournament/field.html"
    client_response = Client(url)
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
    players_list = field_container.find('div', class_="players-list players-list-main clearfix clear")
    player_img_list = field_container.find_all('img', class_="player-img")
    test = player_img_list[0]
    id = test['id']
    fill = 12


# def scrape_money_leaders_year(url):
#     # Downloads the money winners by year for the PGA tour
#     # using Requests
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, 'lxml')
#     money = []
#     for item in soup.findAll("td", "hidden-small")[4::3]:
#         money.append(item.text)
#     return money


# def get_cpi():
#     url = "http://www.minneapolisfed.org/community_education/teacher/calc/hist1913.cfm"
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, 'lxml')
#     info = []
#     for e in soup.findAll('td')[203:]:
#         info.append(e.text)
#     del info[2::3]
#     pairs = dict(zip(info[::2], info[1::2]))
#     return pairs
#
#
# def write_csv(money_winners_list, year):
#     Money_list = open("money_list_by_year.csv", 'ab')
#     money_winners_list.insert(0, year)
#     writer = csv.writer(Money_list, delimiter=',')
#     writer.writerow(money_winners_list)
#     Money_list.close()


# def main():
#     urls = get_links()
#     # for url_list in urls:
#     #     money_list = scrape_money_leaders_year(url_list[0])
#     #     corrected_ml = [int(i.replace(',', '')) * (229.6 / float(cpi_dict[str(url_list[1])])) for i in money_list]
#     # transpose_data()
#
#
# def transpose_data():
#     a = zip(*csv.reader(open("money_list_by_year.csv", "rb")))
#     csv.writer(open("money_list_by_year.csv", "wb")).writerows(a)
#
#
#
if __name__ == '__main__':
    main()
