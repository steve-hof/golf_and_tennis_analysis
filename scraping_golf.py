#!/usr/bin/env python3

"""
A script to scrape golf statistics for STATS 498

by Steve Hof May 18, 2018
"""

import requests
from bs4 import BeautifulSoup


url = "https://www.pgatour.com/tournaments/masters-tournament/field.html"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
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
inside_field_container = list(field_container.descendants)
players_list = field_container.find('div', class_="players-list players-list-main clearfix clear")
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
# if __name__ == '__main__':
#     main()


