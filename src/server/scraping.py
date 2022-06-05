import os
import time
import certifi
import random
import urllib
import urllib3
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

os.chdir('/home/umair/PycharmProjects/')

http = urllib3.PoolManager(num_pools=100, cert_reqs='CERT_REQUIRED',
                           ca_certs=certifi.where())

uas = ['Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:59.0) Gecko/20100101 Firefox/59.0',
       'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
       'Googlebot-News',
       'Googlebot-Image/1.0',
       'AdsBot-Google (+http://www.google.com/adsbot.html)',
       'Mediapartners-Google',
       'APIs-Google (+https://developers.google.com/webmasters/APIs-Google.html)',
       'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0',
       'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_4) AppleWebKit/600.7.12 (KHTML, like Gecko) Version/8.0.7 Safari/600.7.12']

time_ind = list()

abs_urllink = 'ftp://wits.cs.waikato.ac.nz/auckland/8/'


def scrape_wand():
    file_handle = open('/home/umair/Downloads/files.txt', "r")

    lines = file_handle.read().split('\n')

    names = lines[::2]

    for i in range(0, len(names)):
        file_name = names[i][5:]
        finalurl = abs_urllink + file_name
        urllib.request.urlretrieve(finalurl, file_name)
        print(finalurl)
        print(file_name)


def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


def scrape_by_days():
    all_bytes = np.array([])

    dates = ['0402']
    mins = ['00', '15', '30', '45', '60']

    for date in dates:
        for hour in range(24):
            str_hour = str(hour)

            if len(str_hour) == 1:
                str_hour = '0' + str_hour

            for minute in mins:
                t_stamp = '2009' + date + str_hour + minute
                finalurl = abs_urllink + date + str_hour + minute + '.html'

                user_agent = {'User-agent': random.choice(uas)}
                response = http.request('GET', finalurl, headers=user_agent)

                if response.status == 200:
                    soup = BeautifulSoup(response.data, 'html.parser')

                    table = soup.find_all("pre")[1]
                    tab_values = table.get_text().strip().splitlines()
                    total_bytes = find_between(tab_values[2], "(100.00%)", "(100.00%)").strip()
                    all_bytes = np.append(all_bytes, int(total_bytes))
                    time_ind.append(pd.to_datetime(t_stamp))

                    print(finalurl)
                    print(total_bytes)
                    time.sleep(5)
                else:
                    print("Could not receive a response!\n")


def scrape_by_years():
    all_bytes = np.array([])
    data_frame = pd.DataFrame()

    proto_names = ['total', 'ip', 'tcp', 'udp', 'tcp6', 'udp6', 'ip6']

    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    dates = ['01', '02', '03', '04', '05', '06', '07', '08', '09',
             '10', '11', '12', '13', '14', '15', '16', '17', '18',
             '19', '20', '21', '22', '23', '24', '25', '26', '27',
             '28', '29', '30', '31']

    for month in months:
        for date in dates:

            t_stamp = '2006' + month + date + '1400'
            finalurl = abs_urllink + month + date + '1400.html'

            user_agent = {'User-agent': random.choice(uas)}

            response = http.request('GET', finalurl, headers=user_agent)

            if response.status == 200:

                soup = BeautifulSoup(response.data, 'html.parser')

                try:
                    table = soup.find_all("pre")[1]

                except IndexError:
                    table = soup.find_all("pre")[0]

                tab_values = table.get_text().strip().splitlines()

                ctr = 0
                proto_bytes = np.zeros(7)
                temp_ip6 = np.array([])

                for i in range(2, len(tab_values)):
                    for j in range(ctr, len(proto_names) - 1):

                        if tab_values[i].lstrip().split(' ')[0] == proto_names[j]:
                            proto_bytes[j] = int(find_between(tab_values[i], "%)", "(").strip())
                            ctr += 1

                    if tab_values[i].lstrip().split(' ')[0] == proto_names[6]:
                        temp_ip6 = np.append(temp_ip6,
                                             int(find_between(tab_values[i], "%)", "(").strip()))

                proto_bytes[6] = temp_ip6.max()

                dd = pd.DataFrame(np.asmatrix(proto_bytes),
                                  index=[pd.to_datetime(t_stamp)], columns=proto_names)

                data_frame = data_frame.append(dd)

                all_bytes = np.append(all_bytes, int(proto_bytes[0]))
                time_ind.append(pd.to_datetime(t_stamp))

                print(finalurl)
                print(int(proto_bytes[0]))
                print()
                time.sleep(5)
            else:
                print("Could not receive a response!\n")

            # time_series = pd.Series(all_bytes, index=time_ind)

    """
    pd.DataFrame.from_csv('/home/umair/PycharmProjects/files/web_scraping/yearly/2006_total.csv')

    pd.Series.from_csv(('/home/umair/PycharmProjects/files/web_scraping/yearly/2006.csv'))

    data_frame.to_csv('/home/umair/PycharmProjects/files/web_scraping/yearly/2006_total.csv')

    time_series.to_csv('/home/umair/PycharmProjects/files/web_scraping/yearly/2006.csv', header=None)
    """
