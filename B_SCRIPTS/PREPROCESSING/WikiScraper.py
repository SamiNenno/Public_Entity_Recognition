import sys
import os
import re
import json
import requests
import pandas as pd
import wikipediaapi
from tqdm import tqdm
import concurrent.futures
from bs4 import BeautifulSoup
from collections import Counter


def get_bundestag(wahlperioden):
    dct = {}
    for wahlperiode in wahlperioden:
        df = pd.read_html(wahlperiode, extract_links='all')
        if '20._Wahlperiode' in wahlperiode:
            abgeordnete = {re.sub(r'\([^)]*\)', '', abgeordneter):f"https://de.wikipedia.org{url}" for abgeordneter, url in df[2].iloc[:,1].to_list()}
        else:
            abgeordnete = {re.sub(r'\([^)]*\)', '', abgeordneter):f"https://de.wikipedia.org{url}" for abgeordneter, url in df[1].iloc[:,0].to_list()}
        dct = dct | abgeordnete
    return dct

def get_landtag(wahlperioden):
    dct = {}
    for wahlperiode in wahlperioden:
        df = pd.read_html(wahlperiode, extract_links='all')
        if 'Bayerischen' in wahlperiode:
            abgeordnete = {re.sub(r'\([^)]*\)', '', abgeordneter):f"https://de.wikipedia.org{url}" for abgeordneter, url in df[1].iloc[:,0].to_list()}
        elif 'Bremisch' in wahlperiode:
            abgeordnete = {re.sub(r'\([^)]*\)', '', abgeordneter):f"https://de.wikipedia.org{url}" for abgeordneter, url in df[0].iloc[:,1].to_list()}
        elif 'Berlin' in wahlperiode:
            abgeordnete = {re.sub(r'\([^)]*\)', '', abgeordneter):f"https://de.wikipedia.org{url}" for abgeordneter, url in df[0].iloc[:,1].to_list()}
        elif 'Brandenburg' in wahlperiode or 'Hamburgischen' in wahlperiode:
            abgeordnete = {re.sub(r'\([^)]*\)', '', abgeordneter):f"https://de.wikipedia.org{url}" for abgeordneter, url in df[1].iloc[:,1].to_list()}
        elif 'Hessischen' in wahlperiode or 'Mecklenburg-Vorpommern' in wahlperiode:
            abgeordnete = {re.sub(r'\([^)]*\)', '', abgeordneter):f"https://de.wikipedia.org{url}" for abgeordneter, url in df[2].iloc[:,0].to_list()}
        elif 'Saarlandes' in wahlperiode or 'Nordrhein-Westfalen' in wahlperiode:
            abgeordnete = {re.sub(r'\([^)]*\)', '', abgeordneter):f"https://de.wikipedia.org{url}" for abgeordneter, url in df[3].iloc[:,1].to_list()}
        else:
            abgeordnete = {re.sub(r'\([^)]*\)', '', abgeordneter):f"https://de.wikipedia.org{url}" for abgeordneter, url in df[2].iloc[:,1].to_list()}
        dct = dct | abgeordnete
    return dct

def get_behoerden(behoerden_urls):
    def helper(df):
        kuerzels = [t[0] for t in df.iloc[:,0].to_list()]
        names = [t[0] for t in df.iloc[:,1].to_list()]
        urls = [t[1] for t in df.iloc[:,1].to_list()]
        return (
            {re.sub(r'\([^)]*\)|\[[^\]]*\]', '', name):f"https://de.wikipedia.org{url}" for name, url in zip(names, urls)},
            {re.sub(r'\([^)]*\)|\[[^\]]*\]', '', name):kuerzel for name, kuerzel in zip(names, kuerzels)}
        )
    df_list = pd.read_html(behoerden_urls, extract_links='all')
    kuerzel_dict = {}
    url_dict = {}
    for df in df_list:
        u_d, k_d = helper(df)
        kuerzel_dict = kuerzel_dict | k_d
        url_dict = url_dict | u_d
    return url_dict, kuerzel_dict

def get_verfassungsorgane(dct_organe):
    return dct_organe

def get_newspapers():
    df = pd.read_csv('https://raw.githubusercontent.com/Leibniz-HBI/DBoeS-data/main/data/1.csv')
    user_agent ='Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/117.0'
    wiki_wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language='de')
    newspaper_dct = {}
    for newspaper in tqdm(df.Name.to_list(), total=len(df), desc='Get newspaper wiki urls', leave=False):
        search_results = wiki_wiki.page(newspaper)
        if search_results.exists():
            newspaper_dct[newspaper] = search_results.fullurl
    return newspaper_dct

def get_wiki_urls(data_path):
    wiki_url_path = os.path.join(os.path.join(data_path, '01_UTIL'), 'wiki_source_urls.json')
    destination_path = os.path.join(data_path, '01_UTIL')
    with open(wiki_url_path, 'r', encoding='UTF-8') as f:
        wiki_urls = json.load(f)
    bundestag_dct = get_bundestag(wiki_urls['Bundestag'])
    landtag_dct = get_landtag(wiki_urls['Landtag'])
    url_dct, kuerzel_dct = get_behoerden(wiki_urls['Behörde'])
    organe_dct = get_verfassungsorgane(wiki_urls['Verfassungsorgane'])
    newspaper_dct = get_newspapers()
    urls = {
        'Bundestag': bundestag_dct,
        'Landtag': landtag_dct,
        'Behörden':url_dct,
        'Verfassungsorgane':organe_dct,
        'Zeitungen':newspaper_dct
    }
    with open(os.path.join(destination_path, "WIKIPEDIA_URLS.json"), 'w') as f:
        json.dump(urls, f, indent=4, ensure_ascii=False)
    with open(os.path.join(destination_path, "Behördenkürzel.json"), 'w') as f:
        json.dump(kuerzel_dct, f, indent=4, ensure_ascii=False)

def scrape(args):
    url, name = args
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        soup = soup.find('div', {'class':'mw-parser-output'})
        paras = []
        for paragraph in soup.find_all('p'):
            if not is_cat(str(paragraph.text)):
                paras.append(str(paragraph.text).strip())
        text = ' '.join(paras)
        text = re.sub(r"\[.*?\]+", '', text)
        return dict(Pagename = name, Url=url, Text=text)
    else:
        return None
def is_cat(input_string):
    char_counter = Counter(input_string)
    count = char_counter["|"]
    if count > 3:
        return True
    else:
        return False
def scrape_wikipedia(data_path, num_threads = 15):
    args = list()
    with open(os.path.join(os.path.join(data_path, '01_UTIL'), "WIKIPEDIA_URLS.json"), 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    for behoerde, dct in data.items():
        for name, url in dct.items():
            args.append((url, name))
    dict_list = list()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        with tqdm(total=len(args), desc='Scrape Wikipedia') as pbar:
            def process_item(items):
                result = scrape(items)
                pbar.update(1)
                if result is not None:
                    dict_list.append(result)
            futures = [executor.submit(process_item, item) for item in args]
            concurrent.futures.wait(futures)
    df = pd.DataFrame(dict_list)
    df.to_csv(os.path.join(os.path.join(data_path, '02_RAW'), 'WikiPages.csv'), index=False)
    
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'UTILS'))
    from Path_finder import DATA_PATH
    get_wiki_urls(DATA_PATH)
    scrape_wikipedia(DATA_PATH)