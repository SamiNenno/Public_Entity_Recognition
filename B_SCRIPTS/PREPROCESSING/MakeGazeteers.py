import os
import sys
import json
import pandas as pd

def save_group(group, name, path):
    if name in ['Politiker', 'Journalist']:
        entities = [tuple(name.split(',')) for name in group['entity_name'].to_list()]
    else:
        entities = group['entity_name'].to_list()
    dct = {name:entities}
    with open(os.path.join(path, f"{name}.json"), 'w') as f:
        json.dump(dct, f, indent=4, ensure_ascii=False)

def save_kuerzel(kuerzel_path, path):
    with open(kuerzel_path, 'r', encoding='UTF-8') as f:
        beh_ku = dict(Behörde_Kurz = list(json.load(f).values()))
    with open(os.path.join(path, 'Behörde_Kurz.json'), 'w') as f:
        json.dump(beh_ku, f, indent=4,ensure_ascii=False)
        
def save_wiki_politiker(wiki_path, path, df):
    with open(wiki_path, 'r', encoding='UTF-8') as f:
        wiki = json.load(f)
    politiker = list(wiki['Bundestag'].keys()) + list(wiki['Landtag'].keys())
    politiker = dict(Politiker_Wiki=[pol.strip().split(' ') for pol in politiker if pol.strip().replace(' ', ',') not in df[df['label_name'] == 'Politiker']['entity_name'].to_list()])
    with open(os.path.join(path, 'Politiker_Wiki.json'), 'w') as f:
            json.dump(politiker, f, indent=4,ensure_ascii=False)

def make_gazeteers(data_path):
    kuerzel_path = os.path.join(os.path.join(data_path, '01_UTIL'), 'Behördenkürzel.json')
    wiki_path = os.path.join(os.path.join(data_path, '01_UTIL'), 'WIKIPEDIA_URLS.json')
    gazeteer_path = os.path.join(os.path.join(data_path, '01_UTIL'), 'Gazeteers')
    entity_frame_path = os.path.join(os.path.join(data_path, '01_UTIL'), 'Entity_Frame.csv')
    df = pd.read_csv(entity_frame_path)
    for name, group in df.groupby('label_name'):
        save_group(group, name, gazeteer_path)
    save_kuerzel(kuerzel_path, gazeteer_path)
    save_wiki_politiker(wiki_path, gazeteer_path, df)


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'UTILS'))
    from Path_finder import DATA_PATH
    make_gazeteers(DATA_PATH)
