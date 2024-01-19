import os
import sys
import re
import json
import requests
from urllib.error import HTTPError
from tqdm import tqdm
import pandas as pd

GENITIV = {'Radio':'Radios', 
 'Anzeiger':'Anzeigers', 
 'Tageblatt':'Tageblatts', 
 'Tagblatt':'Tagblatts', 
 'Fernsehen':'Fernsehens',
 'Kurier':'Kuriers', 
 'Kreisblatt':'Kreisblatts',
 'Staatsministerium':'Staatsministeriums',
 'Ministerium':'Ministeriums',
 'Bundesministerium':'Bundesministeriums',
'Landtag':'Landtags',
'Bundesamts':'Bundesamts',
'Landeskriminalamt':'Landeskriminalamts'}

def assign_label_name(typ:int):
    if typ in [1, 3, 4, 5, 6, 7]:
    # Nach Bedarf können Medien auch noch weiter unterteilt werden.
    # Allerdings gibt es einige Typen mit sehr wenig Einheiten, hier müsste man subsumieren.
        return 'Medien'
    elif typ == 9:
        return 'Partei'
    elif typ == 15:
        return 'Behörde'
    elif typ == 20:
        return 'Journalist'
    elif typ == 21:
        return 'Politiker'
    else:
        raise KeyError('Dieser Typ ist nicht vergeben!')

def assign_label(label_name:str):
    dct = {name:label for label, name in enumerate(['Medien', 'Partei', 'Behörde', 'Journalist', 'Politiker'])}
    return dct[label_name]

def parse_data(df):
    df['label_name'] = df['Typ'].apply(lambda x: assign_label_name(x))
    df['label'] = df['label_name'].apply(lambda x: assign_label(x))
    df['entity_name'] = df['Name']
    return df[['entity_name', 'label_name', 'label']]

def clean_name(name:str):
    name = name.replace(' Dr. ', ' ')
    name = name.replace(' Prof. ', ' ')
    last, first = name.split(',')[:2]
    name = first.strip() + ',' + last.strip()
    return name

def extract_party(input_string):
    if '(' in input_string and ')' in input_string:
      pattern = r'\(([^)]*)\)'
      matches = re.findall(pattern, input_string)
      result_string = re.sub(pattern, '', input_string).strip()
      return result_string, matches[0]
    else:
      return input_string, None

def get_party_abbreviations(df):
    party_df = df[df['label_name'] == 'Partei'].copy()
    dict_list = list()
    for idx, party in tqdm(party_df.iterrows(), total=len(party_df), desc='Extract party abbreviations'):
        name, abbr = extract_party(party.entity_name)
        if abbr is None:
            continue
        else:
            dict_list.append(dict(entity_name = abbr, label_name = party.label_name, label = party.label))
    df = pd.concat([df, pd.DataFrame(dict_list)], ignore_index=True)
    return df

def add_genitiv_to_persons(df):
    temp = df[df['label_name'].isin(['Politiker', 'Journalist'])].copy()
    updated_rows = []
    for index, row in temp.iterrows():
        if row['entity_name'][-1] not in ['s', 'z', 'x', 'ß']:
            updated_entity_name = row['entity_name'] + 's'
            updated_row = {
                'entity_name': updated_entity_name,
                'label_name': row['label_name'],
                'label': row['label']
            }
            updated_rows.append(updated_row)
    updated_df = pd.DataFrame(updated_rows)
    return pd.concat([df, updated_df], ignore_index=True)

def replace_searchstring(search_string, text):
    pattern = r'\b{}\b(?![sS])'.format(re.escape(search_string.rstrip("s")))
    replaced_text = re.sub(pattern, search_string, text, flags=re.IGNORECASE)
    return replaced_text

def add_genitiv_to_orgs(df):
    temp = df[df['label_name'].isin(['Behörde', 'Medien'])].copy()
    updated_rows = []
    for index, row in temp.iterrows():
        for base, genitiv in GENITIV.items():
            if base in row['entity_name']:
                updated_entity_name = replace_searchstring(genitiv, row['entity_name'])
                if updated_entity_name == row['entity_name']:
                    continue
                updated_row = {
                'entity_name': updated_entity_name,
                'label_name': row['label_name'],
                'label': row['label']
                }
                updated_rows.append(updated_row)
                break
            else:
                continue
    updated_df = pd.DataFrame(updated_rows)
    return pd.concat([df, updated_df], ignore_index=True)

def clean_entity_names(df):
    df.loc[df['label_name'].isin(['Journalist', 'Politiker']),'entity_name'] = df.loc[df['label_name'].isin(['Journalist', 'Politiker']),'entity_name'].apply(lambda x: clean_name(x))
    df = get_party_abbreviations(df)
    return df.sort_values('label').reset_index(drop=True)

def remove_manual(df, remove_path):
    remove = pd.read_csv(remove_path)
    return df[~df.entity_name.isin(remove.entity_name.to_list())]

def extend_manual(df, extend_path):
    return pd.concat([df, pd.read_csv(extend_path)], ignore_index=True)

def parse_and_load(data_path:str):
    source_path = os.path.join(os.path.join(data_path, '01_UTIL'), 'DBoeS.csv')
    destination_path = os.path.join(os.path.join(data_path, '01_UTIL'), 'Entity_Frame.csv')
    extend_path = os.path.join(os.path.join(data_path, '01_UTIL'), 'Extend.csv')
    remove_path = os.path.join(os.path.join(data_path, '01_UTIL'), 'Remove.csv')
    df = pd.read_csv(source_path)
    df = parse_data(df)
    df = clean_entity_names(df)
    df = add_genitiv_to_persons(df)
    df = add_genitiv_to_orgs(df)
    df = remove_manual(df, remove_path)
    df = extend_manual(df, extend_path)
    df.sort_values('label').to_csv(destination_path, index=False)
    return df


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'UTILS'))
    from Path_finder import DATA_PATH
    df = parse_and_load(DATA_PATH)

