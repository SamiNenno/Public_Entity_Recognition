import os
import sys
import pandas as pd
from tqdm import tqdm
from urllib.error import HTTPError

def load_DBoeS(data_path:str):
    file_path = os.path.join(os.path.join(data_path, '01_UTIL'), 'DBoeS.csv')
    print(file_path)
    data_path = 'https://raw.githubusercontent.com/Leibniz-HBI/DBoeS-data/main/data/'
    df_list = list()
    for idx in tqdm(range(22), total=22, desc='Load DBoeS data'):
        path = f"{data_path}{idx}.csv"
        try:
            df_list.append(pd.read_csv(path))
        except HTTPError:
            pass
    df = pd.concat(df_list, ignore_index=True)
    df.to_csv(file_path, index=False)
    
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'UTILS'))
    from Path_finder import DATA_PATH
    load_DBoeS(DATA_PATH)