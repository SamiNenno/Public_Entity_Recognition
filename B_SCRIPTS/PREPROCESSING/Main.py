import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'UTILS'))
from Path_finder import DATA_PATH
from Load_DBoeS import load_DBoeS
from Clean_and_Extend_DBoeS import parse_and_load
from WikiScraper import get_wiki_urls, scrape_wikipedia
from MakeGazeteers import make_gazeteers
from AnnotateData import weak_annotation
from Make_Sample import WeakSampler

def main():
    #load_DBoeS(DATA_PATH)
    #parse_and_load(DATA_PATH)
    #get_wiki_urls(DATA_PATH)
    #scrape_wikipedia(DATA_PATH)
    #make_gazeteers(DATA_PATH)
    #weak_annotation(data_path=DATA_PATH, generative=False)
    WeakSampler(data_path=DATA_PATH).main()
    
if __name__ == '__main__':
    main()