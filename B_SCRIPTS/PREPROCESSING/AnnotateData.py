import os
import sys
import json
import random
import spacy
import pandas as pd
from tqdm import tqdm
from glob import glob
from skweak.base import CombinedAnnotator
from skweak.doclevel import DocumentHistoryAnnotator
from skweak.aggregation import MajorityVoter
from skweak.voting import SequentialMajorityVoter
from skweak.generative import HMM
from skweak.gazetteers import GazetteerAnnotator, Trie
from skweak.heuristics import FunctionAnnotator
                
def make_gazeteers(gazeteer_path, party_hyphen_path, combined):
    with open(party_hyphen_path, 'r') as f:
        hyph_dict = json.load(f)
    def party_hyphen(doc):
        parties = ['SPD-', 'FDP-', 'CDU-', 'CSU-', 'CDU/CSU-', 'Grünen-','Linken-', 'AfD-']
        for tok in doc:
            for party in parties:
                if party in tok.text:
                    checked = False
                    for politiker in hyph_dict["POLITIKER"]:
                        if politiker in tok.text:
                            checked = True
                            yield tok.i, tok.i, "POLITIKER"
                    if not checked:
                        for partei in hyph_dict["PARTEI"]:
                            if partei in tok.text:
                                checked = True
                                yield tok.i, tok.i, "PARTEI"
                    if not checked:
                        yield tok.i, tok.i, "PARTY_HYPHEN"
    labels = set()
    party_hyphen = FunctionAnnotator("party_hyphen", party_hyphen)
    combined.add_annotator(party_hyphen)
    labels.add("PARTY_HYPHEN")
    for json_path in glob(os.path.join(gazeteer_path, '*.json')):
        with open(json_path, 'r', encoding='UTF-8') as f:
            dct = json.load(f)
        name = json_path.split('/')[-1].split('.')[0]
        label = name.split('_')[0].upper()
        labels.add(label)
        if label in ['POLITIKER', 'JOURNALIST']:
            entities = [(n[0], n[1]) for n in dct[name] if len(n) == 2]
        else:
            entities = [tuple(entity.split(' ')) for entity in dct[name]]
        trie = Trie(entities)
        lf = GazetteerAnnotator(name, {label:trie})
        combined.add_annotator(lf)
    maj_voter = MajorityVoter("doclevel_voter", ["POLITIKER", "JOURNALIST"], initial_weights={"doc_history":0.0})
    doc_history= DocumentHistoryAnnotator("doc_history", "doclevel_voter",  ["POLITIKER", "JOURNALIST"])
    combined.add_annotator(maj_voter)
    combined.add_annotator(doc_history)
    return combined, list(labels)

def make_train_list(original_list, frac:float = .3):
    random.seed(2023)
    num_to_select = int(frac * len(original_list))
    selected_elements = random.sample(original_list, num_to_select)
    return list(selected_elements)

def load_data(path, entity_column):
    df = pd.read_csv(path)
    for idx, row in tqdm(df.iterrows(), desc="Load Data", total=len(df), leave=False):
        if isinstance(row[entity_column], str):
            yield row[entity_column]

def turn_to_doc(path, entity_column:str = 'Text', model_checkpoint:str = "de_core_news_md"):
    docs = list()
    nlp = spacy.load(model_checkpoint)
    for text in load_data(path=path, entity_column=entity_column):
        doc = nlp(text[:1000000]) # Max length allowed by spacy nlp
        docs.append(doc)
    return docs

def annotate(path_collection_:tuple, entity_column:str = 'Text', model_checkpoint:str = "de_core_news_md", generative:bool=True, frac:float = .3):
    gazeteer_path, party_hyphen_path, source_data_path, destination_path = path_collection_
    df_list = list()
    combined = CombinedAnnotator()
    combined, labels = make_gazeteers(gazeteer_path, party_hyphen_path,combined)
    print(f"Label documents with the following classes: {labels}")
    docs = turn_to_doc(path=source_data_path, entity_column=entity_column, model_checkpoint=model_checkpoint)    
    docs = list(combined.pipe(docs))
    if generative:
        voter = HMM("voter", labels=labels)
        voter.fit(make_train_list(docs, frac=frac))
    else:
        voter = SequentialMajorityVoter("voter", labels=labels)
    docs = list(voter.pipe(docs))

    for doc in docs:
        doc.ents = doc.spans['voter']
    for doc in docs:
        dict_list = list()
        for sentence in doc.sents:
            for token in sentence:
                label = token.ent_iob_ if token.ent_iob_ == 'O' else token.ent_iob_ + '-' + token.ent_type_
                dict_list.append(dict(Token=token.text, Label=label))
            dict_list.append(dict(Token="SENT_SEP", Label='SENT_SEP'))
        dict_list.append(dict(Token="DOC_SEP", Label='DOC_SEP'))
        df = pd.DataFrame(dict_list)
        if len(df['Label'].unique()) > 1:
            df_list.append(df)
    df = pd.concat(df_list)
    df.to_csv(destination_path, index=False)
    
def weak_annotation(data_path:str, generative:bool = False):
    gazeteer_path = os.path.join(os.path.join(data_path, '01_UTIL'), 'Gazeteers')
    party_hyphen_path = os.path.join(os.path.join(data_path, '01_UTIL'), 'Party_Hyphen.json')
    
    source_data_path = os.path.join(os.path.join(data_path, '02_RAW'), 'WikiPages.csv')
    destination_path = os.path.join(os.path.join(data_path, '03_PROCESSED'), 'WikiData.csv')

    path_collection = (gazeteer_path, party_hyphen_path, source_data_path, destination_path)
    annotate(path_collection_=path_collection,entity_column='Text', generative=False)
    
    for source_data_path in glob(os.path.join(os.path.join(data_path, '02_RAW'), 'Newspaper/*.csv')):
        destination_path = os.path.join(os.path.join(data_path, '03_PROCESSED'), source_data_path.split('/')[-1])
        path_collection = (gazeteer_path, party_hyphen_path, source_data_path, destination_path)
        annotate(path_collection_=path_collection, entity_column='text', generative=False)
    

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'UTILS'))
    from Path_finder import DATA_PATH
    weak_annotation(data_path=DATA_PATH, generative=False)
