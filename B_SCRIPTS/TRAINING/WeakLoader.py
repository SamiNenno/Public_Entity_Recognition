import os
import sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from datasets import Dataset, DatasetDict
from sklearn.model_selection import KFold
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'UTILS'))
from Path_finder import DATA_PATH
from DataUtils import load_label_dicts

class DataLoader():
    def __init__(self, data_path:str, num_folds:int = 5, sample:str="v2", random_state:int = 2023) -> None:
        self.id2label, self.label2id = load_label_dicts()
        self.data_path = data_path
        sample_dict = {"v1":"03_PROCESSED/WeakSample1/*.csv",
                       "v2":"03_PROCESSED/WeakSample2/*.csv",
                       "v3":"03_PROCESSED/WeakSample3/*.csv",
                       "v4":"03_PROCESSED/WeakSample4/*.csv",
                       "v5":"03_PROCESSED/WeakSample5/*.csv",
                       "confident":"03_PROCESSED/ConfidentSample_v2/*.csv"}
        self.path_list = [path for path in glob(os.path.join(self.data_path, sample_dict[sample.lower()]))] 
        self.token_name = "tokens"
        self.label_name = "tags"
        self.num_folds = num_folds
        self.random_state = random_state
        
    def frame_iterator(self, progressbar:bool):
        counter = 0#! REMOVE
        if progressbar:
            for path in tqdm(self.path_list, total=len(self.path_list), desc="Load Example Data Frames", leave=False):
                df = pd.read_csv(path)
                df = df[~df['Token'].isin(['\n', '\t', ' '])].reset_index(drop=True)
                df['Token'] = df['Token'].astype(str)
                yield df
                #counter += 1 #! REMOVE
                #if counter == 100:#! REMOVE
                #    break
        else:
            for path in self.path_list:
                df = pd.read_csv(path)
                df = df[~df['Token'].isin(['\n', '\t', ' '])].reset_index(drop=True)
                df['Token'] = df['Token'].astype(str)
                yield df
                
    def make_vanilla_dataset(self, progressbar:bool):
        dict_list = list()
        for df in self.frame_iterator(progressbar=progressbar):
            df['Label'] = df['Label'].apply(lambda x: self.label2id[x.replace(" ", "")])
            dict_list.append({
                self.token_name : df['Token'].to_list(),
                self.label_name : df['Label'].to_list()
            })
        df = pd.DataFrame(dict_list)
        return Dataset.from_pandas(df)
    
    def init_kfold(self, dataset, progressbar:bool = True):
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.random_state)
        splits = kf.split(np.zeros(dataset.num_rows))
        if progressbar:
            for train_idxs, val_idxs in tqdm(splits, total=self.num_folds, desc=f"{self.num_folds}-fold CV", leave=False):
                yield DatasetDict({
                    'train':dataset.select(train_idxs),
                    'validation':dataset.select(val_idxs),
                })
        else:
            for train_idxs, val_idxs in splits:
                yield DatasetDict({
                    'train':dataset.select(train_idxs),
                    'validation':dataset.select(val_idxs),
                })
                
    def vanilla_cross_validation(self, progressbar:bool):
        dataset = self.make_vanilla_dataset(progressbar)
        cv = self.init_kfold(dataset, progressbar)
        for fold in cv:
            yield fold
            
class EntityShuffler():
    def __init__(self, data_path:str, name_combinations:int = 4) -> None:
        self.data_path = data_path
        self.id2label, self.label2id = load_label_dicts()
        self.name_combinations = name_combinations
        self.token_name = "tokens"
        self.label_name = "tags"
            
    def load_names(self):
        behoerde = pd.read_csv(os.path.join(self.data_path, "01_UTIL/Shuffle/Behoerde.csv"))
        medien = pd.read_csv(os.path.join(self.data_path, "01_UTIL/Shuffle/Medien.csv"))
        partei = pd.read_csv(os.path.join(self.data_path, "01_UTIL/Shuffle/Partei.csv"))
        first_names = pd.read_csv(os.path.join(self.data_path, "01_UTIL/Shuffle/First_Names.csv"))
        last_names = pd.read_csv(os.path.join(self.data_path, "01_UTIL/Shuffle/Last_Names.csv"))
        politiker = pd.read_csv(os.path.join(self.data_path, "01_UTIL/Shuffle/Politiker.csv"))
        pol_names = list()
        for fn, seed in zip(first_names.itertuples(index=False, name=None),self.get_seeds(len(first_names), self.random_state)):
            surnames = last_names.sample(n = self.name_combinations, random_state=seed)["last_name"].to_list()
            for idx in range(self.name_combinations):
                pol_names.append(fn[0] + ' ' + surnames[idx])
        politiker = pd.concat([politiker, pd.DataFrame(dict(POLITIKER=pol_names))])
                
        journ_names = list()        
        for fn, seed in zip(first_names.itertuples(index=False, name=None),self.get_seeds(len(first_names), self.random_state + 1)):
            surnames = last_names.sample(n = self.name_combinations, random_state=seed)["last_name"].to_list()
            for idx in range(self.name_combinations):
                journ_names.append(fn[0] + ' ' + surnames[idx])
        journalisten = pd.DataFrame(dict(JOURNALIST=journ_names))
        self.new_name_dct = {"BEHÖRDE":behoerde, "MEDIEN":medien, "PARTEI":partei, "POLITIKER":politiker, "JOURNALIST":journalisten}
    
    def get_seeds(self, length, random_state):
        idxs = list(range(length))
        random.seed(random_state)
        random.shuffle(idxs)
        return idxs

    def sample_n_entities(self, df, col_name, n, random_state):
        if n == len(df):
            df = df.sample(frac=1, replace=False, random_state=random_state)
        elif n > len(df):
            df = df.sample(n=n, replace=True, random_state=random_state)
        else:
            df = df.sample(n=n, replace=False, random_state=random_state)
        return df[col_name].to_list()
    
    def inspect_frame(self, df, cnt_dct):
        ent_dct = {"BEHÖRDE":[], "MEDIEN":[], "PARTEI":[], "POLITIKER":[], "JOURNALIST":[]}
        ent_idx = []
        entity = ''
        for (idx, token, label) in df.itertuples(index=True, name=None):
            if label == "O":
                if entity != '':
                    ent_dct[entity].append(ent_idx)
                    cnt_dct[entity] += 1
                    entity = ''
                    ent_idx = []
            elif label[0] == "B":
                ent_idx = [idx]
                entity = label[2:]
            else:
                ent_idx.append(idx)
        return ent_dct, cnt_dct

    def locate_entities(self, frame_iterator):
        cnt_dct = {"BEHÖRDE":0, "MEDIEN":0, "PARTEI":0, "POLITIKER":0, "JOURNALIST":0}
        ent_dct_list = list()
        for frame in frame_iterator:
            frame["Label"] = frame["Label"].apply(lambda x: x.replace(" ", ""))
            ent_dct, cnt_dct = self.inspect_frame(frame, cnt_dct)
            ent_dct_list.append(ent_dct)
        return cnt_dct, ent_dct_list
    
    def sample_new_entities(self, frame_iterator):
        cnt_dct, ent_dct_list = self.locate_entities(frame_iterator)
        new_entities = dict()
        seeds = self.get_seeds(len(cnt_dct), random_state=self.random_state)
        for idx, (cat, count) in enumerate(cnt_dct.items()):
            new_entities[cat] = self.sample_n_entities(self.new_name_dct[cat], col_name=cat, n=count, random_state=seeds[idx])
        return cnt_dct, ent_dct_list, new_entities
    
    def make_substitution_df(self, substitution, cat):
        substitution = substitution.split(' ')
        if len(substitution) == 1:
            return pd.DataFrame(dict(Token=substitution, Label=[f"B-{cat}"]))
        else:
            dict_list = [dict(Token=substitution[0], Label=f"B-{cat}")]
            for sub in substitution[1:]:
                dict_list.append(dict(Token=sub, Label=f"I-{cat}"))
            return pd.DataFrame(dict_list)
    
    def remove_and_insert_rows(self, df, remove_indices, new_rows_df):        
        top_idx, bottom_idx = remove_indices[0] - df.index[0], remove_indices[-1] - df.index[0]
        top = df.iloc[:top_idx,:].copy()
        sub = pd.concat([top, new_rows_df])
        bottom = df.iloc[bottom_idx+1:,:].copy()        
        return sub, bottom
    
    def substitute_entities(self, df, ent_dct, cnt_dct, new_entities):
        df["Label"] = df["Label"].apply(lambda x: x.replace(" ", ""))
        df_list = list()
        counter = {"BEHÖRDE":0, "MEDIEN":0, "PARTEI":0, "POLITIKER":0, "JOURNALIST":0}
        for cat in df[df['Label'].str.contains('B-')]['Label'].apply(lambda x: x.split('-')[-1]).to_list():
            if counter[cat] < len(ent_dct[cat]):
                idx = ent_dct[cat][counter[cat]]
            else:
                continue
            if len(idx) == 0:
                continue
            else:
                substitution = new_entities[cat][cnt_dct[cat]-1]
                substitution = self.make_substitution_df(substitution, cat)
                sub, df = self.remove_and_insert_rows(df=df, remove_indices=idx, new_rows_df=substitution)
                df_list.append(sub)
                cnt_dct[cat] -= 1
                counter[cat] += 1
        df_list.append(df)
        return pd.concat(df_list, ignore_index=True), cnt_dct
    
    def shuffled_frame_iterator(self, frame_iterator1, frame_iterator2, random_state:int = 2023):
        self.random_state = random_state
        self.load_names()
        cnt_dct, ent_dct_list, new_entities = self.sample_new_entities(frame_iterator=frame_iterator1)
        for idx, frame in enumerate(frame_iterator2):
            df, cnt_dct = self.substitute_entities(df=frame, ent_dct=ent_dct_list[idx], cnt_dct=cnt_dct, new_entities=new_entities)
            yield df
    
    def make_shuffled_dataset(self, shuffled_frame_iter, label2id, transform_labels:bool = True):
        dict_list = list()
        for df in shuffled_frame_iter:
            if transform_labels:
                df['Label'] = df['Label'].apply(lambda x: label2id[x])
            dict_list.append({
                self.token_name : df['Token'].to_list(),
                self.label_name : df['Label'].to_list()
            })
        df = pd.DataFrame(dict_list)
        return Dataset.from_pandas(df)

class WeakLoader():
    def __init__(self, data_path:str, num_epochs:int = 10, num_folds:int = 5, sample:str = "v1", random_state:int = 2023, name_combinations:int = 4) -> None:
        self.dataloader = DataLoader(data_path=data_path, num_folds=num_folds, sample=sample, random_state=random_state)
        self.shuffler = EntityShuffler(data_path=data_path, name_combinations=name_combinations)
        self.cv_seeds = self.shuffler.get_seeds(length=num_folds, random_state=random_state)
        self.epoch_seeds = self.shuffler.get_seeds(length=num_epochs, random_state=random_state + 1)
        self.crossvalidation = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
        
    def get_label2id(self):
        return self.dataloader.label2id
    
    def get_id2label(self):
        return self.dataloader.id2label
    
    def get_vanilla_dataset(self, progressbar:bool = True):
        return self.dataloader.make_vanilla_dataset(progressbar=progressbar)
        
    def get_vanilla_crossvalidation(self, progressbar:bool = True):
        return self.dataloader.vanilla_cross_validation(progressbar=progressbar)
    
    def get_shuffled_frame_iterator(self, progressbar:bool = True, random_state:int = 2023):
        return self.shuffler.shuffled_frame_iterator(
                                        frame_iterator1=self.dataloader.frame_iterator(progressbar), 
                                        frame_iterator2=self.dataloader.frame_iterator(progressbar), 
                                        random_state=random_state)
    
    def get_folded_shuffled_frame_iterators(self):
        for random_state in self.cv_seeds:
            yield self.get_shuffled_frame_iterator(random_state=random_state)
            
    def get_shuffled_dataset(self, shuffled_frame_iter = None, random_state:int = 2023, transform_labels:bool = True):
        if shuffled_frame_iter is None:
            shuffled_frame_iter = self.get_shuffled_frame_iterator(random_state=random_state)
        return self.shuffler.make_shuffled_dataset(shuffled_frame_iter=shuffled_frame_iter, label2id=self.get_label2id(), transform_labels=transform_labels)
    
    def get_epoch_shuffled_dataset(self, shuffled_frame_iter = None, train_idxs = None):
        transform_labels = True
        for random_state in self.epoch_seeds:
            #print(random_state)
            if shuffled_frame_iter is None:
                shuffled_frame_iter = self.get_shuffled_frame_iterator(random_state=random_state)
                #print(shuffled_frame_iter)
            #ToDo hier muss ich den shuffled_frame_iter nehmen und ihm neue token zuweisen.
            #shuffled_frame_iter = self.shuffler.shuffled_frame_iterator(frame_iterator1=shuffled_frame_iter, 
            #                                                           frame_iterator2=shuffled_frame_iter, 
            #                                                            random_state=random_state)
            shuffled_dataset = self.get_shuffled_dataset(shuffled_frame_iter=shuffled_frame_iter, random_state=random_state, transform_labels=True)
            transform_labels = False
            train_idxs = list(len(shuffled_dataset)) if train_idxs is None else train_idxs
            yield shuffled_dataset.select(train_idxs)
            shuffled_frame_iter = None
            
    def get_shuffled_crossvalidation(self):
        #ToDo Each epoch new shuffled dataset (not only each fold)
        vanilla_dataset = self.get_vanilla_dataset()
        splits = self.crossvalidation.split(np.zeros(vanilla_dataset.num_rows))
        for shuffled_frame_iter, (train_idxs, val_idxs) in zip(self.get_folded_shuffled_frame_iterators(), splits):
            iter_copy = [copy for copy in shuffled_frame_iter]
            shuffled_dataset = self.get_epoch_shuffled_dataset(shuffled_frame_iter=None, train_idxs=train_idxs)
            #shuffled_dataset = self.get_epoch_shuffled_dataset(shuffled_frame_iter=shuffled_frame_iter, train_idxs=train_idxs)
            yield DatasetDict({
                    'train':shuffled_dataset,
                    'validation':vanilla_dataset.select(val_idxs),
                })
    
if __name__ == '__main__':
    random_state = 2023
    weak_loader = WeakLoader(data_path=DATA_PATH, random_state=random_state)
    #print(weak_loader.get_vanilla_dataset())
    #vanilla_cv = weak_loader.get_vanilla_crossvalidation()
    #print(next(iter(vanilla_cv)))
    #print(weak_loader.get_shuffled_dataset())
    shuffled_cv = weak_loader.get_shuffled_crossvalidation()
    trainset = next(iter(shuffled_cv))
    for idx, df in enumerate(trainset["train"]):
        df.to_pandas().to_csv(f"/home/sami/POLITICAL_ENTITY_RECOGNITION/TestShuffle{idx}.csv", index=False)
        tokens, tags = df["tokens"][0], df["tags"][0]
        for token, tag in zip(tokens, tags):
            print(f"{token.strip()} --> {tag}")
        print('---------\n\n')
    
