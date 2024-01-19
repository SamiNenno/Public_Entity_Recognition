import os
from tqdm import tqdm
import pandas as pd
from glob import glob

class WeakSampler():
    def __init__(self, data_path:str, entities_per_class:int = 6000, window_size:int = 64, random_state:int = 2023) -> None:
        self.data_path = data_path
        self.entities_per_class = entities_per_class
        self.window_size = window_size
        self.random_state = random_state
        self.file_dict = dict()

    def entity_counter(self, df, file_id):
        df['Label'] = df['Label'].apply(lambda x: 'O' if 'PARTY' in x else x)
        doc_id = 0
        sentence_id = 0
        entity = ''
        added = False
        dict_list = list()
        for (idx, token, label) in tqdm(df.itertuples(index=True, name=None), desc=f"Compute Entity Stats for {file_id}", total=len(df), leave=False, position=1):
            if label == 'O':
                entity = ''
                try:
                    if not added:
                        dict_list.append(dct)
                        added = True
                except NameError:
                    pass
            elif label[0] == 'B':
                cat = label.split('-')[1]
                entity += token
                dct = dict(Entity=entity, Category=cat, IDX=idx, File_ID=file_id, Doc_ID=doc_id, Sentence_ID=sentence_id)
                added = False
            elif label[0] == 'I':
                dct["Entity"] += ' ' + token
            elif label == 'SENT_SEP':
                entity = ''
                sentence_id += 1
                try:
                    if not added:
                        dict_list.append(dct)
                        added = True
                except NameError:
                    pass
            else:
                entity = ''
                doc_id += 1
                sentence_id = 0
                try:
                    if not added:
                        dict_list.append(dct)
                        added = True
                except NameError:
                    pass
        return pd.DataFrame(dict_list)
    
    def count_entities(self):
        df_list = list()
        path_list = [(p, p.split('/')[-1].split('.')[0]) for p in glob(os.path.join(os.path.join(self.data_path, '03_PROCESSED'), '*.csv'))]
        for p, file_id in tqdm(path_list, total=len(path_list), position=0, desc=('Iterate over processed files')):
            df = pd.read_csv(p)
            df_list.append(self.entity_counter(df, file_id))
            self.file_dict[file_id] = df
        df = pd.concat(df_list, ignore_index=True)
        token_count = df['Entity'].apply(lambda x: len(x.split(' '))).to_list()
        df.insert(1, "Token_Count", token_count)
        return df

    def get_sample_size(self):
        dct = dict()
        for cat, subset in self.entities_population.groupby('Category'):
            entity_count = pd.DataFrame(subset['Entity'].value_counts()).reset_index()
            cat_size = min(self.entities_per_class, len(subset))
            sample_size = 0
            num_entities = 0
            for idx in range(1, entity_count['count'].max()):
                num_entities += len(entity_count[entity_count['count'] >= idx])
                if num_entities >= cat_size:
                    sample_size = idx
                    break
            
            dct[cat] = sample_size
        return dct

    def sample_without_replacement(self, group, n:int):
        if len(group) <= n:
            return group
        return group.sample(n, replace=False, random_state=self.random_state)

    def sample_data(self):
        sample_size = self.get_sample_size()
        df_list = list()
        for category, group in self.entities_population.groupby('Category'):
            sample = group.groupby('Entity', group_keys=False).apply(lambda x: self.sample_without_replacement(x, n=sample_size[category]))
            df_list.append(sample)
        df = pd.concat(df_list).sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        df = df.groupby("Category").sample(n=self.entities_per_class, random_state=self.random_state).reset_index(drop=True)
        return df
    
    def data_iterator(self, df, separator, progress_bar_position:int = 1):
        dict_list = list()
        for (idx, token, label) in tqdm(df[['Token', 'Label']].itertuples(index=True, name=None), desc=f'Separate {"Sentences" if separator == "SENT_SEP" else "Documents"}', total=len(df), leave=False, position=progress_bar_position):
            if token != separator:
                dict_list.append(dict(idx=idx, Token=token, Label=label))
            else:
                yield pd.DataFrame(dict_list)
                dict_list = list()
                
    def context_manager(self, file_id):
        df_list = list()
        sample = self.entities_sample[self.entities_sample['File_ID'] == file_id].drop_duplicates(subset=['Doc_ID','Sentence_ID'])
        for doc_id, doc in enumerate(self.data_iterator(self.file_dict[file_id], separator="DOC_SEP", progress_bar_position=1)):
            doc_copy = doc.copy()
            doc_copy = doc_copy[doc_copy['Token'] != "SENT_SEP"].reset_index(drop=True)
            doc_end = doc_copy['idx'].max()
            for sent_id, sent in enumerate(self.data_iterator(doc, separator="SENT_SEP", progress_bar_position=2)):
                if len(sample[(sample['Doc_ID'] == doc_id) & (sample['Sentence_ID'] == sent_id)]) > 0:
                    sent_start, sent_end = sent['idx'].to_list()[0], sent['idx'].to_list()[-1] + 1
                    window_start = max(0, sent_start - self.window_size)
                    window_end = min(doc_end, sent_end + self.window_size)
                    window_df = doc_copy.loc[window_start:window_end,['Token', 'Label']]
                    df_list.append(window_df)
        entity_count = self.entity_counter(pd.concat(df_list), file_id)
        return df_list, entity_count
    
    def make_dataset(self):
        count_frames = list()
        example_frames = list()
        for file_id in tqdm(self.file_dict.keys(), total=len(self.file_dict), desc="Make Dataset", position=0, leave=False):
            example_list, entity_count = self.context_manager(file_id=file_id)
            count_frames.append(entity_count)
            example_frames += example_list
        count_frame = pd.concat(count_frames, ignore_index=True)
        return count_frame, example_frames
    
    def save(self, df, name):
        df.to_csv(os.path.join(os.path.join(self.data_path, "05_STATS"), f"{name}.csv"), index=False)
    
    def save_dataset(self):
        path = os.path.join(self.data_path, "03_PROCESSED/WeakSample")
        for idx, data in enumerate(self.dataset):
            data.to_csv(os.path.join(path, f"example_{idx}.csv"), index=False)
        
    def main(self):
        self.entities_population = self.count_entities()
        self.save(self.entities_population, "Entity_Population")
        self.entities_sample = self.sample_data()
        self.save(self.entities_sample, "Entity_Target_Sample")
        self.entities_actual_sample, self.dataset = self.make_dataset()
        self.save(self.entities_actual_sample, "Entity_Actual_Sample")
        self.save_dataset()
        
    
if __name__ == '__main__':
    weak_sampler = WeakSampler(data_path="/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA")
    weak_sampler.main()
