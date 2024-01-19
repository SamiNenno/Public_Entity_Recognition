import os
import json
import torch
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from sklearn.model_selection import KFold
from BaseTrainer import BaseTrainer
from Explainer import DataLoaderExplainer
from transformers import AutoTokenizer, AutoModelForTokenClassification

class NoiseSampler():
    def __init__(self, sample_path:str, label2id_path:str, noise_percentage:float = 1.0, num_folds:int = 5, random_state:int = 2023):
        with open(label2id_path, "r") as f:
            self.label2id = json.load(f)
        self.path_list = glob(os.path.join(sample_path, "*.csv"))
        self.noise_percentage = noise_percentage
        self.random_state = random_state
        self.num_folds = num_folds
        self.new_label_dct = self.make_noisy_labels()
        self.token_name = "tokens"
        self.label_name = "tags"
        
    def make_noisy_labels(self):
        df = pd.concat([pd.read_csv(path) for path in tqdm(self.path_list, total=len(self.path_list), desc="Load Sample", leave=False)], ignore_index=True)
        count_df = pd.DataFrame(df[df["Label"] != "O"].value_counts("Token")).reset_index().rename(columns={0:"Count", "count":"Count"})
        target_sum = self.noise_percentage * count_df["Count"].sum()
        sampled_df = count_df.sample(frac=1, replace=False, random_state=self.random_state)
        current_sum = 0
        selected_rows = []
        for _, row in sampled_df.iterrows():
            current_sum += row["Count"]
            selected_rows.append(row)
            if current_sum >= target_sum:
                break
        final_sample_df = pd.DataFrame(selected_rows)
        np.random.seed(self.random_state)
        final_sample_df["Label"] = np.random.choice([label for label in df["Label"].unique() if label != "O"], size=len(final_sample_df), replace=True)
        new_label_dct = {}
        for idx, row in final_sample_df.iterrows():
            new_label_dct[row["Token"]] = row["Label"]
        return new_label_dct
    
    def assign_noisy_label(self, row):
        if row["Label"] == "O":
            return "O"
        noisy_label = self.new_label_dct.get(row["Token"])
        if noisy_label is None:
            return row["Label"]
        else:
            return noisy_label
        
    def noisify(self):
        for path in tqdm(self.path_list, total=len(self.path_list), desc=f"Make noisy data ({int(self.noise_percentage * 100)}%)", leave=False):
            df = pd.read_csv(path)
            df["Label"] = df.apply(lambda row: self.assign_noisy_label(row), axis=1)
            yield df
            
    def make_noisy_dataset(self):
        dict_list = list()
        for df in self.noisify():
            df['Label'] = df['Label'].apply(lambda x: self.label2id[x.replace(" ", "")])
            df = df[df["Token"].notna()]
            dict_list.append({
                self.token_name : df['Token'].to_list(),
                self.label_name : df['Label'].to_list()
            })
        df = pd.DataFrame(dict_list)
        return Dataset.from_pandas(df)
    
    def init_kfold(self, dataset):
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.random_state)
        splits = kf.split(np.zeros(dataset.num_rows))
        for train_idxs, val_idxs in tqdm(splits, total=self.num_folds, desc=f"{self.num_folds}-fold CV (noise: {int(self.noise_percentage * 100)}%)", leave=False):
            yield DatasetDict({
                'train':dataset.select(train_idxs),
                'validation':dataset.select(val_idxs),
            })
            
    def noisy_cross_validation(self):
        dataset = self.make_noisy_dataset()
        cv = self.init_kfold(dataset)
        for fold in cv:
            yield fold

class NoiseTrainer(BaseTrainer):
    def __init__(self, data_path:str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA/", label2id_path:str="/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA/04_TRAINING/02_PER/label2id.json",model_path: str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/NoisyModel", sample: str = "v1", num_folds: int = 5, random_state: int = 2023, train_batch_size: int = 8, eval_batch_size: int = 16, lr: float = 0.000005, num_train_epochs: int = 5, name_combinations: int = 4, noise_percentage:float = 1.0) -> None:
        super().__init__(data_path, model_path, sample, num_folds, random_state, train_batch_size, eval_batch_size, lr, num_train_epochs, name_combinations)
        self.noise_percentage = noise_percentage
        noise_sampler = NoiseSampler(sample_path=os.path.join(data_path, "03_PROCESSED/ExplainabilitySampleV2"), label2id_path=label2id_path, noise_percentage=noise_percentage, random_state=random_state)
        self.cv = noise_sampler.noisy_cross_validation()
        self.noise = str(int(noise_percentage * 100))
        self.model_path = model_path
        
    def noise_train_run(self, train_dataloader, eval_dataloader, size):
        score_list = list()
        model_vars = self.load_training_setup(model_checkpoint=self.get_model_checkpoint(size=size), train_dataloader=train_dataloader, eval_dataloader=eval_dataloader)
        step_size = len(model_vars['train_dataloader'])//10
        for epoch in range(model_vars["epochs"]):
            model_vars['model'].train()
            counter = 0
            for idx, batch in enumerate(model_vars['train_dataloader']):
                outputs = model_vars['model'](**batch)
                loss = outputs.loss
                model_vars['accelerator'].backward(loss)
                model_vars['optimizer'].step()                    
                model_vars['optimizer'].zero_grad()
                model_vars['progressbar'].update(1)
                if idx > 0 and idx % step_size == 0:
                    counter += 1
                    self.predict(model_vars=model_vars, outputs=outputs)
                    scores = self.get_scores()
                    score_list.append(scores)
                    print(f"Noise: {self.noise_percentage} | F1-Score (Epoch {epoch}/{counter}): {scores['overall_f1']}")
            model_vars['scheduler'].step(scores["overall_f1"])
        scores = self.parse_scores(score_list=score_list)
        model_vars['model'].save_pretrained(self.model_path)
        model_vars['tokenizer'].save_pretrained(self.model_path)
        del model_vars
        torch.cuda.empty_cache()
        return scores
    
    def explain(self, dataloader, explainability_path):
        model = AutoModelForTokenClassification.from_pretrained(self.model_path, torch_dtype=torch.float16).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        explainer = DataLoaderExplainer(model=model, tokenizer=tokenizer)
    
        for i, explanation in enumerate(explainer.explain_dataset(dataloader=dataloader)):
            explanation.to_csv(os.path.join(explainability_path, f"InDomain_{i}.csv"), index=False)
    
    def noisy_training(self, size:str):
        self.tokenizer = AutoTokenizer.from_pretrained(self.get_model_checkpoint(size=size))
        score_list = list()
        for fold, raw_dataset in enumerate(self.cv):
            path = os.path.join(self.data_path, f"05_STATS/H_NoisyScores/{self.noise}/Scores/Scores.csv")
            train_dataloader, eval_dataloader = self.load_tokenized_data(raw_dataset=raw_dataset)
            scores = self.noise_train_run(train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, size=size)
            score_list.append(scores)
            self.save_scores(score_list=score_list, path = path)
            explainability_path = os.path.join(self.data_path, f"05_STATS/H_NoisyScores/{self.noise}/Attributions/fold_{fold}")
            self.explain(dataloader=eval_dataloader, explainability_path=explainability_path)
            
    def save_scores(self, score_list:list,  path:str):
        df_list = list()
        for fold, score in enumerate(score_list):
            for cat, res in score.items():
                df = pd.DataFrame(res)
                df["category"] = cat
                df["noise"] = self.noise_percentage
                df["fold"] = fold
                df["epoch"] = df.index
                df["data_sample"] = self.sample
                df_list.append(df)
        df = pd.concat(df_list)
        df["steps"] = df["epoch"]
        df["epoch"] = round((df["epoch"] + 1) / 10, 1)
        df.to_csv(path, index=False)
        
if __name__ == "__main__":
    noise_percentage = 0.8
    trainer = NoiseTrainer(noise_percentage=noise_percentage)
    trainer.noisy_training("Base")
