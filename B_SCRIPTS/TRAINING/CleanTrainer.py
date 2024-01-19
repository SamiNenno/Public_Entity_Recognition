import os
import logging
import evaluate
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from accelerate import Accelerator
from scipy.special import softmax
from cleanlab.token_classification.rank import get_label_quality_scores
from codecarbon import EmissionsTracker
from datasets import load_from_disk, DatasetDict
logging.getLogger("codecarbon").disabled = True
logging.getLogger("EmissionsTracker").disabled = True
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, get_scheduler
from BaseTrainer import BaseTrainer
from WeakLoader import WeakLoader
from Explainer import DataLoaderExplainer

class WeakPreTrainer(BaseTrainer):
    def __init__(self, data_path: str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA", model_path: str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS", pretrain_modelpath:str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/02_PER/Dirty/Vanilla/Base",sample:str = "v1", num_folds: int = 5, random_state: int = 2023, train_batch_size: int = 8, eval_batch_size: int = 16, lr: float = 0.000005, num_train_epochs: int = 5, name_combinations: int = 4) -> None:
        super().__init__(data_path=data_path, model_path=model_path, num_folds=num_folds, random_state=random_state, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, lr=lr, num_train_epochs=num_train_epochs, name_combinations=name_combinations, sample=sample)
        #self.weak_data = WeakLoader(data_path=data_path, sample=sample, random_state=random_state).get_vanilla_dataset()
        self.pretrain_modelpath = pretrain_modelpath
        
    def tokenize_data(self, raw_dataset, size):
        self.tokenizer = AutoTokenizer.from_pretrained(self.get_model_checkpoint(size))
        tokenized_datasets = raw_dataset.map(
                    self.tokenize_and_align_labels,
                    batched=True,
                    remove_columns=raw_dataset.column_names,
                )
        tokenized_datasets = tokenized_datasets.train_test_split(test_size=.05, seed=self.random_state)
        trainloader = DataLoader(
                    tokenized_datasets["train"],
                    shuffle=True,
                    collate_fn=self.data_collator,
                    batch_size=self.train_batch_size,
                )
        testloader = DataLoader(
                    tokenized_datasets["test"],
                    shuffle=True,
                    collate_fn=self.data_collator,
                    batch_size=self.eval_batch_size,
                )
        return trainloader, testloader
    
    def pretrain(self, size):
        trainloader, testloader = self.tokenize_data(raw_dataset=self.weak_data, size=size)
        model_vars = self.load_training_setup(model_checkpoint=self.get_model_checkpoint(size=size), train_dataloader=trainloader, eval_dataloader=testloader)
        counter = 0
        best_f1 = 0
        f1_prev_epoch = 0
        step_size = len(model_vars['train_dataloader'])//10
        for epoch in range(model_vars["epochs"]):
            model_vars['model'].train()
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
                    model_vars['accelerator'].wait_for_everyone()
                    #unwrapped_model = model_vars['accelerator'].unwrap_model(model_vars['model'])
                    #unwrapped_model.save_pretrained(self.model_path, save_function=model_vars['accelerator'].save)
                    if model_vars['accelerator'].is_main_process:
                        self.tokenizer.save_pretrained(self.model_path)
                    scores = self.get_scores()
                    f1 = scores["overall_f1"]
                    model_vars['scheduler'].step(scores["overall_f1"])
                    if f1 > best_f1:
                        print(f"Pre-Training step {counter} (Epoch: {epoch+1}): New best F1-Score: {f1} (Before: {best_f1})")
                        best_f1 = f1
                        model_vars['model'].save_pretrained(self.pretrain_modelpath)
                        model_vars['tokenizer'].save_pretrained(self.pretrain_modelpath)
                        print(f"Model saved to: {self.pretrain_modelpath}")
            if best_f1 == f1_prev_epoch:
                print(f"No improvement from epoch {epoch - 1} to epoch {epoch} (Best F1 = {best_f1} and F1 last epoch = {f1_prev_epoch}). Interrupt Pre-Training")
                break
            else:
                f1_prev_epoch = best_f1
        
class CleanTrainer(WeakPreTrainer):
    def __init__(self, data_path: str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA", model_path: str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS", pretrain_modelpath: str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/02_PER/Dirty/Vanilla/Base", sample: str = "v1", num_folds: int = 5, random_state: int = 2023, train_batch_size: int = 4, eval_batch_size: int = 16, lr: float = 0.000005, num_train_epochs: int = 5, name_combinations: int = 4, clean_epochs: int = 15) -> None:
        super().__init__(data_path, model_path, pretrain_modelpath, sample, num_folds, random_state, train_batch_size, eval_batch_size, lr, num_train_epochs, name_combinations)
        self.clean_epochs = clean_epochs
        
    def clean_validation(self, path):
        dataset = load_from_disk(path)
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.random_state)
        splits = kf.split(np.zeros(dataset.num_rows))
        for train_idxs, val_idxs in tqdm(splits, total=self.num_folds, desc=f"{self.num_folds}-fold CV", leave=False):
            raw_dataset = DatasetDict({
                'train':dataset.select(train_idxs),
                'validation':dataset.select(val_idxs),
            })
            train_dataloader, eval_dataloader = self.load_tokenized_data(raw_dataset=raw_dataset)
            yield train_dataloader, eval_dataloader
            
    def load_clean_data(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrain_modelpath)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        path = os.path.join(self.data_path,"04_TRAINING/02_PER/CleanData/InDomain.hf")
        cv_train_data = self.clean_validation(path=path)
        
        path = os.path.join(self.data_path, "04_TRAINING/02_PER/CleanData/OutDomain.hf")
        out_domain_data = self.clean_loader(path=path)
        
        path = os.path.join(self.data_path, "04_TRAINING/02_PER/CleanData/Twitter.hf") 
        twitter_data = self.clean_loader(path=path)
        
        path = os.path.join(self.data_path, "04_TRAINING/02_PER/CleanData/English.hf") 
        en_data = self.clean_loader(path=path)
        
        path = os.path.join(self.data_path, "04_TRAINING/02_PER/CleanData/Spanish.hf")
        esp_data = self.clean_loader(path=path)
        return cv_train_data, out_domain_data, twitter_data, en_data, esp_data
    
    def clean_train_simple(self, train_dataloader, eval_dataloader, out_domain_data, twitter_data, en_data, esp_data):
        score_list = list()
        out_domain_score_list = list()
        twitter_score_list = list()
        en_score_list = list()
        esp_score_list = list()
        print(self.pretrain_modelpath)
        model_vars = self.load_training_setup(model_checkpoint=self.pretrain_modelpath, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader)
        step_size = len(model_vars['train_dataloader'])//9
        for epoch in range(self.clean_epochs):
                model_vars['model'].train()
                for idx, batch in enumerate(model_vars['train_dataloader']):
                    outputs = model_vars['model'](**batch)
                    loss = outputs.loss
                    model_vars['accelerator'].backward(loss)
                    model_vars['optimizer'].step()
                    model_vars['optimizer'].zero_grad()
                    model_vars['progressbar'].update(1)
                    if idx > 0 and idx%step_size == 0:
                        self.predict(model_vars=model_vars, outputs=outputs)
                        scores = self.get_scores()
                        score_list.append(scores)
                        f1 = scores["overall_f1"]
                        out_domain_scores = self.predict_on_clean(model_vars['model'], out_domain_data, model_vars['accelerator'])
                        out_domain_score_list.append(out_domain_scores)
                        
                        twitter_scores = self.predict_on_clean(model_vars['model'], twitter_data, model_vars['accelerator'])
                        twitter_score_list.append(twitter_scores)
                        
                        en_scores = self.predict_on_clean(model_vars['model'], en_data, model_vars['accelerator'])
                        en_score_list.append(en_scores)
                        
                        esp_scores = self.predict_on_clean(model_vars['model'], esp_data, model_vars['accelerator'])
                        esp_score_list.append(esp_scores)
                self.predict(model_vars=model_vars, outputs=outputs)

                model_vars['accelerator'].wait_for_everyone()
                unwrapped_model = model_vars['accelerator'].unwrap_model(model_vars['model'])
                unwrapped_model.save_pretrained(self.model_path, save_function=model_vars['accelerator'].save)
                if model_vars['accelerator'].is_main_process:
                    self.tokenizer.save_pretrained(self.model_path)
                scores = self.get_scores()
                score_list.append(scores)
                f1 = scores["overall_f1"]
                out_domain_scores = self.predict_on_clean(model_vars['model'], out_domain_data, model_vars['accelerator'])
                out_domain_score_list.append(out_domain_scores)
                
                twitter_scores = self.predict_on_clean(model_vars['model'], twitter_data, model_vars['accelerator'])
                twitter_score_list.append(twitter_scores)
                
                en_scores = self.predict_on_clean(model_vars['model'], en_data, model_vars['accelerator'])
                en_score_list.append(en_scores)
                
                esp_scores = self.predict_on_clean(model_vars['model'], esp_data, model_vars['accelerator'])
                esp_score_list.append(esp_scores)
                        
        model_vars['scheduler'].step(f1)        
        scores = self.parse_scores(score_list=score_list)
        out_domain_scores = self.parse_scores(score_list=out_domain_score_list)
        twitter_scores = self.parse_scores(score_list=twitter_score_list)
        en_scores = self.parse_scores(score_list=en_score_list)
        esp_scores = self.parse_scores(score_list=esp_score_list)
        del train_dataloader 
        del eval_dataloader
        return scores, out_domain_scores, twitter_scores, en_scores, esp_scores, model_vars
    
    def clean_train(self, size:str, explain:bool = False):
        score_list = list()
        out_domain_score_list = list()
        twitter_score_list = list()
        en_score_list = list()
        esp_score_list = list()
        path = os.path.join(self.data_path, f"05_STATS/B_Clean_Scores/")
        cv_train_data, out_domain_data, twitter_data, en_data, esp_data = self.load_clean_data()
        for idx, (train_dataloader, eval_dataloader) in enumerate(cv_train_data):
            scores, out_domain_scores, twitter_scores, en_scores, esp_scores, model_vars = self.clean_train_simple(train_dataloader=train_dataloader, 
                                                                                 eval_dataloader=eval_dataloader, 
                                                                                 out_domain_data=out_domain_data,
                                                                                 twitter_data=twitter_data, 
                                                                                 en_data=en_data,
                                                                                 esp_data=esp_data)
            score_list.append(scores)
            out_domain_score_list.append(out_domain_scores)
            twitter_score_list.append(twitter_scores)
            en_score_list.append(en_scores)
            esp_score_list.append(esp_scores)
            self.save_scores(score_list=score_list, path=path, test_type="InDomain", size=size.capitalize())
            self.save_scores(score_list=out_domain_score_list, path=path, test_type="OutDomain", size=size.capitalize())
            self.save_scores(score_list=twitter_score_list, path=path, test_type="Twitter", size=size.capitalize())
            self.save_scores(score_list=en_score_list, path=path, test_type="English", size=size.capitalize())
            self.save_scores(score_list=esp_score_list, path=path, test_type="Spanish", size=size.capitalize())
            if explain:
                explainability_path = os.path.join(self.data_path, f"05_STATS/C_Explainability/fold_{idx}")
                self.explain(model=model_vars["model"], tokenizer=model_vars["tokenizer"], path=explainability_path,
                            eval_dataloader=eval_dataloader, out_domain_data=out_domain_data, twitter_data=twitter_data,
                            en_data=en_data, esp_data=esp_data)
            del model_vars
            torch.cuda.empty_cache()
            
    def save_scores(self, score_list:list,  path:str, test_type:str, size:str):
        df_list = list()
        for fold, score in enumerate(score_list):
            for cat, res in score.items():
                df = pd.DataFrame(res)
                df["category"] = cat
                df["fold"] = fold
                df["epoch"] = df.index
                df["label_type"] = test_type
                df["data_sample"] = self.sample
                df["model_type"] = size
                df_list.append(df)
        df = pd.concat(df_list)
        df["steps"] = df["epoch"]
        df["epoch"] = round((df["epoch"] + 1) / 10, 1)
        df.to_csv(os.path.join(path, f"clean_OnlyXL_{test_type}_{size}L_scores.csv"), index=False)
        #df.to_csv(os.path.join(path, f"clean_{self.sample}_{test_type}_{size}_scores.csv"), index=False)
        #df.to_csv(os.path.join(path, f"clean_no_pretrain_{test_type}_{size}_scores.csv"), index=False)
        
    def explain(self, model, tokenizer, path, eval_dataloader, out_domain_data, twitter_data, en_data, esp_data):
        explainer = DataLoaderExplainer(model=model, tokenizer=tokenizer)
        
        explanations = explainer.explain_dataset(dataloader=eval_dataloader)
        for idx, explanation in enumerate(explanations):
            explanation.to_csv(os.path.join(path, f"InDomain_{idx}.csv"), index=False)
            
        explanations = explainer.explain_dataset(dataloader=out_domain_data)
        for idx, explanation in enumerate(explanations):
            explanation.to_csv(os.path.join(path, f"OutDomain{idx}.csv"), index=False)
            
        explanations = explainer.explain_dataset(dataloader=twitter_data)
        for idx, explanation in enumerate(explanations):
            explanation.to_csv(os.path.join(path, f"Twitter{idx}.csv"), index=False)
            
        explanations = explainer.explain_dataset(dataloader=en_data)
        for idx, explanation in enumerate(explanations):
            explanation.to_csv(os.path.join(path, f"English{idx}.csv"), index=False)
            
        explanations = explainer.explain_dataset(dataloader=esp_data)
        for idx, explanation in enumerate(explanations):
            explanation.to_csv(os.path.join(path, f"Spanish{idx}.csv"), index=False)
            
if __name__ == "__main__":
    sample = "v4"
    size = "Large"
    model_path = f"/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/XL_Data/Large/simple"
    emission_path = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA/05_STATS/F_Emissions"
    emission_name = f"Pretrain_{size}_{sample}"
    trainer = CleanTrainer(pretrain_modelpath=model_path, sample=sample, num_train_epochs=15)
    tracker = EmissionsTracker(output_dir=emission_path, project_name=emission_name)
    tracker.start()
    trainer.clean_train(size=size)
    tracker.stop()
    print(f"Finished: sample - {sample} | model - {size}")
    del trainer
    torch.cuda.empty_cache()
    
    '''sample = "v4"
    size = "Large"
    model_path = f"/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/{size}/{sample}"
    trainer = CleanTrainer(pretrain_modelpath=model_path, sample=sample, num_train_epochs=4)
    emission_path = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA/05_STATS/F_Emissions"
    emission_name = f"Pretrain_{size}_{sample}"
    tracker = EmissionsTracker(output_dir=emission_path, project_name=emission_name)
    tracker.start()
    trainer.pretrain(size=size)
    tracker.stop()
    
    emission_name = f"Cleantrain_{size}_{sample}"
    tracker = EmissionsTracker(output_dir=emission_path, project_name=emission_name)
    tracker.start()
    trainer.clean_train(size=size)
    tracker.stop()
    print(f"Finished: sample - {sample} | model - {size}")
    del trainer
    torch.cuda.empty_cache()'''