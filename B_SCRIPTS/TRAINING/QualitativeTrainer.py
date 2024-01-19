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
        
class QualiTrainer(BaseTrainer):
    def __init__(self, data_path: str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA", model_path: str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS", pretrain_modelpath:str= "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/02_PER/Dirty/Vanilla/Base",sample: str = "v1", num_folds: int = 5, random_state: int = 2023, train_batch_size: int = 8, eval_batch_size: int = 16, lr: float = 0.000005, num_train_epochs: int = 5, name_combinations: int = 4, clean_epochs:int = 15) -> None:
        super().__init__(data_path, model_path, sample, num_folds, random_state, train_batch_size, eval_batch_size, lr, num_train_epochs, name_combinations)
        self.clean_epochs = clean_epochs
        self.weak_data = WeakLoader(data_path=data_path, sample=sample, random_state=random_state).get_vanilla_dataset()
        self.pretrain_modelpath = pretrain_modelpath
        
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
    
    
    def quali_train_simple(self, train_dataloader, eval_dataloader, out_domain_data, twitter_data, en_data, esp_data, fold, folder_path):
        loaders = [eval_dataloader, out_domain_data, twitter_data, en_data, esp_data]
        dataset_names = ["InDomain", "OutDomain", "Twitter", "English", "Spanish"]
        if fold != 0:
            loaders = [eval_dataloader]
            dataset_names = ["InDomain"]
        model_vars = self.load_training_setup(model_checkpoint=self.pretrain_modelpath, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader)
        for epoch in tqdm(range(self.clean_epochs), total=self.clean_epochs, desc="Train", leave=False):
            model_vars['model'].train()
            for idx, batch in tqdm(enumerate(model_vars['train_dataloader']), total=len(model_vars['train_dataloader']), desc=f"Epoch: {epoch}", leave=False):
                outputs = model_vars['model'](**batch)
                loss = outputs.loss
                model_vars['accelerator'].backward(loss)
                model_vars['optimizer'].step()
                model_vars['optimizer'].zero_grad()
        
            self.predict(model_vars=model_vars, outputs=outputs)
            scores = self.get_scores()
            f1 = scores["overall_f1"]
            model_vars['scheduler'].step(f1)  
            
        model_vars['model'].eval()
        for dataset_name, loader in tqdm(zip(dataset_names, loaders), total=len(loaders), desc="Run Test Predictions", leave=False):
            for batch_idx, batch in tqdm(enumerate(loader), desc=dataset_name, total=len(loader), leave=False):
                with torch.no_grad():
                    batch = batch.to(self.device)
                    outputs = model_vars['model'](**batch)
                predictions = outputs.logits.argmax(dim=-1).detach().cpu()
                labels = batch["labels"]
                tokens = batch["input_ids"]
                for example_idx, (pred, lab, tok) in enumerate(zip(predictions, labels, tokens)):
                    dct = dict(
                        Token = tok.detach().cpu().tolist(),
                        Label = lab.detach().cpu().tolist(),
                        Pred =pred.detach().cpu().tolist()
                    )
                    path = os.path.join(os.path.join(folder_path, dataset_name), f"example_{example_idx}_batch_{batch_idx}_fold_{fold}.csv")
                    df = pd.DataFrame(dct)
                    df = df[df["Label"] != -100]
                    df["Token"] = df["Token"].apply(lambda x: self.tokenizer.decode(x))
                    df.to_csv(path, index=False)
        del model_vars
        torch.cuda.empty_cache()
    
    def quali_train(self, size:str="XL", path:str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA/05_STATS/I_Quali_Sample"):
        path = os.path.join(path, size)
        cv_train_data, out_domain_data, twitter_data, en_data, esp_data = self.load_clean_data()
        for idx, (train_dataloader, eval_dataloader) in tqdm(enumerate(cv_train_data), total=self.num_folds, desc="CV", leave=False):
            self.quali_train_simple(train_dataloader=train_dataloader,
                                    eval_dataloader=eval_dataloader,
                                    out_domain_data=out_domain_data,
                                    twitter_data=twitter_data, 
                                    en_data=en_data,
                                    esp_data=esp_data,
                                    fold = idx,
                                    folder_path=path)
            
            
            
if __name__ == "__main__":
    sample = "v4"
    size = "Base"
    clean_epochs = 15
    model_path = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/XL_Data/Base/double"
    #model_path = f"/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/{size}/{sample}"
    trainer = QualiTrainer(pretrain_modelpath=model_path, sample=sample, clean_epochs=clean_epochs)
    emission_path = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA/05_STATS/F_Emissions"
    emission_name = f"Pretrain_{size}_{sample}"
    tracker = EmissionsTracker(output_dir=emission_path, project_name=emission_name)
    tracker.start()
    trainer.quali_train()
    tracker.stop()
    