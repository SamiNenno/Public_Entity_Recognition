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
from VanillaTrainer import VanillaTrainer

class ConfidentTrainer(VanillaTrainer):
    def __init__(self, data_path: str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA", model_path: str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS", num_folds: int = 5, random_state: int = 2023, train_batch_size: int = 4, eval_batch_size: int = 8, lr: float = 0.000005, num_train_epochs: int = 3, name_combinations: int = 4, clean_epochs:int = 15, sample:str="v2") -> None:
        super().__init__(data_path, model_path, sample, num_folds, random_state, train_batch_size, eval_batch_size, lr, num_train_epochs, name_combinations)
        self.clean_epochs = clean_epochs
    
    def unpack_validation_set(self, raw_dataset):
        dct_list = list()
        for idx, tokens, tags in raw_dataset["validation"].to_pandas().itertuples(index=True, name=None):
            for token, tag in zip(tokens, tags):
                    dct_list.append(dict(
                        sentence = idx,
                        token=token,
                        tag = tag
                    ))
        return pd.DataFrame(dct_list)
    
    def confidence_scores(self, df):
        labels = list()
        pred_probs = list()
        for name, group in df.groupby("example"):
            labels.append(group["label"].values)
            pred_probs.append(group[[idx for idx in range(11)]].values)
        sentence_scores, token_scores = get_label_quality_scores(labels, pred_probs)
        df_list = list()
        for idx, (name, group) in enumerate(df.groupby("example")):
            group["Token_label_quality"] = token_scores[idx].to_list()
            group["Sentence_label_quality"] = sentence_scores[idx]
            df_list.append(group)
        df = pd.concat(df_list, ignore_index=True)
        return df
    
    def confident_simple_train(self, raw_dataset, size:str):
        self.tokenizer = AutoTokenizer.from_pretrained(self.get_model_checkpoint(size))
        train_dataloader, eval_dataloader = self.load_tokenized_data(raw_dataset=raw_dataset)
        model_vars = self.load_training_setup(model_checkpoint=self.get_model_checkpoint(size), train_dataloader=train_dataloader, eval_dataloader=eval_dataloader)
        self.clean_data = self.load_clean_data()
        clean_traindata = self.clean_data["InDomain"]
        step_size = len(model_vars['train_dataloader'])//10
        counter = 0
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
                    print(f"Weak F1 (epoch {epoch}): {f1}")
                    model_vars['scheduler'].step(f1)  
        lr_scheduler = get_scheduler(
            "reduce_lr_on_plateau",
            optimizer=model_vars["optimizer"],
            num_warmup_steps=0,
            num_training_steps=model_vars["epochs"] * len(clean_traindata),
        )
        lr_scheduler = model_vars["accelerator"].prepare(lr_scheduler)        
        model_vars['scheduler'] = lr_scheduler
        counter = 0
        step_size = len(clean_traindata)//10
        for epoch in tqdm(range(self.clean_epochs), total=self.clean_epochs, desc="Train on clean data", leave=False):
            model_vars['model'].train()
            for idx, batch in enumerate(clean_traindata):
                batch = batch.to("cuda")
                outputs = model_vars['model'](**batch)
                loss = outputs.loss
                model_vars['accelerator'].backward(loss)
                model_vars['optimizer'].step()
                model_vars['optimizer'].zero_grad()
                if idx > 0 and idx % step_size == 0:
                    model_vars['scheduler'].step(loss)

        model_vars['model'].eval()
        df_list = list()
        counter = 0    
        for batch in tqdm(model_vars['eval_dataloader'], desc='Evaluate', total=len(model_vars['eval_dataloader']), leave=False):
            with torch.no_grad():
                outputs = model_vars['model'](**batch)
            for example_ids, labels, example_logits in zip(batch["input_ids"], batch["labels"],outputs.logits):
                words = [model_vars["tokenizer"].convert_ids_to_tokens(token_id) for token_id in example_ids.tolist()]
                df = pd.concat([pd.DataFrame(dict(word=words, label=labels.detach().cpu().numpy().tolist(),example=[counter]*len(words))), pd.DataFrame(softmax(example_logits.detach().cpu().numpy(), axis=1), columns=[idx for idx in range(11)])], axis=1)
                df_list.append(df)
                counter += 1
        df = pd.concat(df_list, ignore_index=True)
        df = df[~df["word"].isin(["<pad>", "<s>", "</s>"])]
        df = self.confidence_scores(df)
        del model_vars
        torch.cuda.empty_cache()
        return df      
        
    def confident_learning(self, size:str):
        df_list = list()
        path = os.path.join(self.data_path, f"05_STATS/G_Confidence_Scores/ConfidenceProbs_{size}_{self.sample}.csv")
        train_iter = self.load_train_data(cv=True, shuffled=False, clean=False, pretrain=False)
        for idx, train_data in enumerate(train_iter):
            df = self.confident_simple_train(raw_dataset=train_data, size=size)
            df["fold"] = idx
            df_list.append(df)
            pd.concat(df_list, ignore_index=True).to_csv(path, index=False)
            
if __name__ == "__main__":
    trainer = ConfidentTrainer()
    trainer.confident_learning(size="de_distil")