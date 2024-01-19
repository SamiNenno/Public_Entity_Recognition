import os
import logging
import evaluate
import gc
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
        
class QuantiTrainer(BaseTrainer):
    def __init__(self, pretrain_model_path:str, data_path: str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA", model_path: str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS", size:str = "Base", sample: str = "v1", num_folds: int = 5, random_state: int = 2023, train_batch_size: int = 8, eval_batch_size: int = 16, lr: float = 0.000005, num_train_epochs: int = 5, name_combinations: int = 4, clean_epochs: int = 15) -> None:
        super().__init__(data_path, model_path, sample, num_folds, random_state, train_batch_size, eval_batch_size, lr, num_train_epochs, name_combinations)
        self.clean_epochs = clean_epochs
        self.size = size
        self.pretrain_model_path = pretrain_model_path
        self.best_f1 = 0.0
        self.df_list = list()
        
    def quanti_validation(self, path):
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrain_model_path)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        path = os.path.join(self.data_path,"04_TRAINING/02_PER/CleanData/InDomain.hf")
        cv_train_data = self.quanti_validation(path=path)
        
        path = os.path.join(self.data_path, "04_TRAINING/02_PER/CleanData/OutDomain.hf")
        out_domain_data = self.clean_loader(path=path)
        
        path = os.path.join(self.data_path, "04_TRAINING/02_PER/CleanData/Twitter.hf") 
        twitter_data = self.clean_loader(path=path)
        
        path = os.path.join(self.data_path, "04_TRAINING/02_PER/CleanData/English.hf") 
        en_data = self.clean_loader(path=path)
        
        path = os.path.join(self.data_path, "04_TRAINING/02_PER/CleanData/Spanish.hf")
        esp_data = self.clean_loader(path=path)
        return cv_train_data, out_domain_data, twitter_data, en_data, esp_data
    
    def load_quantized_model(self, model_vars, path:str):
        model_vars['accelerator'].wait_for_everyone()
        unwrapped_model = model_vars['accelerator'].unwrap_model(model_vars['model'])
        unwrapped_model.save_pretrained(path, save_function=model_vars['accelerator'].save)
        model_vars['model'] = None
        model_vars.pop('model')
        unwrapped_model = None
        del unwrapped_model
        gc.collect()
        torch.cuda.empty_cache()
        for precision in ["fp16", "int8", "int4"]:
            gc.collect()
            torch.cuda.empty_cache()
            if precision == "fp16":
                model = AutoModelForTokenClassification.from_pretrained(path, torch_dtype=torch.float16)
                model = model_vars["accelerator"].prepare(model)
                yield precision, model.eval()
            elif precision == "int8":
                model = AutoModelForTokenClassification.from_pretrained(path, load_in_8bit=True, device_map="auto")
                model = model_vars["accelerator"].prepare(model)
                yield precision, model.eval()
            else:
                model = AutoModelForTokenClassification.from_pretrained(path, load_in_4bit=True, device_map="auto")
                model = model_vars["accelerator"].prepare(model)
                yield precision, model.eval()
                
    def quant_prediction(self, model_vars, data_dict, fold, epoch, step):
        path = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/Quantization"
        dict_list = list()
        f1 = list()
        for name, dataset in data_dict.items():
            scores = self.predict_on_clean(model_vars['model'], dataset, model_vars['accelerator'])
            dict_list.append(dict(
                Data = name,
                Model = self.size,
                Quantization = "fp32",
                Fold = fold,
                Epoch = epoch,
                Step = step,
                F1 = scores["overall_f1"],
                Recall = scores["overall_recall"],
                Precision = scores["overall_precision"]
            ))
            f1.append(scores["overall_f1"])
        f1 = np.sum(np.array(f1))
        if f1 > self.best_f1:
            self.best_f1 = f1
            print(f"New top F1= {f1}")
        else:
            gc.collect()
            torch.cuda.empty_cache()
            temp = [dict(Data=None, Model=None, Quantization=None, Fold=fold, Epoch=epoch, Step=step, F1=None, Recall=None, Precision=None)] * 4
            return None, pd.DataFrame(temp)
        for precision, model in self.load_quantized_model(model_vars=model_vars, path=path):
            for name, dataset in data_dict.items():
                scores = self.predict_on_clean(model, dataset, model_vars['accelerator'])
                dict_list.append(dict(
                    Data = name,
                    Model = self.size,
                    Quantization = precision,
                    Fold = fold,
                    Epoch = epoch,
                    Step = step,
                    F1 = scores["overall_f1"],
                    Recall = scores["overall_recall"],
                    Precision = scores["overall_precision"]
                ))
            model = None
            del model
            gc.collect()
            torch.cuda.empty_cache()
        model = None
        model_vars['model'] = None
        gc.collect()
        torch.cuda.empty_cache()
        return path, pd.DataFrame(dict_list)
    
    def quanti_train_simple(self, train_dataloader, eval_dataloader, out_domain_data, twitter_data, en_data, esp_data, fold):
        
        path = os.path.join(self.data_path, f"05_STATS/K_Quantization_Scores/")
        model_vars = self.load_training_setup(model_checkpoint=self.pretrain_model_path, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader)
        data_dict = {
            "InDomain":eval_dataloader,
            "OutDomain":model_vars['accelerator'].prepare(out_domain_data),
            "Twitter":model_vars['accelerator'].prepare(twitter_data),
            "English" :model_vars['accelerator'].prepare(en_data),
            "Spanish" :model_vars['accelerator'].prepare(esp_data)
        }
        step_size = len(model_vars['train_dataloader'])//9
        counter = 0
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
                    if counter > 125:
                        path, res = self.quant_prediction(model_vars=model_vars, data_dict=data_dict, fold=fold, epoch=epoch, step=idx)
                        gc.collect()
                        torch.cuda.empty_cache()
                        from time import sleep
                        self.df_list.append(res)
                        pd.concat(self.df_list, ignore_index=True).to_csv(os.path.join("/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA/05_STATS/K_Quantization_Scores", f"{self.size}_shortcut.csv"), index=False)
                        if path is not None:
                            model_vars['model'] = model_vars['accelerator'].prepare(AutoModelForTokenClassification.from_pretrained(path))
                            model_vars['model'].train()    
                    scores = self.get_scores()
                    f1 = scores["overall_f1"]
                    counter += 1
            model_vars['scheduler'].step(f1)  
        model_vars = None     
        del train_dataloader 
        del eval_dataloader
        del model_vars
        gc.collect()
        torch.cuda.empty_cache()
        self.best_f1 = 0.0
        
    def quanti_train(self):
        cv_train_data, out_domain_data, twitter_data, en_data, esp_data = self.load_clean_data()
        for idx, (train_dataloader, eval_dataloader) in enumerate(cv_train_data):
            self.quanti_train_simple(train_dataloader=train_dataloader, 
                                                eval_dataloader=eval_dataloader, 
                                                out_domain_data=out_domain_data,
                                                twitter_data=twitter_data, 
                                                en_data=en_data,
                                                esp_data=esp_data,
                                                fold=idx)
            
  
if __name__ == "__main__":
    sample = "v5"
    size = "Large"
    model_path = f"/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/XL_Data/{size}/double"
    emission_path = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA/05_STATS/F_Emissions"
    emission_name = f"Pretrain_{size}_{sample}"
    trainer = QuantiTrainer(pretrain_model_path=model_path, size=size, sample=sample, num_train_epochs=15)
    tracker = EmissionsTracker(output_dir=emission_path, project_name=emission_name)
    tracker.start()
    trainer.quanti_train()
    tracker.stop()
    print(f"Finished: sample - {sample} | model - {size}")
    del trainer
    torch.cuda.empty_cache()
    