import os
import logging
import evaluate
import pandas as pd
import numpy as np
from tqdm import tqdm
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
from WeakLoader import WeakLoader



class Prediction():
    def __init__(self, label_dict) -> None:
        self.metric = evaluate.load("seqeval")
        self.true_predictions, self.true_labels = [], []
        self.pred_dict = dict()
        self.epoch_counter = -1
        self.label_dict = label_dict
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def postprocess(self, predictions, labels):
        predictions = predictions.detach().cpu().clone().numpy()
        labels = labels.detach().cpu().clone().numpy()

        true_labels = [[self.label_dict[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.label_dict[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]        
        return true_labels, true_predictions  
    
    def predict(self, model_vars, outputs):
        model_vars['model'].eval()
        #with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in tqdm(model_vars['eval_dataloader'], desc='Evaluate', total=len(model_vars['eval_dataloader']), leave=False):
            with torch.no_grad():
                outputs = model_vars['model'](**batch)
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]

            predictions = model_vars['accelerator'].pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = model_vars['accelerator'].pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = model_vars['accelerator'].gather(predictions)
            labels_gathered = model_vars['accelerator'].gather(labels)

            true_predictions, true_labels = self.postprocess(predictions_gathered, labels_gathered)
            self.metric.add_batch(predictions=true_predictions, references=true_labels)
            
    def get_scores(self):
        return self.metric.compute(zero_division=0)
    
    def parse_scores(self, score_list):
        results = {'BEHÖRDE':[], 'JOURNALIST':[], 'MEDIEN':[], 'PARTEI':[], 'POLITIKER':[], "OVERALL":[]}
        for idx, score in enumerate(score_list):
            overall = dict()
            for key, value in score.items():
                if key in results.keys():
                    results[key].append(value)
                else:
                    overall[key.split('_')[1]] = value
            results["OVERALL"].append(overall)
        return results
    
    def predict_on_clean(self, model, dataloader, accelerator):
        metric = evaluate.load("seqeval")
        model.eval()
        for batch in tqdm(dataloader, desc="Eval on Clean", total=len(dataloader), leave=False):
            with torch.no_grad():
                batch = batch.to(self.device)
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            
            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)

            true_predictions, true_labels = self.postprocess(predictions_gathered, labels_gathered)
            metric.add_batch(predictions=true_predictions, references=true_labels)

        clean_score = metric.compute(zero_division=0)
        return clean_score
    
class ModelUtils(Prediction):
    def __init__(self, 
                 data_path:str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA", 
                 model_path:str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS",
                 sample:str = "v1",
                 num_folds:int = 5,
                 random_state:int = 2023,
                 train_batch_size:int = 8,
                 eval_batch_size:int = 16,
                 lr:float = 5e-6,
                 num_train_epochs:int = 5,
                 name_combinations:int = 4) -> None:
        self.weak_loader = WeakLoader(data_path=data_path, 
                                      num_epochs=num_train_epochs,
                                      num_folds=num_folds,
                                      sample=sample,
                                      random_state=random_state,
                                      name_combinations=name_combinations)
        self.id2label, self.label2id  = self.weak_loader.get_id2label(), self.weak_loader.get_label2id()
        super().__init__(label_dict=self.id2label)
        self.sample = sample
        self.model_path = model_path
        self.data_path = data_path
        self.num_folds = num_folds
        self.random_state = random_state
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_train_epochs = num_train_epochs
        self.lr = lr
        
    def get_model_checkpoint(self, size:str):
        if size.lower() in ["large", "l", "big"]:
            checkpoint = "xlm-roberta-large"
        elif size.lower() in ["base", "b", "small", "s"]:
            checkpoint = "xlm-roberta-base"
        elif size.lower() in ["xlarge", "xl"]:
            checkpoint = "facebook/xlm-roberta-xl" 
        elif size.lower() in ["xxlarge", "xxl"]:
            checkpoint = "facebook/xlm-roberta-xxl"
        elif size.lower() in ["de_base", "ger_base"]:
            checkpoint = "deepset/gbert-base"
        elif size.lower() in ["de_large", "de_large"]:
            checkpoint = "deepset/gbert-large"
        elif size.lower() in ["distil_de", "de_distil"]:
            checkpoint = "distilbert-base-german-cased"
        else:
            raise KeyError(f"{size} is not a valid model size. Choose either 'large' or 'base'!")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        return checkpoint
    
    def load_model_utils(self, model_checkpoint:str):
        dtype = torch.float16 if model_checkpoint in ["facebook/xlm-roberta-xl","facebook/xlm-roberta-xxl"] else torch.float32
        try:
            del model_util
        except NameError:
            pass
        accelerator = Accelerator()
        model = AutoModelForTokenClassification.from_pretrained(
                model_checkpoint,
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True,
                torch_dtype=dtype
                )
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        optimizer = AdamW(model.parameters(), lr=self.lr)
        model_util = dict(
            model = model,
            tokenizer = tokenizer,
            accelerator = accelerator,
            optimizer = optimizer,
            epochs = self.num_train_epochs,
            model_path = self.model_path,
            checkpoint_path = os.path.join(self.model_path, "03_CHECKPOINT")
        )
        return model_util
        
    def load_train_data(self, cv:bool, shuffled:bool, clean:bool, pretrain:bool):
        if pretrain:
            pass #ToDo load ner dataset
        if clean:
            pass #ToDo load clean dataset
        if cv:
            if shuffled:
                return self.weak_loader.get_shuffled_crossvalidation()
            else:
                return self.weak_loader.get_vanilla_crossvalidation()
        else:
            if shuffled:
                return self.weak_loader.get_epoch_shuffled_dataset()
            else:
                return self.weak_loader.get_vanilla_dataset()
        
class BaseTrainer(ModelUtils):
    def __init__(self, data_path: str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA", model_path: str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS", sample: str = "v1", num_folds: int = 5, random_state: int = 2023, train_batch_size: int = 8, eval_batch_size: int = 16, lr: float = 0.000005, num_train_epochs: int = 5, name_combinations: int = 4) -> None:
        super().__init__(data_path, model_path, sample, num_folds, random_state, train_batch_size, eval_batch_size, lr, num_train_epochs, name_combinations)
       
    def load_training_setup(self, model_checkpoint:str, train_dataloader, eval_dataloader):
        model_util = self.load_model_utils(model_checkpoint=model_checkpoint)
        model_util["train_dataloader"] = train_dataloader
        model_util["eval_dataloader"] = eval_dataloader
        num_training_steps = self.num_train_epochs * len(train_dataloader)
        model_util["num_training_steps"] = num_training_steps
        lr_scheduler = get_scheduler(
            "reduce_lr_on_plateau",#"linear",
            optimizer=model_util["optimizer"],
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        model_util["scheduler"] = lr_scheduler
        model_util["model"], model_util["optimizer"], model_util["train_dataloader"], model_util["eval_dataloader"] = model_util["accelerator"].prepare(
            model_util["model"], model_util["optimizer"], model_util["train_dataloader"], model_util["eval_dataloader"]
        )
        model_util["progressbar"] = tqdm(range(num_training_steps), desc='Train', total=num_training_steps, leave=False)
        return model_util
        
    def tokenize_and_align_labels(self, examples):        
        def align_labels_with_tokens(labels, word_ids):
            new_labels = []
            current_word = None
            for word_id in word_ids:
                if word_id != current_word:
                    current_word = word_id
                    label = -100 if word_id is None else labels[word_id]
                    new_labels.append(label)
                elif word_id is None:
                    new_labels.append(-100)
                else:
                    label = labels[word_id]
                    if label % 2 == 1:
                        label += 1
                    new_labels.append(label)

            return new_labels
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, padding=True,is_split_into_words=True
        )
        all_labels = examples["tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs
    
    def load_tokenized_data(self, raw_dataset):
        tokenized_datasets = raw_dataset.map(
                    self.tokenize_and_align_labels,
                    batched=True,
                    remove_columns=raw_dataset["train"].column_names,
                )
        train_dataloader = DataLoader(
                    tokenized_datasets["train"],
                    shuffle=True,
                    collate_fn=self.data_collator,
                    batch_size=self.train_batch_size,
                )
        eval_dataloader = DataLoader(
                    tokenized_datasets["validation"], 
                    collate_fn=self.data_collator, 
                    batch_size=self.eval_batch_size
                )
        return train_dataloader, eval_dataloader
    
    def clean_loader(self, path):
        data = load_from_disk(path)
        tokenized_dataset = data.map(self.tokenize_and_align_labels, batched=True, remove_columns=data.column_names)
        loader = DataLoader(tokenized_dataset, collate_fn=self.data_collator, batch_size=self.eval_batch_size)
        return loader
    
    def load_clean_data(self):
        clean_data = dict()
        path = os.path.join(self.data_path, "04_TRAINING/02_PER/CleanData/InDomain.hf") 
        clean_data["InDomain"] = self.clean_loader(path=path)
            
        path = os.path.join(self.data_path, "04_TRAINING/02_PER/CleanData/OutDomain.hf") 
        clean_data["OutofDomain"] = self.clean_loader(path=path)
        
        path = os.path.join(self.data_path, "04_TRAINING/02_PER/CleanData/Twitter.hf") 
        clean_data["Twitter"] = self.clean_loader(path=path)
        
        path = os.path.join(self.data_path, "04_TRAINING/02_PER/CleanData/English.hf") 
        clean_data["English"] = self.clean_loader(path=path)
        
        path = os.path.join(self.data_path, "04_TRAINING/02_PER/CleanData/Spanish.hf") 
        clean_data["Spanish"] = self.clean_loader(path=path)
        
        return clean_data