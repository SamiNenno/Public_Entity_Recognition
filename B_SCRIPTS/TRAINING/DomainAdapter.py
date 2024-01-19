import re
import os
import json
import torch
import random
import logging
import evaluate
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from codecarbon import EmissionsTracker
logging.getLogger("codecarbon").disabled = True
logging.getLogger("EmissionsTracker").disabled = True
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, get_scheduler

class DomainTrainer():
    def __init__(self, model_checkpoint:str, label2id_path:str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA/04_TRAINING/02_PER/label2id.json", epochs:int = 10) -> None:
        self.model_checkpoint = model_checkpoint
        self.epochs = epochs
        self.train_batch_size = 1
        self.eval_batch_size = 64
        self.get_label_dct(label2id_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        self.metric = evaluate.load("seqeval")
        
    def get_label_dct(self, label2id_path):
        with open(label2id_path, "r") as f:
            self.label2id = json.load(f)
        self.id2label = {i:l for l, i in self.label2id.items()}
        
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
                    tokenized_datasets["test"], 
                    collate_fn=self.data_collator, 
                    batch_size=self.eval_batch_size
                )
        return train_dataloader, eval_dataloader
        
    def make_model_vars(self, data):
        train_dataloader, eval_dataloader = self.load_tokenized_data(raw_dataset=data)
        
        model = AutoModelForTokenClassification.from_pretrained(
                self.model_checkpoint,
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True,
                )
        
        optimizer = AdamW(model.parameters(), lr=0.000005)
        num_training_steps = self.epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            "reduce_lr_on_plateau",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        accelerator = Accelerator()
        model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
            model, train_dataloader, eval_dataloader, optimizer, lr_scheduler)
        self.model_vars = dict(
            model = model,
            tokenizer = self.tokenizer,
            accelerator = accelerator,
            optimizer = optimizer,
            epochs = self.epochs,
            lr_scheduler = lr_scheduler,
            train_dataloader = train_dataloader,
            eval_dataloader = eval_dataloader,
            num_training_steps = num_training_steps
        )
        self.model_vars["progressbar"] = tqdm(range(num_training_steps), desc='Train', total=num_training_steps, leave=False)
        self.model_vars["pred_steps"] = list(range(int(len(self.model_vars["train_dataloader"])/5), self.model_vars["num_training_steps"]+1, int(len(self.model_vars["train_dataloader"])/5)))
        self.model_vars["pred_counter"] = 0
        
    def postprocess(self, predictions, labels):
        predictions = predictions.detach().cpu().clone().numpy()
        labels = labels.detach().cpu().clone().numpy()

        true_labels = [[self.id2label[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]        
        return true_labels, true_predictions  
    
    def predict(self, model_vars, outputs):
        model_vars['model'].eval()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
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
        model_vars['model'].train()
            
    def get_scores(self):
        return self.metric.compute(zero_division=0)
    
    def train(self, data):
        best_scores = {}
        best_f1 = 0.0
        counter = 0
        self.make_model_vars(data)
        for epoch in range(self.model_vars["epochs"]):
            self.model_vars['model'].train()
            for batch in self.model_vars['train_dataloader']:
                outputs = self.model_vars['model'](**batch)
                loss = outputs.loss
                self.model_vars['accelerator'].backward(loss)
                self.model_vars['optimizer'].step()                    
                self.model_vars['optimizer'].zero_grad()
                self.model_vars["pred_counter"] += 1
                self.model_vars['progressbar'].update(1)
                self.predict(model_vars=self.model_vars, outputs=outputs)
                scores = self.get_scores()
                f1 = scores["overall_f1"]
                if f1 > best_f1:
                    best_scores = scores
                    best_f1 = f1
                    best_step = counter
                counter += 1
        return best_scores, best_step, counter

def sample_and_remove(sent_ids, n, random_state:int = 2023):
    random.seed(random_state)
    sample = random.sample(sent_ids, n)
    remainder = [id_ for id_ in sent_ids if id_ not in sample]
    return sample, remainder

def train_test_split(sent_ids, train_fraction:float = 0.7, random_state:int = 2023):
    n = int(len(sent_ids) * train_fraction)
    train_ids, test_ids = sample_and_remove(sent_ids, n, random_state)
    return train_ids, test_ids

def make_train_test_dataset(df, train_ids, test_ids):
    dict_list = list()
    for train_id in train_ids:
        group = df[df["Sentence"] == train_id]
        dict_list.append({
            "tokens" : group['Entity'].to_list(),
            "tags" : group['Label'].to_list()
        })
    train_data = Dataset.from_pandas(pd.DataFrame(dict_list))
    dict_list = list()
    for test_id in test_ids:
        group = df[df["Sentence"] == test_id]
        dict_list.append({
            "tokens" : group['Entity'].to_list(),
            "tags" : group['Label'].to_list()
        })
    test_data = Dataset.from_pandas(pd.DataFrame(dict_list))
    return DatasetDict({"train":train_data, "test": test_data})

def train_test_generator(df_path:str, train_fraction:float = 0.7, initial_n:int = 20, step:int = 10, random_state:int = 2023):
    label2id = {
        "O": 0,
        "B-BEHÖRDE": 1,
        "I-BEHÖRDE": 2,
        "B-POLITIKER": 3,
        "I-POLITIKER": 4,
        "B-MEDIEN": 5,
        "I-MEDIEN": 6,
        "B-PARTEI": 7,
        "I-PARTEI": 8,
        "B-JOURNALIST": 9,
        "I-JOURNALIST": 10
    }
    df = pd.read_csv(df_path)
    df["Label"] = df["Label"].apply(lambda x: label2id[x.upper()])
    df_dct = {name:group for name, group in df.groupby("Sentence")}
    sent_ids = list(df_dct.keys())
    train_ids_pool, test_ids = train_test_split(sent_ids=sent_ids, train_fraction=train_fraction, random_state=random_state)
    train_ids = list()
    while True:
        sample, train_ids_pool = sample_and_remove(train_ids_pool, min(step, len(train_ids_pool)), random_state)
        train_ids += sample
        if len(train_ids) >= initial_n:
            yield len(train_ids), len(test_ids), make_train_test_dataset(df.copy(), train_ids, test_ids)
        if len(train_ids_pool) == 0:
            break

def get_successive_datasets(data_sample:str = "English",
                            train_fraction:float = 0.7, 
                            initial_n:int = 20,
                            step:int = 10,
                            random_state:int = 2023, 
                            num_seeds:int = 5):
    df_path = f"/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA/03_PROCESSED/LanguageAdaptation/{data_sample}.csv"
    random.seed(random_state)
    seeds = random.sample(range(1, 10000), num_seeds)
    dict_list = list()
    for seed in tqdm(seeds, total=len(seeds), desc="Train on different seeds"):
        data_generator = train_test_generator(df_path=df_path, train_fraction=train_fraction, initial_n=initial_n, step=step, random_state=seed)
        for len_train, len_test, data in data_generator:
            yield seed, data


def get_stats(lang:str = "English"):
    pattern1 = r'[.,]$'
    pattern2 = r"\s*['’]s$"
    id2label={0: "O",
    1: "B-BEHÖRDE",
    2: "I-BEHÖRDE",
    3: "B-POLITIKER",
    4: "I-POLITIKER",
    5: "B-MEDIEN",
    6: "I-MEDIEN",
    7: "B-PARTEI",
    8: "I-PARTEI",
    9: "B-JOURNALIST",
    10: "I-JOURNALIST"}
    df_list = list()
    gen = get_successive_datasets(data_sample=lang)
    for seed, data in tqdm(gen):
        dataset_size = len(data["train"])
        for sent_idx, (tokens, tags) in enumerate(data["train"].to_pandas().itertuples(index=False, name=None)):
            words = list()
            entities = list()
            beginnings = list()
            for idx in range(len(tags)):
                if tags[idx] != 0:
                    words.append(tokens[idx])
                    entities.append(id2label[tags[idx]])
                    if id2label[tags[idx]].startswith("B"):
                        beginnings.append(len(entities) - 1)
            dct_list = list()
            if len(entities) == 0:
                continue
            elif len(beginnings) == 1:
                dct_list.append({"Entity":words, "Label":entities})
            else:
                for i in range(len(beginnings[:-1])):
                    dct_list.append({"Entity":words[beginnings[i]:beginnings[i+1]], "Label":entities[beginnings[i]:beginnings[i+1]]})
                    last_i = i+1
                if last_i != len(beginnings):
                    dct_list.append({"Entity":words[beginnings[last_i]:], "Label":entities[beginnings[last_i]:]})
            frame = pd.DataFrame(dct_list)
            frame["token_count"] = frame["Entity"].apply(lambda x: len(x))
            frame["Entity"] = frame["Entity"].apply(lambda x: re.sub(pattern2, '', re.sub(pattern1, '', " ".join(x).lower())))
            frame["Label"] = frame["Label"].apply(lambda x: x[0][2:])
            frame["Sent_IDX"] = sent_idx
            frame["Seed"] = seed
            frame["train_size"] = dataset_size
            df_list.append(frame)
    dct_list = list()
    for (seed, trainsize), group in pd.concat(df_list, ignore_index=True).groupby(["Seed", "train_size"]):
        for label in ["BEHÖRDE","JOURNALIST","MEDIEN","PARTEI","POLITIKER"]:
            temp = group[group["Label"] == label]
            avg_tokens = temp["token_count"].mean()
            num_entities = len(temp)
            unique_entities = len(temp["Entity"].unique())
            dct_list.append(
                dict(
                    Label = label,
                    Seed = seed,
                    Trainsize=trainsize,
                    NumEntities = num_entities,
                    UniqueEntities=unique_entities,
                    AvgTokens = avg_tokens,
                )
            )
    df = pd.DataFrame(dct_list)
    df["AvgTokens"] = df["AvgTokens"].fillna(0)
    df["Lang"] = lang
    return df

def add_f1(df, lang:str = "English", model:str="Base"):
    frame = df.copy(deep=True)
    score_path = f"/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA/05_STATS/J_Domain_Adaptation/{lang}{model}.csv"
    scores = pd.read_csv(score_path)
    score_list = list()
    for label, seed, trainsize, numentities, uniqueentities, avgtokens, language in frame.itertuples(index=False, name=None):
        score_list.append(
            scores[(scores["Seed"] == seed) & (scores["Trainsize"] == trainsize)][label].values[0]
        )
        
    frame["Model"] = model
    frame["F1"] = score_list
    return frame

def compute_stats(lang:str = "English"):
    df = get_stats(lang=lang)
    frame1 = add_f1(df=df, lang=lang, model="Base")
    frame2 = add_f1(df=df, lang=lang, model="Large")
    df = pd.concat([frame1, frame2], ignore_index=True)
    #df = frame2
    df.to_csv(f"/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA/05_STATS/J_Domain_Adaptation/{lang}Stats.csv", index=False)
    return df       
            
def run_successive_training(model_checkpoint, 
                            model_size:str = "Base",
                            data_sample:str = "English", 
                            result_path:str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA/05_STATS/J_Domain_Adaptation",
                            train_fraction:float = 0.7, 
                            initial_n:int = 20,
                            step:int = 10,
                            random_state:int = 2023, 
                            num_seeds:int = 5):
    df_path = f"/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA/03_PROCESSED/LanguageAdaptation/{data_sample}.csv"
    random.seed(random_state)
    seeds = random.sample(range(1, 10000), num_seeds)
    dict_list = list()
    for seed in tqdm(seeds, total=len(seeds), desc="Train on different seeds"):
        data_generator = train_test_generator(df_path=df_path, train_fraction=train_fraction, initial_n=initial_n, step=step, random_state=seed)
        for len_train, len_test, data in data_generator:
            trainer = DomainTrainer(model_checkpoint=model_checkpoint)
            scores, best_step, total_steps = trainer.train(data=data)
            del trainer
            torch.cuda.empty_cache()
            dict_list.append(dict(
                Trainset = data_sample,
                Model = model_size,
                Seed = seed,
                Trainsize = len_train,
                Test_size = len_test,
                Best_step = best_step,
                Total_steps = total_steps,
                OVERALL = scores["overall_f1"],
                RECALL = scores["overall_recall"],
                PRECISION = scores["overall_precision"],
                BEHÖRDE = scores["BEHÖRDE"]["f1"] if scores.get("BEHÖRDE") is not None else 0.0,
                JOURNALIST= scores["JOURNALIST"]["f1"] if scores.get("JOURNALIST") is not None else 0.0,
                MEDIEN = scores["MEDIEN"]["f1"] if scores.get("MEDIEN") is not None else 0.0,
                PARTEI = scores["PARTEI"]["f1"] if scores.get("PARTEI") is not None else 0.0,
                POLITIKER = scores["POLITIKER"]["f1"] if scores.get("POLITIKER") is not None else 0.0,
            ))
            df = pd.DataFrame(dict_list)
            df.to_csv(os.path.join(result_path, f"{data_sample}{model_size}.csv"), index=False)
            


    
if __name__ == "__main__":
    train_fraction = 0.7
    initial_n = 20
    step = 10
    random_state = 2023
    data_sample = "Spanish"
    model_size = "Base"
    model_checkpoint = f"/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/XL_Data/{model_size}/triple"
    run_successive_training(model_checkpoint=model_checkpoint, 
                            model_size=model_size, 
                            data_sample=data_sample
                            )
    #lang = "English"
    #compute_stats(lang=lang)    