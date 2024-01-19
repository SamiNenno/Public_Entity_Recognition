import os
import json
import torch
import logging
import evaluate
from tqdm import tqdm
from torch.optim import AdamW
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset
from codecarbon import EmissionsTracker
logging.getLogger("codecarbon").disabled = True
logging.getLogger("EmissionsTracker").disabled = True
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, get_scheduler


class CFT():
    def __init__(self, model_checkpoint:str, data_path:str, model_destination_path:str, label2id_path:str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA/04_TRAINING/02_PER/label2id.json", epochs:int = 15, test_size:float=.1, train_batch_size:int = 8, eval_batch_size:int = 16, random_state:int = 2023) -> None:
        self.model_checkpoint = model_checkpoint
        self.model_destination_path = model_destination_path
        self.data_path = data_path
        self.epochs = epochs
        self.test_size = test_size
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.random_state = random_state
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
        
    def make_model_vars(self):
        data = load_from_disk(self.data_path)
        data = data.shuffle(seed=42)
        data = data.train_test_split(test_size=self.test_size, seed=self.random_state)
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
            
    def get_f1(self):
        return self.metric.compute(zero_division=0)['overall_f1']
    
    def train(self):
        logger_prefix = self.data_path.split("/")[-1].split(".")[0]
        best_f1 = 0.0
        self.make_model_vars()
        tracker = EmissionsTracker(output_dir=self.model_destination_path, project_name="CFT")
        tracker.start()
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
                if self.model_vars["pred_counter"] in self.model_vars["pred_steps"]:
                    self.predict(model_vars=self.model_vars, outputs=outputs)
                    f1 = self.get_f1()
                    self.model_vars['lr_scheduler'].step(f1)
                    log = f"Train F1 (epoch: {epoch}/{self.model_vars['epochs']} | step: {self.model_vars['pred_counter']}/{self.model_vars['num_training_steps']}) = {f1} | Best F1 = {best_f1}"
                    if f1 >= best_f1:
                        log += f"\nSave Model to {self.model_destination_path}"
                        self.model_vars['model'].save_pretrained(self.model_destination_path)
                        self.model_vars['tokenizer'].save_pretrained(self.model_destination_path)
                        best_f1 = f1
                    with open(os.path.join(self.model_destination_path, f"{logger_prefix}_logger.txt"), 'a') as file:
                        file.write(log)
                        file.write("\n")
                        log = ""
        tracker.stop()
            
if __name__ == "__main__":
    model_checkpoint = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/XL_Data/Large/triple"
    model_destination_path = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/XL_Data/Base/English"
    data_path = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA/04_TRAINING/02_PER/CleanData/OutDomain.hf"
    trainer = CFT(model_checkpoint=model_checkpoint, model_destination_path=model_destination_path, data_path=data_path, test_size=0.05, eval_batch_size=32, epochs=3)
    trainer.train()
    del trainer 
    torch.cuda.empty_cache()
    
    model_checkpoint = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/XL_Data/Base/English"
    model_destination_path = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/XL_Data/Base/English"
    data_path = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA/04_TRAINING/02_PER/CleanData/Twitter.hf"
    trainer = CFT(model_checkpoint=model_checkpoint, model_destination_path=model_destination_path, data_path=data_path, test_size=0.05, eval_batch_size=32, epochs=3)
    trainer.train()
    del trainer 
    torch.cuda.empty_cache()
    
    model_checkpoint = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/XL_Data/Base/English"
    model_destination_path = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/XL_Data/Base/English"
    data_path = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA/04_TRAINING/02_PER/CleanData/Spanish.hf"
    trainer = CFT(model_checkpoint=model_checkpoint, model_destination_path=model_destination_path, data_path=data_path, test_size=0.05, eval_batch_size=32, epochs=3)
    trainer.train()
    del trainer 
    torch.cuda.empty_cache()
    
    model_checkpoint = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/XL_Data/Base/English"
    model_destination_path = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/XL_Data/Base/English"
    data_path = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA/04_TRAINING/02_PER/CleanData/English.hf"
    trainer = CFT(model_checkpoint=model_checkpoint, model_destination_path=model_destination_path, data_path=data_path, test_size=0.05, eval_batch_size=32, epochs=3)
    trainer.train()
    del trainer 
    torch.cuda.empty_cache()
    
    