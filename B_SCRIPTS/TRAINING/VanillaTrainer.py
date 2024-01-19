import os
import logging
import pandas as pd
import torch
from codecarbon import EmissionsTracker
logging.getLogger("codecarbon").disabled = True
logging.getLogger("EmissionsTracker").disabled = True
from BaseTrainer import BaseTrainer
from transformers import AutoTokenizer, DataCollatorForTokenClassification

class VanillaTrainer(BaseTrainer):
    def __init__(self, data_path: str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA", model_path: str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS", sample: str = "v1", num_folds: int = 5, random_state: int = 2023, train_batch_size: int = 8, eval_batch_size: int = 16, lr: float = 0.000005, num_train_epochs: int = 5, name_combinations: int = 4) -> None:
        super().__init__(data_path, model_path, sample, num_folds, random_state, train_batch_size, eval_batch_size, lr, num_train_epochs, name_combinations)
        self.score_list = list()
        self.in_domain_score_list = list()
        self.out_domain_score_list = list()
        self.twitter_score_list = list()
        self.en_score_list = list()
        self.esp_score_list = list()
        self.tokenizer = AutoTokenizer.from_pretrained("/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/XL_Data/Base/simple")
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        
    def vanilla_simple_train(self, raw_dataset, model_checkpoint:str):
        score_list = list()
        in_domain_score_list = list()
        out_domain_score_list = list()
        twitter_score_list = list()
        en_score_list = list()
        esp_score_list = list()
        train_dataloader, eval_dataloader = self.load_tokenized_data(raw_dataset=raw_dataset)
        model_vars = self.load_training_setup(model_checkpoint=model_checkpoint, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader)
        self.clean_data = self.load_clean_data()
        step_size = len(model_vars['train_dataloader'])//9
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

                        model_vars['accelerator'].wait_for_everyone()
                        unwrapped_model = model_vars['accelerator'].unwrap_model(model_vars['model'])
                        unwrapped_model.save_pretrained(self.model_path, save_function=model_vars['accelerator'].save)
                        if model_vars['accelerator'].is_main_process:
                            self.tokenizer.save_pretrained(self.model_path)
                        scores = self.get_scores()
                        model_vars['scheduler'].step(scores["overall_f1"])
                        score_list.append(scores)
                        print(f"Weak F1 ({epoch}/{counter}): {scores['overall_f1']}")
                        in_domain_scores = self.predict_on_clean(model_vars['model'], self.clean_data["InDomain"], model_vars['accelerator'])
                        in_domain_score_list.append(in_domain_scores)
                        
                        out_domain_scores = self.predict_on_clean(model_vars['model'], self.clean_data["OutofDomain"], model_vars['accelerator'])
                        out_domain_score_list.append(out_domain_scores)
                        
                        twitter_scores = self.predict_on_clean(model_vars['model'], self.clean_data["Twitter"], model_vars['accelerator'])
                        twitter_score_list.append(twitter_scores)
                        
                        en_scores = self.predict_on_clean(model_vars['model'], self.clean_data["English"], model_vars['accelerator'])
                        en_score_list.append(en_scores)
                        
                        esp_scores = self.predict_on_clean(model_vars['model'], self.clean_data["Spanish"], model_vars['accelerator'])
                        esp_score_list.append(esp_scores)
                
                
        scores = self.parse_scores(score_list=score_list)
        in_domain_scores = self.parse_scores(score_list=in_domain_score_list)
        out_domain_scores = self.parse_scores(score_list=out_domain_score_list)
        twitter_lang_scores = self.parse_scores(score_list=twitter_score_list)
        en_lang_scores = self.parse_scores(score_list=en_score_list)
        esp_lang_scores = self.parse_scores(score_list=esp_score_list)
        clean_scores = dict(
            InDomain = in_domain_scores,
            OutDomain = out_domain_scores,
            Twitter = twitter_lang_scores,
            English=en_lang_scores,
            Spanish=esp_lang_scores
        )
        del train_dataloader 
        del eval_dataloader
        return model_vars, scores, clean_scores
                    
    def vanilla_train(self, size:str):
        path = os.path.join(self.data_path, f"05_STATS/A_Vanilla_Scores")
        train_iter = self.load_train_data(cv=True, shuffled=False, clean=False, pretrain=False)
        for counter, train_data in enumerate(train_iter):
            #model_checkpoint = f"/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/XL_Data/{size}/simple"
            model_vars, scores, clean_scores = self.vanilla_simple_train(raw_dataset=train_data, model_checkpoint=self.get_model_checkpoint(size=size))
            #model_vars, scores, clean_scores = self.vanilla_simple_train(raw_dataset=train_data, model_checkpoint=model_checkpoint)
            del model_vars
            torch.cuda.empty_cache()
            self.score_list.append(scores)
            self.in_domain_score_list.append(clean_scores["InDomain"])
            self.out_domain_score_list.append(clean_scores["OutDomain"])
            self.twitter_score_list.append(clean_scores["Twitter"])
            self.en_score_list.append(clean_scores["English"])
            self.esp_score_list.append(clean_scores["Spanish"])
            self.save_scores(score_list=self.score_list, path=path, test_type="weak", size=size.capitalize())
            self.save_scores(score_list=self.in_domain_score_list, path=path, test_type="InDomain", size=size.capitalize())
            self.save_scores(score_list=self.out_domain_score_list, path=path, test_type="OutDomain", size=size.capitalize())
            self.save_scores(score_list=self.twitter_score_list, path=path, test_type="Twitter", size=size.capitalize())
            self.save_scores(score_list=self.en_score_list, path=path, test_type="English", size=size.capitalize())
            self.save_scores(score_list=self.esp_score_list, path=path, test_type="Spanish", size=size.capitalize())
    
    def save_scores(self, score_list:list,  path:str, test_type:str = "weak", size:str = "Base"):
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
        #df.to_csv(os.path.join(path, f"vanilla_v5_{test_type}_{size}_scores.csv"), index=False)
        df.to_csv(os.path.join(path, f"vanilla_{self.sample}_{test_type}_{size}_scores.csv"), index=False)


if __name__ == "__main__":
    size = "Large"
    sample = "v3"
    emission_name = "VanillaTrain"
    trainer = VanillaTrainer(sample=sample, num_train_epochs=5)
    emission_path = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA/05_STATS/F_Emissions"
    tracker = EmissionsTracker(output_dir=emission_path, project_name=emission_name)
    tracker.start()
    trainer.vanilla_train(size=size)
    tracker.stop()
    print(f"Finished: sample - {sample} | model - {size}")
    del trainer
    torch.cuda.empty_cache()
    