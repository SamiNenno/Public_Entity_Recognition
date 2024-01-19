import os
import torch
from datasets import load_from_disk, DatasetDict
from VanillaTrainer import VanillaTrainer

class ShuffledTrainer(VanillaTrainer):
    def __init__(self, data_path: str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA", model_path: str = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS", num_folds: int = 10, random_state: int = 2023, train_batch_size: int = 4, eval_batch_size: int = 16, lr: float = 0.000005, num_train_epochs: int = 5, name_combinations: int = 4) -> None:
        super().__init__(data_path, model_path, num_folds, random_state, train_batch_size, eval_batch_size, lr, num_train_epochs, name_combinations)
    
    def accelerate_traindata(self, train_dataloader, eval_dataloader, model_vars):
        model_vars['train_dataloader'] = model_vars['accelerator'].prepare(train_dataloader)
        model_vars['eval_dataloader'] = model_vars['accelerator'].prepare(eval_dataloader)
        return model_vars
    
    def epoch_iter(self, raw_dataset):
        validation_set = raw_dataset["validation"]
        for train_set in raw_dataset["train"]:
            data_dict = DatasetDict({
                    'train':train_set,
                    'validation':validation_set,
                })
            train_dataloader, eval_dataloader = self.load_tokenized_data(raw_dataset=data_dict)
            yield train_dataloader, eval_dataloader
            
        
    def shuffle_simple_train(self, raw_dataset, model_checkpoint:str):
        score_list = list()
        in_domain_score_list = list()
        out_domain_score_list = list()
        out_lang_score_list = list()
        epochs = self.epoch_iter(raw_dataset=raw_dataset)
        train_dataloader, eval_dataloader = next(iter(epochs))
        model_vars = self.load_training_setup(model_checkpoint=model_checkpoint, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader)
        self.clean_data = self.load_clean_data()
        self.eval_steps = [idx for idx in range(len(model_vars['train_dataloader'])) if idx > 0 and idx % (len(model_vars['train_dataloader'])//self.num_evals) == 0]
        self.eval_steps[-1] = len(model_vars['train_dataloader']) -1
        for epoch in range(model_vars["epochs"]):
                model_vars['model'].train()
                for idx, batch in enumerate(model_vars['train_dataloader']):
                    outputs = model_vars['model'](**batch)
                    loss = outputs.loss
                    model_vars['accelerator'].backward(loss)
                    model_vars['optimizer'].step()
                    model_vars['scheduler'].step()
                    model_vars['optimizer'].zero_grad()
                    model_vars['progressbar'].update(1)
                    if idx in self.eval_steps:
                        self.predict(model_vars=model_vars, outputs=outputs)

                        model_vars['accelerator'].wait_for_everyone()
                        unwrapped_model = model_vars['accelerator'].unwrap_model(model_vars['model'])
                        unwrapped_model.save_pretrained(self.model_path, save_function=model_vars['accelerator'].save)
                        if model_vars['accelerator'].is_main_process:
                            self.tokenizer.save_pretrained(self.model_path)
                        scores = self.get_scores()
                        score_list.append(scores)

                        in_domain_scores = self.predict_on_clean(model_vars['model'], self.clean_data["InDomain"], model_vars['accelerator'])
                        in_domain_score_list.append(in_domain_scores)
                        
                        out_domain_scores = self.predict_on_clean(model_vars['model'], self.clean_data["OutofDomain"], model_vars['accelerator'])
                        out_domain_score_list.append(out_domain_scores)
                        
                        out_lang_scores = self.predict_on_clean(model_vars['model'], self.clean_data["OutofLang"], model_vars['accelerator'])
                        out_lang_score_list.append(out_lang_scores)
                try:
                    train_dataloader, eval_dataloader = next(iter(epochs))
                    model_vars = self.accelerate_traindata(train_dataloader, eval_dataloader, model_vars)
                except StopIteration:
                    pass
                
        scores = self.parse_scores(score_list=score_list)
        in_domain_scores = self.parse_scores(score_list=in_domain_score_list)
        out_domain_scores = self.parse_scores(score_list=out_domain_score_list)
        out_lang_scores = self.parse_scores(score_list=out_lang_score_list)
        clean_scores = dict(
            InDomain = in_domain_scores,
            OutDomain = out_domain_scores,
            OutLang = out_lang_scores
        )
        del train_dataloader 
        del eval_dataloader
        return model_vars, scores, clean_scores
    
    def shuffle_train(self, size:str, kfold:bool = True):
        path = os.path.join(self.data_path, f"05_STATS/B_Shuffled_Scores/{size.capitalize()}")
        train_iter = self.load_train_data(cv=True, shuffled=True, clean=False, pretrain=False)
        for train_data in train_iter:
            model_vars, scores, clean_scores = self.shuffle_simple_train(raw_dataset=train_data, model_checkpoint=self.get_model_checkpoint(size=size))
            if not kfold:
                return model_vars, scores
            else:
                self.score_list.append(scores)
                self.in_domain_score_list.append(clean_scores["InDomain"])
                self.out_domain_score_list.append(clean_scores["OutDomain"])
                self.out_lang_score_list.append(clean_scores["OutLang"])
                self.save_scores(score_list=self.score_list, path=path, test_type="weak")
                self.save_scores(score_list=self.in_domain_score_list, path=path, test_type="InDomain")
                self.save_scores(score_list=self.out_domain_score_list, path=path, test_type="OutDomain")
                self.save_scores(score_list=self.out_lang_score_list, path=path, test_type="OutLang")
                model_path = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/02_PER/ShuffleExperiment" #!Remove
                model_vars['model'].save_pretrained(model_path) #!Remove
                model_vars['tokenizer'].save_pretrained(model_path) #!Remove
                del model_vars
                torch.cuda.empty_cache()