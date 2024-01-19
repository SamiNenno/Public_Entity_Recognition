import torch
import numpy as np
import pandas as pd
from scipy.stats import mode
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers_interpret import TokenClassificationExplainer


class SentenceExplainer():
    def __init__(self) -> None:
        self.label2id = {"B-POLITIKER":0, 
                         "B-MEDIEN":1, 
                         "B-BEHÖRDE":2, 
                         "B-PARTEI":3, 
                         "B-JOURNALIST":4,
                         "I-POLITIKER":5, 
                         "I-MEDIEN":6, 
                         "I-BEHÖRDE":7, 
                         "I-PARTEI":8, 
                         "I-JOURNALIST":9,
                         "O":10}
        self.id2label = {v:k for k,v in self.label2id.items()}
        
    def merge_subwords(self, subwords):
        def remove_marker(word):
            return word.replace('▁', "")

        def is_proper_word(word):
            return word.startswith('▁')

        def is_no_word(word):
            return word == '▁'

        word_list = list()
        running_word = ""
        for word in subwords:
            if word == '<s>':
                continue
            elif is_no_word(word):
                continue
            elif not isinstance(word, str):
                continue
            elif is_proper_word(word):
                if running_word == "":
                    running_word += remove_marker(word)
                else:
                    word_list.append(running_word)
                    running_word = remove_marker(word)
            else:
                running_word += remove_marker(word)
        return word_list

    def turn_to_sentence(self, tokenizer, input_ids:list):
        subwords = [tokenizer.convert_ids_to_tokens(token_id) for token_id in input_ids]
        sentence = " ".join(self.merge_subwords(subwords))
        return sentence
    
    def merge_attribution_scores(self, attr_scores, entity):
        def remove_marker(word):
            return word.replace('▁', "")

        def is_proper_word(word):
            return word.startswith('▁')

        def is_no_word(word):
            return word == '▁'

        word_list = list()
        score_list = list()
        running_word = ""
        running_scores = list()
        for word, score in attr_scores:
            if word == '<s>':
                continue
            elif is_no_word(word):
                continue
            elif not isinstance(word, str):
                continue
            elif is_proper_word(word):
                if running_word == "":
                    running_word += remove_marker(word)
                    running_scores.append(score)
                else:
                    word_list.append(running_word)
                    score_list.append(np.array(running_scores).mean(axis=0))
                    running_word = remove_marker(word)
                    running_scores = [score]
            else:
                running_word += remove_marker(word)
                running_scores.append(score)
        return pd.DataFrame({"words":word_list, entity:score_list})
    
    def merge_entity_scores(self, df, predictions):
        def remove_marker(word):
            return word.replace('▁', "")

        def is_proper_word(word):
            return word.startswith('▁')

        def is_no_word(word):
            return word == '▁'

        word_list = list()
        score_list = list()
        label_list = list()
        running_word = ""
        running_scores = list()
        running_label = list()
        for idx, word in enumerate(df.columns[1:]):
            label = predictions[idx]
            #if label == "O":
            #    continue
            label = self.label2id[label]#[2:]
            if word == '<s>':
                continue
            elif is_no_word(word):
                continue
            elif not isinstance(word, str):
                continue
            elif is_proper_word(word):
                if running_word == "":
                    running_word += remove_marker(word)
                    running_scores.append(df.iloc[:,idx+1].to_list())
                    running_label.append(label)
                else:
                    word_list.append(running_word)
                    score_list.append(np.array(running_scores).mean(axis=0))
                    label_list.append(mode(running_label).mode)
                    running_word = remove_marker(word)
                    running_scores = [df.iloc[:,idx+1].to_list()]
                    running_label = [label]
            else:
                running_word += remove_marker(word)
                running_scores.append(df.iloc[:,idx+1].to_list())
                running_label.append(label)
        word_list.append(running_word)
        score_list.append(np.array(running_scores).mean(axis=0))
        label_list.append(mode(running_label).mode)
        try:
            df = pd.concat([pd.DataFrame({word:score.tolist()}) for word, score in zip(word_list, score_list)]\
                + [pd.DataFrame({"words":df["words"].to_list()})], axis=1)
        except ValueError as e:
            print("ERROR:")
            print(word_list)
            print(score_list)
            for word, score in zip(word_list, score_list):
                print(word, score)
            print(e)
            return pd.DataFrame()
        df = df.T
        df.columns = df.iloc[-1]
        df = df.iloc[:-1]
        df = df.reset_index().rename(columns={"index":"words"})
        df["predictions"] = label_list
        df["predictions"] = df["predictions"].apply(lambda x: self.id2label[x])
        return df 
    
    def explain_sentence(self, input_ids:list, ner_explainer, tokenizer):
        self.sample_text = self.turn_to_sentence(tokenizer, input_ids)
        if "hfnfjotbn" in self.sample_text or "bluvfmm" in self.sample_text:
            return pd.DataFrame(dict(Error=["hfnfjotbn"]))
        word_attributions = ner_explainer(self.sample_text[:1400], ignored_labels=['O']) #Must be shortened otherwise error for long sentences.
        df_list = list()
        predictions = list()
        for entity in word_attributions.keys():
            prediction = word_attributions[entity]["label"]
            attr_scores = word_attributions[entity]["attribution_scores"]
            df = self.merge_attribution_scores(attr_scores, entity)
            df_list.append(df)
            predictions.append(prediction)
        if len(predictions) == 0:
            return None
        df = df_list[0]
        for frame in df_list[1:]:
            df = pd.concat([df, frame.iloc[:, -1]], axis=1)
        df = self.merge_entity_scores(df, predictions=predictions)
        return df


class DataLoaderExplainer(SentenceExplainer):
    def __init__(self, model, tokenizer) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.ner_explainer = TokenClassificationExplainer(self.model, self.tokenizer)
        self.tokenizer = tokenizer
        
    def explain_dataset(self, dataloader, raise_oom:bool = False):
        counter = 0
        for batch in tqdm(dataloader, total=len(dataloader), desc="Explain dataset", leave=False):
            for input_ids in tqdm(batch["input_ids"], total=len(batch["input_ids"]), desc="Explain batch", leave=False):
                input_ids = input_ids.tolist()
                try:
                    explanation = self.explain_sentence(input_ids=input_ids, 
                                                    ner_explainer=self.ner_explainer, 
                                                    tokenizer=self.tokenizer)
                    if explanation is not None:
                        yield explanation
                    else:
                        yield pd.DataFrame()
                except RuntimeError as e:
                    if not raise_oom:
                        print(f'| WARNING ran out of memory: {str(e)}')
                        print(f"For the following sentence:\n{self.sample_text}")
                        del self.ner_explainer
                        torch.cuda.empty_cache()
                        self.ner_explainer = TokenClassificationExplainer(self.model, self.tokenizer)
                        yield pd.DataFrame(dict(Error=["Error"]))
                    else:
                        raise e

                
                
    
if __name__ == "__main__":
    from CleanTrainer import CleanTrainer
    checkpoint = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/02_PER/Dirty/Vanilla/BaseWeakData2"
    model = AutoModelForTokenClassification.from_pretrained(checkpoint).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    ct = CleanTrainer(sample="confident")
    cv_train_data, out_domain_data, twitter_data, en_data, esp_data = ct.load_clean_data()
    train_dataloader, eval_dataloader = next(iter(cv_train_data))
    print(twitter_data)
    explainer = DataLoaderExplainer(model=model, tokenizer=tokenizer)
    explanations = explainer.explain_dataset(dataloader=en_data)
    for ex in explanations:
        print(ex)