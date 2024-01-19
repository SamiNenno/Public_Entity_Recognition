import os
import json
from itertools import chain
from glob import glob
from wtpsplit import WtP
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import warnings
warnings.filterwarnings("ignore")

def filter_newspapers(newspaper_path_list):
    def extract_date(file_path):
        date_str = file_path.split('_')[1]
        return datetime.strptime(date_str, '%Y-%m-%d')
    start_date = datetime(2023, 10, 25)
    end_date = datetime(2023, 11, 25)
    return [path for path in newspaper_path_list if start_date <= extract_date(path) <= end_date]

def load_sentencizer():
    wtp = WtP("wtp-canine-s-1l")
    wtp.half().to("cuda")
    return wtp

def get_label_dct(label2id_path):
    with open(label2id_path, "r") as f:
        label2id = json.load(f)
    id2label = {i:l for l, i in label2id.items()}
    return label2id, id2label

def make_classifier(model_path, label2id_path, batch_size:int = 128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label2id, id2label = get_label_dct(label2id_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path, id2label=id2label,
                    label2id=label2id,
                    ignore_mismatched_sizes=True,
                    torch_dtype=torch.float16)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, re="pt")
    classifier = TokenClassificationPipeline(model=model, 
                        tokenizer=tokenizer, 
                        device=device, batch_size=batch_size, 
                        ignore_labels=[])
    return classifier

def make_examples(results, save_checkpoint:int = 50):
    for result_index, res in tqdm(enumerate(results), total=len(results), desc="parse results", leave=False):
        sentence = list()
        labels = list()
        for idx, r in enumerate(res):
            if r["word"].startswith("▁"):
                if idx > 0:
                    sentence.append(current_word)
                    labels.append(current_label) 
                current_word = r["word"][1:]
                current_label = r["entity"]
            else:
                current_word = current_word + r["word"]
        sentence.append(current_word)
        labels.append(current_label)
        df = pd.DataFrame(dict(Entity = sentence, Label=labels))
        yield df
        
def enhance_quality(df_list, min_entities:int = 5):
    return [df for df in df_list if len(df[df["Label"] != "O"]) >= min_entities]

def sentence_loader(filtered_newspaper_path_list):
    for news_day_idx, news_day_path in tqdm(enumerate(filtered_newspaper_path_list), total=len(filtered_newspaper_path_list), desc="Iterate over Newspaperdays"):
        if news_day_idx < 2:
            print(news_day_idx)
            continue
        news_day = pd.read_csv(news_day_path).drop_duplicates(subset="text")
        #news_day = pd.read_csv(news_day_path).drop_duplicates(subset="Text")
        news_day['text'] = news_day['text'].apply(lambda x: None if not isinstance(x, str) else x)
        news_day = news_day.dropna(subset=['text'])
        #news_day['Text'] = news_day['Text'].apply(lambda x: None if not isinstance(x, str) else x)
        #news_day = news_day.dropna(subset=['Text'])
        df_chunks = [pd.DataFrame(chunk, columns=news_day.columns) for chunk in np.array_split(news_day.to_numpy(), len(news_day) // 1001)]
        for chunk_idx, df in tqdm(enumerate(df_chunks), total=len(df_chunks), desc="Iterate over chunks"):
            sentencizer = load_sentencizer()
            sentences = sentencizer.split(df["text"].to_list(), lang_code="de", style="ud", batch_size=128)
            #sentences = sentencizer.split(df["Text"].to_list(), lang_code="de", style="ud", batch_size=128)
            del sentencizer
            torch.cuda.empty_cache()
            yield news_day_idx, chunk_idx, list(chain(*[s for s in sentences]))

def main():
    newspaper_path_list = [path for path in glob("/home/sami/UniversalNewsScraper/Results/*.csv")]
    #filtered_newspaper_path_list = [path for path in glob("/home/sami/Political_Entities/Data/Text/Poltrack/*.csv")]
   
    filtered_newspaper_path_list = filter_newspapers(newspaper_path_list)
    model_path = "/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/CFT/Large"
    label2id_path = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA/04_TRAINING/02_PER/label2id.json"
    destination_path = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA/03_PROCESSED/WeakSample5"
    sentences = sentence_loader(filtered_newspaper_path_list)
    for news_day_idx, chunk_idx, list_of_sentences in sentences:
        classifier = make_classifier(model_path=model_path, label2id_path=label2id_path, batch_size=128)
        results = classifier(list_of_sentences)
        del classifier
        torch.cuda.empty_cache()
        df_list = make_examples(results)
        dfs = enhance_quality(df_list)
        for example_idx, df in enumerate(dfs):
            file_name = f"day_{news_day_idx}_chunk_{chunk_idx}_example_{example_idx}.csv" 
            #file_name = f"Pol_day_{news_day_idx}_chunk_{chunk_idx}_example_{example_idx}.csv"
            df.to_csv(os.path.join(destination_path, file_name), index=False)

if __name__ == "__main__":
    main()
    
