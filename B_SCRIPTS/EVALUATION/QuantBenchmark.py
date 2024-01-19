import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, BitsAndBytesConfig


def load_model(model_checkpoint:str, quantization:str = "fp32"):
    if quantization == "fp32":
        model = AutoModelForTokenClassification.from_pretrained(model_checkpoint).to("cuda")
    elif quantization == "fp16":
        model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, torch_dtype=torch.float16).to("cuda")
    elif quantization == "int8":
        model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, load_in_8bit=True, device_map="auto")
    elif quantization == "int4":
        model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, load_in_4bit=True, device_map="auto")
    else:
        raise KeyError(f"{quantization} is not supported!")
    return model.eval()
        
def make_benchmark_data(model_checkpoint:str, batch_size:int = 16):
    def tokenization(example):
        return tokenizer(example["Token"], truncation=True, padding=True, max_length=256, return_tensors="pt")
    test_example = '''In Hessen haben sich CDU und SPD am späten Mittwochabend auf einen Koalitionsvertrag zur Bildung einer neuen Landesregierung geeinigt. Schwerpunkte – unter der Führung von Ministerpräsident Boris Rhein (CDU) – setzen die künftigen Regierungsparteien in der Bildungspolitik, bei der „Stärkung des Rechtsstaats“, bei der Schaffung gleichwertiger Lebensverhältnisse in Stadt und Land sowie bei der „Begrenzung der irregulären Migration“. In der öffentlichen Verwaltung, in staatlichen und öffentlich-rechtlichen Institutionen wie Schulen und Universitäten soll auf das „Gendern der Sprache mit Sonderzeichen“ verzichtet werden. Das Papier, um das die Verhandlungsdelegationen der beiden Parteien bis zuletzt gerungen haben, umfasst fast 200 Seiten. Der Zeitplan war ins Rutschen gekommen, weil Hessens SPD-Landesvorsitzende Nancy Faeser in ihrer Rolle als Bundesinnenministerin bei den Beratungen über den Haushalt und bei den Bundestagsdebatten in Berlin unabkömmlich war. An diesem Donnerstag, nur zwei Tage vor den abschließenden Beratungen der Gremien, wurde der Vertragsentwurf öffentlich vorgestellt. Vor allem beim Ausbau der Kitas und bei der der'''
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    dataset = Dataset.from_pandas(pd.DataFrame({"Token":[test_example] * (batch_size * 500)})).with_format("torch", device="cuda") #!Change
    dataset = dataset.map(tokenization, batched=True, remove_columns=dataset.column_names)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

size = "Large"
model_checkpoint = f"/home/sami/POLITICAL_ENTITY_RECOGNITION/C_MODELS/XL_Data/{size}/triple"
quantization = "int4"
warm_up_iterations = 10
batch_size = 16
duration = list()
data = make_benchmark_data(model_checkpoint=model_checkpoint,batch_size=batch_size)
model = load_model(model_checkpoint=model_checkpoint, quantization=quantization)
for idx, batch in tqdm(enumerate(data), total=len(data)):
    if idx < warm_up_iterations:
        outputs = model(**batch)
    else:
        start = time.time()
        outputs = model(**batch)
        end = time.time()
        duration.append(end - start)
    
duration = np.array(duration)
mean_duration = np.round(np.mean(duration)/batch_size, 8)
footprint =round(model.get_memory_footprint()/1e6,2)
print(f"Model: {size}/ Quantization: {quantization}/ Batchsize: {batch_size}/ Duration: {mean_duration}/ Memory: {footprint}MB/ vRAM: ")
#Model: Base/ Quantization: fp32/ Batchsize: 16/ Duration: 0.00049345/ Memory: 1109.85MB/ vRAM: 7824MiB
#Model: Base/ Quantization: fp16/ Batchsize: 16/ Duration: 0.00049759/ Memory: 554.93MB/ vRAM: 4494MiB
#Model: Base/ Quantization: int8/ Batchsize: 16/ Duration: 0.00426438/ Memory: 470.0MB/ vRAM: 5098MiB
#Model: Base/ Quantization: int4/ Batchsize: 16/ Duration: 0.00121438/ Memory: 427.53MB/ vRAM: 5942MiB
#Model: Large/ Quantization: fp32/ Batchsize: 16/ Duration: 0.00096408/ Memory: 2235.42MB/ vRAM: 18488MiB
#Model: Large/ Quantization: fp16/ Batchsize: 16/ Duration: 0.00096838/ Memory: 1117.71MB/ vRAM: 10478MiB
#Model: Large/ Quantization: int8/ Batchsize: 16/ Duration: 0.0092402/ Memory: 815.72MB/ vRAM: 11222MiB
#Model: Large/ Quantization: int4/ Batchsize: 16/ Duration: 0.00271956/ Memory: 664.73MB/ vRAM: 13812MiB