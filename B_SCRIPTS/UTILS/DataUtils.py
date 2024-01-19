import os
import json
from Path_finder import DATA_PATH

def load_label_dicts():
    path = os.path.join(DATA_PATH, "04_TRAINING/02_PER")
    with open(os.path.join(path, "id2label.json"), "r") as f:
        id2label = json.load(f)
        id2label = {int(k):v for k,v in id2label.items()}
    with open(os.path.join(path, "label2id.json"), "r") as f:
        label2id = json.load(f)
    return id2label, label2id
