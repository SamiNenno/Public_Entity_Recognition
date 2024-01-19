import os
import json
import spacy
import evaluate
import pandas as pd
from glob import glob
from spacy.tokens import Doc
from skweak.base import CombinedAnnotator
from skweak.doclevel import DocumentHistoryAnnotator
from skweak.aggregation import MajorityVoter
from skweak.voting import SequentialMajorityVoter
from skweak.gazetteers import GazetteerAnnotator, Trie
from skweak.heuristics import FunctionAnnotator

def make_gazeteers(gazeteer_path, party_hyphen_path, combined):
    with open(party_hyphen_path, 'r') as f:
        hyph_dict = json.load(f)
    def party_hyphen(doc):
        parties = ['SPD-', 'FDP-', 'CDU-', 'CSU-', 'CDU/CSU-', 'Grünen-','Linken-', 'AfD-']
        for tok in doc:
            for party in parties:
                if party in tok.text:
                    checked = False
                    for politiker in hyph_dict["POLITIKER"]:
                        if politiker in tok.text:
                            checked = True
                            yield tok.i, tok.i, "POLITIKER"
                    if not checked:
                        for partei in hyph_dict["PARTEI"]:
                            if partei in tok.text:
                                checked = True
                                yield tok.i, tok.i, "PARTEI"
                    if not checked:
                        yield tok.i, tok.i, "PARTY_HYPHEN"
    labels = set()
    party_hyphen = FunctionAnnotator("party_hyphen", party_hyphen)
    combined.add_annotator(party_hyphen)
    labels.add("PARTY_HYPHEN")
    for json_path in glob(os.path.join(gazeteer_path, '*.json')):
        with open(json_path, 'r', encoding='UTF-8') as f:
            dct = json.load(f)
        name = json_path.split('/')[-1].split('.')[0]
        label = name.split('_')[0].upper()
        labels.add(label)
        if label in ['POLITIKER', 'JOURNALIST']:
            entities = [(n[0], n[1]) for n in dct[name] if len(n) == 2]
        else:
            entities = [tuple(entity.split(' ')) for entity in dct[name]]
        trie = Trie(entities)
        lf = GazetteerAnnotator(name, {label:trie})
        combined.add_annotator(lf)
    maj_voter = MajorityVoter("doclevel_voter", ["POLITIKER", "JOURNALIST"], initial_weights={"doc_history":0.0})
    doc_history= DocumentHistoryAnnotator("doc_history", "doclevel_voter",  ["POLITIKER", "JOURNALIST"])
    combined.add_annotator(maj_voter)
    combined.add_annotator(doc_history)
    return combined, list(labels)

def make_pretokenized_doc(folder_path, nlp):
    dict_list = list()
    path_list = glob(os.path.join(folder_path, "*.csv"))
    for path in path_list:
        name = path.split('/')[-1].split('.')[0]
        df = pd.read_csv(path)
        df = df[df["Entity"].notna()]
        tokens = df["Entity"].to_list()
        tags = df["Label"].to_list()
        doc = Doc(nlp.vocab, tokens)
        dct = {"tokens":tokens, "tags":tags, "doc":doc, "name":name}
        dict_list.append(dct)
    return dict_list

def annotate(data_path:str, spacy_checkpoint:str = "de_core_news_md"):
    folder_path = os.path.join(data_path, "03_PROCESSED/Manual_Labeled")
    gazeteer_path = os.path.join(os.path.join(data_path, '01_UTIL'), 'Gazeteers')
    party_hyphen_path = os.path.join(os.path.join(data_path, '01_UTIL'), 'Party_Hyphen.json')
    nlp = spacy.load(spacy_checkpoint)
    combined = CombinedAnnotator()
    combined, labels = make_gazeteers(gazeteer_path, party_hyphen_path,combined)
    doc_list = make_pretokenized_doc(folder_path, nlp)
    docs = [dct["doc"] for dct in doc_list]
    docs = list(combined.pipe(docs))
    voter = SequentialMajorityVoter("voter", labels=labels)
    docs = list(voter.pipe(docs))
    for doc in docs:
        doc.ents = doc.spans['voter']
    for idx, doc in enumerate(docs):
        val_tokens = list()
        preds = list()
        for token in doc:
            label = token.ent_iob_ if token.ent_iob_ == 'O' else token.ent_iob_ + '-' + token.ent_type_
            tok = token.text
            val_tokens.append(tok)
            preds.append(label)
        doc_list[idx]["doc"] = doc
        doc_list[idx]["val_tokens"] = val_tokens
        doc_list[idx]["preds"] = preds
    df_list = [pd.DataFrame(dct) for dct in doc_list]
    for df in df_list:
        df["tags"] = df["tags"].apply(lambda x: x.upper())
        df["preds"] = df["preds"].apply(lambda x: "O" if x == "B-PARTY_HYPHEN" else x)
    return df_list

def parse_scores(results):
    dct = {"precision":[], "recall":[], "f1":[], "category":[]}
    for key, value in results.items():
        if key == "overall_precision":
            dct["precision"].append(value)
            dct["category"].append("OVERALL")
        elif key == "overall_recall":
            dct["recall"].append(value)
        elif key == "overall_f1":
            dct["f1"].append(value)
        elif key == "overall_accuracy":
            continue
        else:
            dct["precision"].append(value["precision"])
            dct["recall"].append(value["recall"])
            dct["f1"].append(value["f1"])
            dct["category"].append(key)
    return pd.DataFrame(dct)

def compute_scores(df_list):
    seqeval = evaluate.load('seqeval')
    result_list = list()
    for idx in range(len(df_list)):
        name = df_list[idx]["name"][0]
        results = seqeval.compute(predictions=[df_list[idx]["preds"].to_list()], references=[df_list[idx]["tags"].to_list()])
        results = parse_scores(results)
        results["label_type"] = name
        result_list.append(results)
    df = pd.concat(result_list, ignore_index=True)
    return df

def compute_weak_baseline(data_path:str):
    df_list = annotate(data_path)
    scores = compute_scores(df_list)
    scores.to_csv(os.path.join(data_path, "05_STATS/D_Baseline_Scores/WeakBaseline.csv"), index=False)

if __name__ == "__main__":
    data_path = "/home/sami/POLITICAL_ENTITY_RECOGNITION/A_DATA"
    compute_weak_baseline(data_path)
