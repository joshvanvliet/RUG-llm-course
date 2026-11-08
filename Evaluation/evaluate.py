import json
import sys

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import hamming_loss, jaccard_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from Data.dataset_maker import EMOTIONS
from Data.paths import PROJECT_ROOT
from Models.gemma_4B import load_gemma_4B
from Training.train import MODEL_ID, load_splits, model_dir, parse_answer

LOGS_DIR = PROJECT_ROOT / "logs"
SEEDS = [0, 1]
BATCH_SIZE = 16
# (trained on, evaluated on)
CONDITIONS = [("EN", "EN"), ("EN", "NL"), ("NL", "NL"), ("BI", "EN"), ("BI", "NL")]


# Greedy generation for a batch of chats, returns the new tokens per chat and,
# when choices is given, the probability of the first choice against the second at every step
def generate(model, tokenizer, chats, choices=None):
    vocab = tokenizer.get_vocab()
    stop_ids = [tokenizer.eos_token_id] + [vocab[t] for t in ("<turn|>", "<|im_end|>") if t in vocab]
    texts = [tokenizer.apply_chat_template(chat, add_generation_prompt=True, enable_thinking=False, tokenize=False) for chat in chats]
    tokenizer.padding_side = "left"
    inputs = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=64, do_sample=False, eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True, output_scores=choices is not None
        )
    tokens = outputs.sequences[:, inputs["input_ids"].shape[1]:].tolist()
    if choices is None:
        return tokens, None
    return tokens, torch.stack([torch.softmax(step[:, choices].float(), dim=-1)[:, 0] for step in outputs.scores], dim=1).tolist()


# Probability of yes for each emotion, read at the generated yes/no token
def predict_probabilities(model, tokenizer, data):
    yes = tokenizer.encode(" yes", add_special_tokens=False)[0]
    no = tokenizer.encode(" no", add_special_tokens=False)[0]
    predictions = []
    for batch in tqdm(data.iter(batch_size=BATCH_SIZE), total=-(-len(data) // BATCH_SIZE)):
        tokens, choice_probabilities = generate(model, tokenizer, batch["prompt"], [yes, no])
        for row, row_probabilities in zip(tokens, choice_probabilities):
            probabilities, line_start = {}, 0
            for step, token in enumerate(row):
                emotion = tokenizer.decode(row[line_start:step]).split(":")[0].strip()
                if token in (yes, no) and emotion in EMOTIONS:
                    probabilities[emotion] = row_probabilities[step]
                if "\n" in tokenizer.decode([token]):
                    line_start = step + 1
            predictions.append(probabilities)
    return predictions


def compute_metrics(actual, predicted):
    binarizer = MultiLabelBinarizer(classes=range(1, 9))
    y_true, y_pred = binarizer.fit_transform(actual), binarizer.transform(predicted)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    micro = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    return {
        "micro_f1": micro[2],
        "jaccard": jaccard_score(y_true, y_pred, average="samples", zero_division=0),
        "macro_f1": f1.mean(),
        "micro_precision": micro[0],
        "micro_recall": micro[1],
        "hamming_loss": hamming_loss(y_true, y_pred),
        "exact_match": (y_true == y_pred).all(axis=1).mean(),
        "labels_per_line": y_pred.sum(axis=1).mean(),
        "per_emotion": {e: [precision[i], recall[i], f1[i], int(support[i])] for i, e in enumerate(EMOTIONS)},
    }


def rows_path(seed, trained_on, language, split):
    return LOGS_DIR / f"probabilities_seed{seed}_{trained_on}_on_{language}_{split}.json"


def load_rows(seed, trained_on, language, split):
    with open(rows_path(seed, trained_on, language, split)) as file:
        return json.load(file)


def score(rows, threshold):
    predicted = [[n for n, e in enumerate(EMOTIONS, start=1) if row["probabilities"].get(e, 0) >= threshold] for row in rows]
    return compute_metrics([row["labels"] for row in rows], predicted)


# Threshold with the best micro-F1 on the develop set of the language(s) the model was trained on
def pick_threshold(seed, trained_on):
    languages = ["EN", "NL"] if trained_on == "BI" else [trained_on]
    rows = sum((load_rows(seed, trained_on, language, "develop") for language in languages), [])
    return max(np.arange(0.05, 1, 0.05).round(2), key=lambda t: score(rows, t)["micro_f1"])


# Only generates what is not saved yet
def predict_missing(seed, split, data):
    for trained_on in ["EN", "NL", "BI"]:
        languages = [l for t, l in CONDITIONS if t == trained_on and not rows_path(seed, t, l, split).exists()]
        if not languages:
            continue
        model, processor = load_gemma_4B(MODEL_ID)
        model = PeftModel.from_pretrained(model, model_dir(trained_on, seed)).merge_and_unload().eval()
        for language in languages:
            probabilities = predict_probabilities(model, processor.tokenizer, data[language])
            gold = [parse_answer(example["completion"][0]["content"]) for example in data[language]]
            with open(rows_path(seed, trained_on, language, split), "w") as file:
                json.dump([{"labels": g, "probabilities": p} for g, p in zip(gold, probabilities)], file)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "develop"
    index = {"develop": 1, "test": 2}[split]
    english, dutch, _ = load_splits()

    results = {}
    for seed in SEEDS:
        LOGS_DIR.mkdir(exist_ok=True)
        predict_missing(seed, split, {"EN": english[index], "NL": dutch[index]})
        for trained_on, language in CONDITIONS:
            threshold = pick_threshold(seed, trained_on)
            metrics = score(load_rows(seed, trained_on, language, split), threshold)
            results.setdefault(f"{trained_on} on {language}", []).append(metrics | {"seed": seed, "threshold": threshold})

    for name, runs in results.items():
        print(name, "  ".join(
            f"{m} {np.mean([r[m] for r in runs]):.3f} ± {np.std([r[m] for r in runs], ddof=1):.3f}"
            for m in ["micro_f1", "jaccard", "macro_f1", "hamming_loss"]
        ))
    (PROJECT_ROOT / "results").mkdir(exist_ok=True)
    with open(PROJECT_ROOT / "results" / f"lora_{split}.json", "w") as file:
        json.dump(results, file, indent=1)
