import json
import random
import sys

from tqdm import tqdm

from Data.dataset_maker import EMOTIONS, load_dutch_splits
from Data.paths import PROJECT_ROOT
from Evaluation.evaluate import BATCH_SIZE, compute_metrics, generate
from Models.gemma_4B import load_gemma_4B
from Models.qwen_4B import load_qwen_4B

MODELS = {"gemma": (load_gemma_4B, "google/gemma-4-E4B-it"), "qwen": (load_qwen_4B, "Qwen/Qwen3.5-4B")}
LABELS = [emotion.lower() for emotion in EMOTIONS]
INSTRUCTION = (
    "Classify the emotions expressed in this subtitle line from a movie.\n"
    f"Allowed labels: {', '.join(LABELS)}.\n"
    "A line can express one emotion or several.\n"
    "Answer with only a JSON list of the labels that apply, written exactly as above."
)
# We only used this line on the develop set to check how sensitive the results are to specific wording
ALTERNATIVE = INSTRUCTION.replace("A line can express one emotion or several.", "Every line expresses at least one emotion, often several.")


def to_json(labels):
    return json.dumps([LABELS[n - 1] for n in sorted(labels)])


# Same English instruction for both languages with examples in the evaluated language
def build_prompt(sentence, examples):
    shots = "".join(f"Line: {e['sentence']}\nAnswer: {to_json(e['labels'])}\n\n" for e in examples)
    return f"{INSTRUCTION}\n\n{shots}Line: {sentence}\nAnswer:"


# Only case and whitespace are normalized, and None when the output isn't a JSON list
def parse_output(text):
    text = text.replace("```json", "").replace("```", "")
    try:
        items = json.loads(text[text.index("["):text.rindex("]") + 1])
    except ValueError:
        return None
    if not isinstance(items, list):
        return None
    names = {" ".join(item.split()).lower() for item in items if isinstance(item, str)}
    return [n for n, label in enumerate(LABELS, start=1) if label in names]


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "gemma"
    split = sys.argv[2] if len(sys.argv) > 2 else "develop"
    index = {"develop": 1, "test": 2}[split]
    tag = f"{name}_{split}"
    if len(sys.argv) > 3 and sys.argv[3] == "alternative":
        INSTRUCTION, tag = ALTERNATIVE, tag + "_alternative"
    english, dutch = load_dutch_splits()

    # Five fixed training pairs whose English and Dutch labels agree
    agree = [i for i in range(len(english[0])) if sorted(english[0][i]["labels"]) == sorted(dutch[0][i]["labels"])]
    demo_rows = random.Random(0).sample(agree, 5)

    load, model_id = MODELS[name]
    model, processor = load(model_id)
    results, outputs = {"demo_ids": [english[0][i]["id"] for i in demo_rows]}, []
    for shots in [0, 5]:
        for language, splits in [("EN", english), ("NL", dutch)]:
            examples = [splits[0][i] for i in demo_rows][:shots]
            predicted = []
            for batch in tqdm(splits[index].iter(batch_size=BATCH_SIZE), desc=f"{shots}-shot {language}"):
                chats = [[{"role": "user", "content": build_prompt(sentence, examples)}] for sentence in batch["sentence"]]
                tokens, _ = generate(model, processor.tokenizer, chats)
                for i, row in enumerate(tokens):
                    text = processor.tokenizer.decode(row, skip_special_tokens=True)
                    predicted.append(parse_output(text))
                    outputs.append({"shots": shots, "language": language, "id": batch["id"][i],
                                    "sentence": batch["sentence"][i], "labels": batch["labels"][i], "output": text})

            metrics = compute_metrics(splits[index]["labels"], [p or [] for p in predicted])
            results[f"{shots}-shot {language}"] = metrics | {"malformed_rate": predicted.count(None) / len(predicted)}
            print(f"{shots}-shot {language}: micro-F1 {metrics['micro_f1']:.3f}  Jaccard {metrics['jaccard']:.3f}  "
                  f"malformed {predicted.count(None) / len(predicted):.1%}", flush=True)

    (PROJECT_ROOT / "results").mkdir(exist_ok=True)
    with open(PROJECT_ROOT / "results" / f"prompting_{tag}.json", "w") as file:
        json.dump(results, file, indent=1)
    with open(PROJECT_ROOT / "logs" / f"prompting_{tag}_outputs.json", "w", encoding="utf-8") as file:
        json.dump(outputs, file, ensure_ascii=False, indent=1)
