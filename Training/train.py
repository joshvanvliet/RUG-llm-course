from Models.gemma_4B import load_gemma_4B
from Data.dataset_maker import create_split_dataset
from Data.paths import XED_DIR
from datasets import concatenate_datasets
from peft import LoraConfig
from trl import SFTTrainer

EMOTIONS = {
    1: "Anger",
    2: "Anticipation",
    3: "Disgust",
    4: "Fear",
    5: "Joy",
    6: "Sadness",
    7: "Surprise",
    8: "Trust"
}

MODEL_ID = "google/gemma-4-E4B-it"


def format_dataset(dataset):
    return dataset.map(
        lambda example: {
            "prompt": (
                "Classify the following sentence using these emotions:\n"
                "1: Anger, 2: Anticipation, "
                "3: Disgust, 4: Fear, 5: Joy, "
                "6: Sadness, 7: Surprise, 8: Trust.\n\n"
                f"Sentence: {example['0']}"
            ),
            "completion": ", ".join(
                str(label) for label in sorted(example["1"])
            )
        },
        remove_columns=dataset.column_names
    )


def train_PEFT(train_data):
    model = load_gemma_4B()

    print([
        name for name, module in model.named_modules()
        if "proj" in name
    ][:30])


def train_PEFT_EN(train_data_EN):
    trained_model = train_PEFT(train_data_EN)
    trained_model.save_pretrained("./trained_gemma_EN")


def train_PEFT_NL(train_data_NL):
    trained_model = train_PEFT(train_data_NL)
    trained_model.save_pretrained("./trained_gemma_NL")


def train_PEFT_BI(train_data_BI):
    trained_model = train_PEFT(train_data_BI)
    trained_model.save_pretrained("./trained_gemma_BI")


if __name__ == "__main__":
    english_path = XED_DIR / "AnnotatedData" / "en-annotated.tsv"
    dutch_path = XED_DIR / "Projections" / "nl-projections.tsv"
    alignment_path = XED_DIR / "subtitle-retrieval" / "students" / "pairs-nl.txt"
    english_splits, dutch_splits = create_split_dataset(
        english_path, dutch_path, alignment_path
    )
    train_data_EN = format_dataset(english_splits[0])
    train_data_NL = format_dataset(dutch_splits[0])
    train_data_BI = concatenate_datasets([train_data_EN, train_data_NL])
    train_PEFT_EN(train_data_EN)
    train_PEFT_NL(train_data_NL)
    train_PEFT_BI(train_data_BI)
