from Models.gemma_4B import load_gemma_4B
from Data.dataset_maker import create_split_dataset
from Data.paths import XED_DIR
from datasets import concatenate_datasets
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

"""
Individual models are trained for the English dataset, the Dutch dataset and the biligual dataset (both English and Dutch).
These datasets consists of sentences in the according language followed by which emotions relate to that sentence. The emotions available are:
EMOTIONS =
    1: "Anger",
    2: "Anticipation",
    3: "Disgust",
    4: "Fear",
    5: "Joy",
    6: "Sadness",
    7: "Surprise",
    8: "Trust"

The models are trained to give a multi label of emotions when input a sentence of according language.
"""

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


def train_PEFT(train_data, output_dir):
    model, processor = load_gemma_4B(MODEL_ID)

    # Configure Lora
    peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    )

    # Configure training
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        max_length=128,
        bf16=True,
        completion_only_loss=True,
        report_to="none"
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        peft_config=peft_config,
        processing_class=processor.tokenizer
    )

    # Start training
    trainer.train()

    return trainer.model


def train_PEFT_EN(train_data_EN):
    output_dir = "./trained_gemma_EN"
    trained_model = train_PEFT(train_data_EN, output_dir)
    trained_model.save_pretrained("./trained_gemma_EN")


def train_PEFT_NL(train_data_NL):
    output_dir = "./trained_gemma_NL"
    trained_model = train_PEFT(train_data_NL, output_dir)
    trained_model.save_pretrained("./trained_gemma_NL")


def train_PEFT_BI(train_data_BI):
    output_dir = "./trained_gemma_BI"
    trained_model = train_PEFT(train_data_BI, output_dir)
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
