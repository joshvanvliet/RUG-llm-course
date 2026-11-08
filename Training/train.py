import sys

from Models.gemma_4B import load_gemma_4B
from Data.dataset_maker import EMOTIONS, load_dutch_splits
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


def build_prompt(sentence):
    return (
        "For each emotion, answer yes if the sentence expresses it and no if it does not. "
        "A sentence often expresses more than one emotion.\n"
        f"Answer with one line per emotion in this order: {', '.join(EMOTIONS)}.\n\n"
        f"Sentence: {sentence}"
    )


# A yes/no answer per emotion, so that every emotion is decided
# on its own instead of the model stopping after the first label
def build_answer(labels):
    return "\n".join(
        f"{emotion}: {'yes' if number in labels else 'no'}"
        for number, emotion in enumerate(EMOTIONS, start=1)
    )


def parse_answer(text):
    labels = []
    for line in text.splitlines():
        emotion, _, answer = line.partition(":")
        if emotion.strip() in EMOTIONS and answer.strip().lower().startswith("yes"):
            labels.append(EMOTIONS.index(emotion.strip()) + 1)
    return labels


# Chat format so the Gemma chat template is used in training and evaluation
def format_dataset(dataset):
    return dataset.map(
        lambda example: {
            "prompt": [{"role": "user", "content": build_prompt(example["sentence"])}],
            "completion": [{"role": "assistant", "content": build_answer(example["labels"])}],
        },
        remove_columns=dataset.column_names
    )


def model_dir(language, seed):
    return f"./trained_gemma_{language}_seed{seed}"


# Returns (train, develop, test) for English, Dutch and bilingual
def load_splits():
    english_splits, dutch_splits = load_dutch_splits()
    english = [format_dataset(split) for split in english_splits]
    dutch = [format_dataset(split) for split in dutch_splits]
    bilingual = [concatenate_datasets([en, nl]) for en, nl in zip(english, dutch)]
    return english, dutch, bilingual


def train_PEFT(train_data, eval_data, output_dir, seed):
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

    # Configure training, keep the epoch with the lowest develop loss
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_length=256,
        bf16=True,
        completion_only_loss=True,
        seed=seed,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        processing_class=processor.tokenizer
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    english, dutch, bilingual = load_splits()
    train_PEFT(english[0], english[1], model_dir("EN", seed), seed)
    train_PEFT(dutch[0], dutch[1], model_dir("NL", seed), seed)
    train_PEFT(bilingual[0], bilingual[1], model_dir("BI", seed), seed)
