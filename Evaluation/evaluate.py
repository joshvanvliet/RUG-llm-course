from Data.dataset_maker import create_split_dataset
from Data.paths import XED_DIR
from datasets import concatenate_datasets
from Training.train import format_dataset
from peft import PeftModel
from Models.gemma_4B import load_gemma_4B
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support

"""
Evaluate the trained models on the according validation datasets.
"""


# Return the predicted emotions for the validation data sentences
def predictions_trained_model(trained_model, validation_data):
    model, processor = load_gemma_4B("google/gemma-4-E4B-it")
    model = PeftModel.from_pretrained(model, trained_model)
    model.eval()

    # Create empty list for predictions
    predictions = []

    # Have the trained model make predictions on the validation sentences
    for sentence in validation_data:
        inputs = processor(
            text=sentence["prompt"],
            return_tensors="pt"
        ).to(model.device)

        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                 **inputs,
                max_new_tokens=10
            )

        # Decode generated text
        prediction = processor.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # Make the labels into integers
        predicted_labels = [
            int(label.strip())
            for label in prediction.split(",")
            if label.strip().isdigit()
        ]

        # Append the predictions to the list
        predictions.append(predicted_labels)

    return predictions


# Return the real labels for the validation data sentences
def labels_validation_data(validation_data):
    # Create empty list for actual labels
    actual_labels = []

    # Append every label to the list
    for sentence in validation_data:
        labels = [
            int(label.strip())
            for label in sentence["completion"].split(",")
        ]
        actual_labels.append(labels)

    return actual_labels


# Evaluate the predictions against the real labels for the validation data sentence
def evaluate_trained_model(trained_model, validation_data):
    predictions = predictions_trained_model(trained_model, validation_data)
    actual_labels = labels_validation_data(validation_data)

    # Convert labels into a multi-label binary format
    mlb = MultiLabelBinarizer(classes=range(1, 9))

    y_true = mlb.fit_transform(actual_labels)
    y_pred = mlb.transform(predictions)

    # Calculate evaluation metrics
    evaluations = {}

    for average in ["micro", "macro", "weighted"]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=average,
            zero_division=0
        )

        evaluations[average] = {
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1
        }

    return evaluations


if __name__ == "__main__":
    english_path = XED_DIR / "AnnotatedData" / "en-annotated.tsv"
    dutch_path = XED_DIR / "Projections" / "nl-projections.tsv"
    alignment_path = XED_DIR / "subtitle-retrieval" / "students" / "pairs-nl.txt"
    english_splits, dutch_splits = create_split_dataset(
        english_path, dutch_path, alignment_path
    )
    validation_data_EN = format_dataset(english_splits[1])
    validation_data_NL = format_dataset(dutch_splits[1])
    validation_data_BI = concatenate_datasets([validation_data_EN, validation_data_NL])
    MODEL_EN = "./trained_gemma_EN"
    MODEL_NL = "./trained_gemma_NL"
    MODEL_BI = "./trained_gemma_BI"
    print("English:", evaluate_trained_model(MODEL_EN, validation_data_EN))
    print("Dutch:", evaluate_trained_model(MODEL_NL, validation_data_NL))
    print("Bilingual:", evaluate_trained_model(MODEL_BI, validation_data_BI))
