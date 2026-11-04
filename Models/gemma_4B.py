from transformers import AutoProcessor, AutoModelForMultimodalLM


MODEL_ID = "google/gemma-4-E4B-it"


# Load model
def load_gemma_4B(MODEL_ID):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto"
    )
    return model, processor
