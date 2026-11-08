from transformers import AutoProcessor, AutoModelForMultimodalLM


MODEL_ID = "google/gemma-4-E2B-it"


# Load model
def load_gemma_2B(MODEL_ID):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto"
    )
    return model, processor
