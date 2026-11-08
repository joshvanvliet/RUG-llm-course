from transformers import AutoProcessor, AutoModelForMultimodalLM


MODEL_ID = "Qwen/Qwen3.5-4B"


# Load model
def load_qwen_4B(MODEL_ID):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto"
    )
    return model, processor
