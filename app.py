import os
import gradio as gr
import nltk
from huggingface_hub import hf_hub_download

from deployment import load_bundle, load_networks, predict_text


MODEL_REPO_ID = "Lily-Trinh/PersonalityTraitsFromText"

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

model_path = hf_hub_download(
    repo_id=MODEL_REPO_ID,
    filename="pytorch_model.bin",
    token=os.getenv("HF_TOKEN"),
)

bundle = load_bundle(model_path)
networks = load_networks(bundle)


def predict(essay):
    
    if not essay.strip():
        return {"error": "Please enter an essay."}

    predictions = predict_text(essay, bundle, networks=networks)

    return {
        dim: {
            "probability": round(values["probability"], 4),
            "prediction": values["prediction"],
        }
        for dim, values in predictions.items()
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Essay",
        lines=12,
        placeholder="Paste a new essay here...",
    ),
    outputs=gr.JSON(label="Big Five Predictions"),
    title="Essay Big Five Personality Predictor",
    description=(
        "Predicts Openness, Conscientiousness, Extraversion, "
        "Agreeableness, and Neuroticism from essay text."
    ),
)

demo.launch(show_error=True)
