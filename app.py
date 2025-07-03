from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os
import torch

app = Flask(__name__)

# ------------------------------------------------------------------
# Pick the model at container start-up so you can swap it by env var
# ------------------------------------------------------------------
MODEL_ID = os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")

def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    # If running in a GPU-enabled container and CUDA is available, use it
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

nlp = load_model(MODEL_ID)


@app.route("/", methods=["GET"])
def root():
    return (
        "POST a file (text/plain, .txt) to /analyze or send JSON "
        '{"text": "..."} to /analyze',
        200,
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    # ------------- handle raw text via JSON ---------------
    if request.is_json and "text" in request.json:
        text = request.json["text"]
    # ------------- handle uploaded text file --------------
    elif "file" in request.files:
        text = request.files["file"].read().decode("utf-8", errors="ignore")
    else:
        return jsonify({"error": "No text or file supplied"}), 400

    result = nlp(text)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
