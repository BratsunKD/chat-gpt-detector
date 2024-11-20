from src.predict_service.predict import initialize_model_with_lora, process_text
from flask import Flask, request, jsonify
import os


BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME")
LORA_WEGHTS_PATH = os.getenv("LORA_WEGHTS_PATH")

app = Flask(__name__)


model, tokenizer = initialize_model_with_lora(BASE_MODEL_NAME, LORA_WEGHTS_PATH)


@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()
    text = data.get('text')
    if text is None:
        return jsonify({'error': 'No text provided'}), 400

    prediction = process_text(model, tokenizer, text)
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)






