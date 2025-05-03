from enum import Enum
from flask import Flask, request, jsonify

from src.document_classifier.classifier import classify_file
from src.model.model_utils import ModelType

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg','docx'}
CURRENT_MODEL_TYPE = ModelType.NAIVE_BAYES

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/classify_file', methods=['POST'])
def classify_file_route():

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed"}), 400

    # Classify using the current model set 
    file_class = classify_file(file, CURRENT_MODEL_TYPE)
    return jsonify({"file_class": file_class}), 200

if __name__ == '__main__':
    app.run(debug=True)
