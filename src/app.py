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

    files = request.files.getlist("file")

    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No files selected"}), 400

    results = []

    for file in files:
        if not allowed_file(file.filename):
            results.append({
                "filename": file.filename,
                "error": "File type not allowed"
            })
            continue

        try:
            file_class = classify_file(file, CURRENT_MODEL_TYPE)
            results.append({
                "filename": file.filename,
                "file_class": file_class
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return jsonify({"results": results}), 200

if __name__ == '__main__':
    app.run(debug=True)
