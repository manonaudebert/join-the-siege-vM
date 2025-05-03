import os
from werkzeug.datastructures import FileStorage
from io import BytesIO

from src.document_classifier.classifier import classify_file
from src.model.model_utils import ModelType

"""
Test that classifier can correctly classify the test examples. 
If the model accuracy goes down, these tests may fail
"""

image_file = "drivers_license_1.jpg"
pdf_file = "bank_statement_1.pdf"
docx_file = "invoice_4.docx"

def test_classify_drivers_license():
    file = load_filestorage(image_file)
    result = classify_file(file, ModelType.NAIVE_BAYES)
    assert result == "drivers_license"

def test_classify_bank_statement():
    file = load_filestorage(pdf_file)
    result = classify_file(file, ModelType.NAIVE_BAYES)
    assert result == "bank_statement"

def test_classify_invoice():
    file = load_filestorage(docx_file)
    result = classify_file(file, ModelType.NAIVE_BAYES)
    assert result == "invoice"

def test_classify_unknown_file():
    file = FileStorage(stream=BytesIO(b""), filename="badinput.exe")
    result = classify_file(file, ModelType.NAIVE_BAYES)
    assert result == "Unknown file type"

def get_file_path(filename):
    # Resolve path to test files
    test_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(test_dir, "..", "files", filename))

def load_filestorage(filename):
    # Get the test file in FileStorage object
    with open(get_file_path(filename), "rb") as f:
        return FileStorage(stream=BytesIO(f.read()), filename=filename)


