import os
from werkzeug.datastructures import FileStorage
from io import BytesIO
import pytest

from src.utils.file_handler import FileHandlerFactory, ImageHandler, PdfHandler

"""
Unit tests for file_handler. 

Test that factory creates correct file handler type and that each handler can extract text
"""

image_file_name = "drivers_license_1.jpg"
pdf_file_name = "bank_statement_1.pdf"

def test_image_handler():
    # Test image text extraction success case
    with open(get_file_path(image_file_name), "rb") as f:
        file = FileStorage(
            stream=BytesIO(f.read()),
            filename=image_file_name
        )

    file_handler = FileHandlerFactory.get_file_handler(file)
    text = file_handler.extract_text(file)

    assert isinstance(text, str)
    assert isinstance(file_handler, ImageHandler)
    assert "license" in text.lower()

def test_pdf_handler():
    # Test pdf text extraction success case
    with open(get_file_path(pdf_file_name), "rb") as f:
        file = FileStorage(
            stream=BytesIO(f.read()),
            filename=pdf_file_name
        )

    file_handler = FileHandlerFactory.get_file_handler(file)
    text = file_handler.extract_text(file)

    assert isinstance(text, str)
    assert isinstance(file_handler, PdfHandler)
    assert "statement" in text.lower()

def test_no_handler():
    # Test no handler found for file type
    file = FileStorage(
        stream=BytesIO(b"invalid content"),
        filename="invalid_file.exe"
    )

    with pytest.raises(ValueError, match="Could not get file type for invalid_file.exe"):
        FileHandlerFactory.get_file_handler(file)

def get_file_path(test_file_name):
    test_dir = os.path.dirname(__file__)
    file_path = os.path.abspath(os.path.join(test_dir, "..", "files", test_file_name))
    return file_path
