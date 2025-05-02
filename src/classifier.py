from werkzeug.datastructures import FileStorage

from src.utils.file_handler import FileHandlerFactory

def classify_file(file: FileStorage):
    file_handler = FileHandlerFactory.get_file_handler(file)
    text = file_handler.extract_text(file)
    print(f'TEXT:: {text}')

    filename = file.filename.lower()

    if "drivers_license" in filename:
        return "drivers_licence"

    if "bank_statement" in filename:
        return "bank_statement"

    if "invoice" in filename:
        return "invoice"

    return "unknown file"

