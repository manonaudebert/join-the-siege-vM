from enum import Enum
from werkzeug.datastructures import FileStorage
from io import BytesIO
from pathlib import Path

class ModelType(str, Enum):
    LOGISTIC = "logistic"
    NAIVE_BAYES = "naive_bayes"


def load_file_content_with_labels(data_dir, file_handler_factory):
    """
    Prepares the training data from the file_data folder. 

    Training data must be in folders with the label as the folder name.
    For example: 

    file_data/
    ├── bank_statement/
        ├── file1.pdf
        ├── file2.docx
    ├── drivers_license/
       ├── license1.jpg
    ├── invoice/
       ├── invoice1.docx
    """
    content, labels = [], []

    for label_dir in Path(data_dir).iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name

        for file_path in label_dir.iterdir():
            if file_path.suffix.lower() not in [".pdf", ".docx", ".jpg", ".jpeg", ".png"]:
                continue

            with open(file_path, "rb") as f:
                file = FileStorage(
                    stream=BytesIO(f.read()),
                    filename=file_path.name,
                )

                handler = file_handler_factory.get_file_handler(file)
                text = handler.extract_text(file)
                content.append(text)
                labels.append(label)

    return content, labels

def is_better_than_previous_accuracy(accuracy_score_path, new_mean_accuracy):
    # Compare with existing accuracy to ensure the new one is better. 
    existing_accuracy = -1
    if accuracy_score_path.exists():
        try:
            with open(accuracy_score_path, "r") as f:
                existing_accuracy = float(f.read().strip())
        except Exception:
            print("Warning: Could not read existing accuracy.")
            return True

    return new_mean_accuracy > existing_accuracy