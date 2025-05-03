from pathlib import Path
from werkzeug.datastructures import FileStorage

from src.document_classifier.text_extractor import FileHandlerFactory
import joblib

from src.model.model_utils import ModelType


def classify_file(file: FileStorage, model_name: ModelType):
    try:
        model_path = Path(__file__).resolve().parents[1] / "model" / "trained_models" / f"{model_name.value}.pkl"
        model = joblib.load(model_path)

        file_handler = FileHandlerFactory.get_file_handler(file)
        text = file_handler.extract_text(file)
        predicted_class = model.predict([text])[0]
        return predicted_class
    except:
        return "Unknown file type"
    


