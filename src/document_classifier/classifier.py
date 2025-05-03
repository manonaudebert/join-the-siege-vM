from pathlib import Path
from werkzeug.datastructures import FileStorage

from src.document_classifier.text_extractor import FileHandlerFactory
import joblib

from src.model.model_utils import ModelType


def classify_file(file: FileStorage, model_name: ModelType):
    try:
        # Load the PKL file. If not found raise error. 
        model_path = Path(__file__).resolve().parents[1] / "model" / "trained_models" / f"{model_name.value}.pkl"
        if not model_path.exists():
            raise FileNotFoundError()

        model = joblib.load(model_path)

        # Extract text and predict class
        file_handler = FileHandlerFactory.get_file_handler(file)
        text = file_handler.extract_text(file)
        predicted_class = model.predict([text])[0]

        return predicted_class
    except FileNotFoundError as e:
        raise FileNotFoundError(f"No model file found. Please train model first. ")
    except ValueError as e:
        return f"Unknown file type"
    except Exception as e:
        raise RuntimeError(f"Error during file classification {e}")


