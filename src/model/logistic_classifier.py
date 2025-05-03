import joblib
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import numpy as np

from src.document_classifier.text_extractor import FileHandlerFactory
from src.model.model_utils import ModelType, is_better_than_previous_accuracy, load_file_content_with_labels

MODEL_NAME = ModelType.LOGISTIC
MODEL_PATH = Path(__file__).resolve().parents[0] / "trained_models" / f"{MODEL_NAME.value}.pkl"
SCORE_PATH = Path(__file__).resolve().parents[0] / "trained_models" / f"{MODEL_NAME.value}_accuracy.txt"


def train_model():
    """
    Train a the logistic classifier model based on the data and labels in the file_data folder. 
    Save the model to a pkl file if it performs better than the last run. 
    """
    current_directory = Path(__file__).resolve()

    # Get the training data and label it based on directory name
    data_directory = current_directory.parents[2] / "file_data"
    contents, labels = load_file_content_with_labels(data_directory, FileHandlerFactory)

    X_train, X_test, y_train, y_test =  train_test_split(contents, labels)

    model = make_pipeline(
        TfidfVectorizer(token_pattern=r"(?u)\b[a-zA-Z]{2,}\b", max_features=100, ngram_range=(1,2)),
        LogisticRegression(max_iter=500)
    )
    model.fit(X_train, y_train)

    # Test on test set and print metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nEvaluation Results: ")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    scores = cross_val_score(model, contents, labels, cv=5)
    mean_accuracy = np.mean(scores)
    print("Cross-validation scores: ", scores)
    print("Mean accuracy: ", mean_accuracy)

    vectorizer = model.named_steps["tfidfvectorizer"]
    classifier = model.named_steps["logisticregression"]

    # Show top predictive terms
    for i, class_label in enumerate(classifier.classes_):
        top_indices = classifier.coef_[i].argsort()[-10:][::-1]
        top_terms = [vectorizer.get_feature_names_out()[j] for j in top_indices]
        print(f"Top terms for class '{class_label}': {top_terms}")  

    if not MODEL_PATH.exists() or is_better_than_previous_accuracy(SCORE_PATH, mean_accuracy):
        # Save pkl file for model if it performs better than the last or if there's no last run and save the new accuracy.
        joblib.dump(model, MODEL_PATH)
        with open(SCORE_PATH, "w") as f:
            f.write(str(mean_accuracy))

        print(f"Model trained and saved")
    else:
        print(f"Accuracy was worse than previous run. Not saving new model. ")


if __name__ == "__main__":
    print("Training Logistic")
    train_model()