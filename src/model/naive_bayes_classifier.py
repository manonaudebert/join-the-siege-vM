import joblib
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

import numpy as np

from src.document_classifier.text_extractor import FileHandlerFactory
from src.model.model_utils import ModelType, is_better_than_previous_accuracy, load_file_content_with_labels

MODEL_NAME = ModelType.NAIVE_BAYES
MODEL_PATH = Path(__file__).resolve().parents[0] / "trained_models" / f"{MODEL_NAME.value}.pkl"
SCORE_PATH = Path(__file__).resolve().parents[0] / "trained_models" / f"{MODEL_NAME.value}_accuracy.txt"

def train_model():
    """
    Train a the naive bayes classifier model based on the data and labels in the file_data folder. 
    Save the model to a pkl file if it performs better than the last run.
    """

    current_directory = Path(__file__).resolve()

    # Get the training data and label it based on directory name
    data_directory = current_directory.parents[2] / "file_data"
    contents, labels = load_file_content_with_labels(data_directory, FileHandlerFactory)

    X_train, X_test, y_train, y_test =  train_test_split(contents, labels)

    model = make_pipeline(
        TfidfVectorizer(token_pattern=r"(?u)\b[a-zA-Z]{2,}\b", max_features=100, ngram_range=(1,2)),
        MultinomialNB()
    )
    model.fit(X_train, y_train)

    joblib.dump(model, 'naive_bayes.pkl')
    print(f"Model trained and saved")

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
    classifier = model.named_steps["multinomialnb"]

    # Show top predictive terms
    if hasattr(classifier, "feature_log_prob_"):
        for i, class_label in enumerate(classifier.classes_):
            top_indices = classifier.feature_log_prob_[i].argsort()[-10:][::-1]
            top_terms = [vectorizer.get_feature_names_out()[j] for j in top_indices]
            print(f"Top terms for class '{class_label}': {top_terms}")

    if not MODEL_PATH.exists() or is_better_than_previous_accuracy(SCORE_PATH, mean_accuracy):
        # Save pkl file for model if it performs better than the last or if there's no last run
        # Save the updated accuracy as well 
        # If needing to train for a new document type and model isn't being save due to accuracy being worse,this can be commented out or delete the existing pkl file.
        joblib.dump(model, MODEL_PATH)
        with open(SCORE_PATH, "w") as f:
            f.write(str(mean_accuracy))

        print(f"Model trained and saved")
    else:
        print(f"Accuracy was worse than previous run. Not saving new model. ")


if __name__ == "__main__":
    print("Training Naive Bayes")
    train_model()