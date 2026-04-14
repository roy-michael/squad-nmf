"""
Classification model for underwater vehicle sound classification.

Provides machine learning models and a complete pipeline for predicting
vehicle types from NMF-extracted audio features.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import joblib
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class ClassifierDataset:
    """
    Dataset manager for classification with stratified train/test splitting.

    Handles feature and label organization, stratified splitting to maintain
    class distribution, and provides label name mappings.
    """

    def __init__(self, features_list: List[np.ndarray], labels_list: List[int]):
        """
        Initialize dataset from features and labels.

        Parameters
        ----------
        features_list : List[np.ndarray]
            List of feature vectors from feature_extractor (each shape: (n_components*8,))
        labels_list : List[int]
            Corresponding class labels (0, 1, 2, ...)

        Raises
        ------
        ValueError
            If features and labels have mismatched lengths or invalid dimensions.
        """
        if len(features_list) == 0:
            raise ValueError("features_list cannot be empty")
        if len(features_list) != len(labels_list):
            raise ValueError(
                f"Length mismatch: {len(features_list)} features but {len(labels_list)} labels"
            )

        # Stack features into matrix
        self.features = np.array(features_list, dtype=np.float32)
        self.labels = np.array(labels_list, dtype=np.int32)

        if self.features.ndim != 2:
            raise ValueError(
                f"Features must be 2D array, got shape {self.features.shape}"
            )

        if self.labels.ndim != 1:
            raise ValueError(
                f"Labels must be 1D array, got shape {self.labels.shape}"
            )

        # Extract unique classes and create label mapping
        self.classes = np.unique(self.labels)
        self.n_classes = len(self.classes)
        self.label_names = {
            i: class_id
            for i, class_id in enumerate(self.classes)
        }

    def train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[Dict, Dict]:
        """
        Perform stratified train/test split.

        Maintains class distribution in both train and test sets.

        Parameters
        ----------
        test_size : float, optional
            Proportion of data for testing (default: 0.2)
        random_state : int, optional
            Random seed for reproducibility (default: 42)

        Returns
        -------
        train_data : dict
            Dictionary with 'features' (np.ndarray) and 'labels' (np.ndarray)
        test_data : dict
            Dictionary with 'features' (np.ndarray) and 'labels' (np.ndarray)
        """
        X_train, X_test, y_train, y_test = sklearn_train_test_split(
            self.features,
            self.labels,
            test_size=test_size,
            stratify=self.labels,
            random_state=random_state,
        )

        return (
            {"features": X_train, "labels": y_train},
            {"features": X_test, "labels": y_test},
        )

    def get_label_names(self) -> Dict[int, str]:
        """
        Get mapping of class IDs to class names.

        Returns
        -------
        label_names : dict
            Mapping of integer class IDs to string names.
            For now, returns {0: '0', 1: '1', ...} but can be extended.
        """
        return {int(class_id): str(class_id) for class_id in self.classes}


class SoundClassifier:
    """
    Machine learning classifier for underwater sound classification.

    Supports multiple model types (Random Forest, SVM, Gradient Boosting)
    with training, prediction, evaluation, and serialization capabilities.
    """

    def __init__(
        self,
        model_type: str = "rf",
        random_state: int = 42,
    ):
        """
        Initialize classifier with specified model type.

        Parameters
        ----------
        model_type : str, optional
            Type of classifier:
            - 'rf': RandomForestClassifier (default)
            - 'svm': SVC (Support Vector Classifier)
            - 'gb': GradientBoostingClassifier
        random_state : int, optional
            Random seed for reproducibility (default: 42)

        Raises
        ------
        ValueError
            If model_type is not recognized.
        """
        if model_type not in ["rf", "svm", "gb"]:
            raise ValueError(
                f"Unknown model_type: {model_type}. Choose from 'rf', 'svm', 'gb'"
            )

        self.model_type = model_type
        self.random_state = random_state
        self.is_trained = False
        self.training_metadata = {}

        if model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=random_state,
                n_jobs=-1,
            )
        elif model_type == "svm":
            self.model = SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                random_state=random_state,
            )
        else:  # gb
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state,
            )

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """
        Train the classifier on training data.

        Parameters
        ----------
        X_train : np.ndarray
            Training features (n_samples, n_features)
        y_train : np.ndarray
            Training labels (n_samples,)
        **kwargs : optional
            Additional arguments (stored in training_metadata)

        Raises
        ------
        ValueError
            If input data is invalid.
        """
        if X_train.shape[0] == 0:
            raise ValueError("X_train cannot be empty")
        if len(X_train) != len(y_train):
            raise ValueError(
                f"Length mismatch: {len(X_train)} samples but {len(y_train)} labels"
            )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        self.training_metadata = {
            "n_samples": len(X_train),
            "n_features": X_train.shape[1],
            "n_classes": len(np.unique(y_train)),
            "model_type": self.model_type,
            **kwargs,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.

        Parameters
        ----------
        X : np.ndarray
            Features (n_samples, n_features)

        Returns
        -------
        predictions : np.ndarray
            Predicted class labels (n_samples,)

        Raises
        ------
        RuntimeError
            If model has not been trained.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.

        Parameters
        ----------
        X : np.ndarray
            Features (n_samples, n_features)

        Returns
        -------
        probabilities : np.ndarray
            Class probabilities (n_samples, n_classes)

        Raises
        ------
        RuntimeError
            If model has not been trained.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict_proba(X)

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate classifier on test data.

        Computes accuracy, precision, recall, and F1-score (weighted average
        for multiclass classification).

        Parameters
        ----------
        X_test : np.ndarray
            Test features (n_samples, n_features)
        y_test : np.ndarray
            Test labels (n_samples,)

        Returns
        -------
        metrics : dict
            Dictionary with keys:
            - 'accuracy': Accuracy score
            - 'precision': Weighted precision
            - 'recall': Weighted recall
            - 'f1': Weighted F1-score

        Raises
        ------
        RuntimeError
            If model has not been trained.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        y_pred = self.predict(X_test)

        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted")),
            "recall": float(recall_score(y_test, y_pred, average="weighted")),
            "f1": float(f1_score(y_test, y_pred, average="weighted")),
        }

    def save(self, filepath: str) -> None:
        """
        Save trained model to disk using joblib.

        Parameters
        ----------
        filepath : str
            Path to save the model file

        Raises
        ------
        RuntimeError
            If model has not been trained.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")

        model_data = {
            "model": self.model,
            "model_type": self.model_type,
            "random_state": self.random_state,
            "is_trained": self.is_trained,
            "training_metadata": self.training_metadata,
        }
        joblib.dump(model_data, filepath)

    @staticmethod
    def load(filepath: str) -> "SoundClassifier":
        """
        Load a trained model from disk.

        Parameters
        ----------
        filepath : str
            Path to the saved model file

        Returns
        -------
        classifier : SoundClassifier
            Loaded classifier instance

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        """
        model_data = joblib.load(filepath)

        classifier = SoundClassifier(
            model_type=model_data["model_type"],
            random_state=model_data["random_state"],
        )
        classifier.model = model_data["model"]
        classifier.is_trained = model_data["is_trained"]
        classifier.training_metadata = model_data["training_metadata"]

        return classifier


class AudioClassificationPipeline:
    """
    End-to-end pipeline for audio classification.

    Integrates audio loading, preprocessing, feature extraction, and
    classification into a single pipeline.
    """

    def __init__(self, preprocessor, feature_extractor, classifier):
        """
        Initialize pipeline with components.

        Parameters
        ----------
        preprocessor : AudioPreprocessor
            Preprocessor instance for mel spectrogram extraction
        feature_extractor : NMFFeatureExtractor
            Feature extractor instance for NMF-based feature vectors
        classifier : SoundClassifier
            Classifier instance for predictions
        """
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.label_names = {}

    def train(
        self,
        audio_paths: List[str],
        labels: List[int],
        test_size: float = 0.2,
    ) -> Dict:
        """
        Train the complete pipeline on audio files.

        Loads audio → preprocesses → extracts features → trains classifier.

        Parameters
        ----------
        audio_paths : List[str]
            List of paths to audio files
        labels : List[int]
            Corresponding class labels
        test_size : float, optional
            Proportion of data for testing (default: 0.2)

        Returns
        -------
        results : dict
            Dictionary with keys:
            - 'train_metrics': Training set metrics
            - 'test_metrics': Test set metrics
            - 'n_samples': Total number of samples processed

        Raises
        ------
        ValueError
            If audio_paths and labels have mismatched lengths.
        Exception
            If audio loading or processing fails.
        """
        if len(audio_paths) != len(labels):
            raise ValueError(
                f"Mismatch: {len(audio_paths)} audio files but {len(labels)} labels"
            )

        # Import here to avoid circular dependencies
        from audio_loader import load_audio

        features_list = []

        # Load, preprocess, and extract features for each audio file
        for audio_path in audio_paths:
            audio_data, sr = load_audio(audio_path)
            mel_spec = self.preprocessor.preprocess(audio_data, sr)
            features = self.feature_extractor.extract(mel_spec)
            features_list.append(features)

        # Create dataset and split
        dataset = ClassifierDataset(features_list, labels)
        self.label_names = dataset.get_label_names()

        train_data, test_data = dataset.train_test_split(
            test_size=test_size, random_state=42
        )

        # Train classifier
        self.classifier.train(
            train_data["features"],
            train_data["labels"],
        )

        # Evaluate on training and test sets
        train_metrics = self.classifier.evaluate(
            train_data["features"], train_data["labels"]
        )
        test_metrics = self.classifier.evaluate(
            test_data["features"], test_data["labels"]
        )

        return {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "n_samples": len(features_list),
        }

    def classify(self, audio_path: str) -> Dict:
        """
        Classify a single audio file.

        Loads audio → preprocesses → extracts features → predicts class.

        Parameters
        ----------
        audio_path : str
            Path to the audio file

        Returns
        -------
        result : dict
            Dictionary with keys:
            - 'predicted_class': Predicted class ID (int)
            - 'predicted_label': Predicted class name (str)
            - 'confidence': Confidence score (float, max probability)
            - 'probabilities': Class probabilities (np.ndarray)

        Raises
        ------
        RuntimeError
            If classifier has not been trained.
        Exception
            If audio loading or processing fails.
        """
        if not self.classifier.is_trained:
            raise RuntimeError("Classifier must be trained before classification")

        # Import here to avoid circular dependencies
        from audio_loader import load_audio

        # Load, preprocess, and extract features
        audio_data, sr = load_audio(audio_path)
        mel_spec = self.preprocessor.preprocess(audio_data, sr)
        features = self.feature_extractor.extract(mel_spec)

        # Predict
        predicted_class = self.classifier.predict(features.reshape(1, -1))[0]
        probabilities = self.classifier.predict_proba(features.reshape(1, -1))[0]
        confidence = float(np.max(probabilities))

        predicted_label = self.label_names.get(
            int(predicted_class), str(predicted_class)
        )

        return {
            "predicted_class": int(predicted_class),
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probabilities": probabilities,
        }
