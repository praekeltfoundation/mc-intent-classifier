import json
from pathlib import Path
from typing import Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class IntentClassifier:
    def __init__(self, json_path, model_name=DEFAULT_MODEL):
        """
        Initialize with precomputed embeddings from a JSON file and model.
        """
        self.model_name = model_name
        self.json_path = Path(json_path)

        self._load_model()
        self.embeddings = self._load_embeddings()

    def _load_model(self):
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise ValueError(f"Error loading model '{self.model_name}': {e}") from e

    def _load_embeddings(self):
        """
        Load precomputed embeddings from a JSON file.
        """
        if not self.json_path.exists():
            raise FileNotFoundError(f"The file '{self.json_path}' does not exist.")

        try:
            with self.json_path.open("r") as f:
                embeddings_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON file '{self.json_path}': {e}") from e

        try:
            embeddings = {
                intent: np.array(embedding_list)
                for intent, embedding_list in embeddings_data.items()
            }
        except Exception as e:
            raise ValueError(
                f"Error processing embeddings data from '{self.json_path}':{e}"
            ) from e

        return embeddings

    def similarity(self, incoming_embedding, intent_embeddings):
        """Compute similarity between incoming and intent embeddings."""
        try:
            return cosine_similarity(incoming_embedding, intent_embeddings)
        except Exception as e:
            raise ValueError(f"Error computing similarity: {e}") from e

    def calculate_average_similarity(self, incoming_text):
        """Calculate average similarity for each intent class."""
        try:
            incoming_embedding = self.model.encode([incoming_text])
        except Exception as e:
            raise ValueError(f"Error encoding incoming text: {e}") from e

        average_similarities = {}
        for intent, intent_embeddings in self.embeddings.items():
            try:
                similarities = self.similarity(incoming_embedding, intent_embeddings)
                average_similarities[intent] = np.mean(similarities)
            except Exception as e:
                raise ValueError(
                    f"Error calculating similarity for intent '{intent}': {e}"
                ) from e

        return average_similarities

    def classify(self, incoming_text) -> Tuple[str, float]:
        """
        Classify the intent of the incoming text and return the confidence score.

        Returns:
            tuple: (intent, confidence_score)
        """
        average_similarities = self.calculate_average_similarity(incoming_text)
        best_intent = max(average_similarities, key=average_similarities.get)
        confidence = average_similarities[best_intent]
        return best_intent, confidence
