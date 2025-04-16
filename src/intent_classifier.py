import yaml
import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# LEGACY_MODEL = "all-MiniLM-L6-v2"  # 60% performance
# ALTERNATE_MODEL = sentence-transformers/paraphrase-multilingual-mpnet-base-v2  # 68% performance
DEFAULT_MODEL = "BAAI/bge-m3"
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)


class IntentClassifier:
    """
    Classifies incoming text into predefined intents using sentence embeddings.
    Uses Mean Embedding Similarity and Margin-Based Confidence.
    """
    def __init__(
        self,
        embeddings_path: str | Path,
        nlu_path: str | Path,
        model_name: str = DEFAULT_MODEL
    ):
        """
        Initializes the classifier, loads the model, and loads or computes
        mean embeddings.

        Args:
            embeddings_path: Path to the JSON file storing mean embeddings.
            nlu_path: Path to the YAML file containing NLU training examples.
            model_name: Name of the Sentence Transformer model to use.
        """

        self.model_name = model_name
        self.embeddings_path = Path(embeddings_path)
        self.nlu_path = Path(nlu_path)
        self.model: SentenceTransformer | None = None
        self.mean_embeddings: dict[str, np.ndarray] = {}

        self._load_model()
        if not self.embeddings_path.exists():
            labeled_data = read_yaml(self.nlu_path)
            if not labeled_data:
                raise ValueError(
                    f"""No valid labeled data loaded from '{self.nlu_path}'.
                    Cannot compute embeddings."""
                )

            mean_embeddings_data = compute_mean_embeddings(
                labeled_data, self.model
            )
            if not mean_embeddings_data:
                raise ValueError(
                    f"""Failed to compute any mean embeddings
                    from '{self.nlu_path}'. Check data and model."""
                )

            save_embeddings_to_file(mean_embeddings_data, self.embeddings_path)
            self.mean_embeddings = self._load_mean_embeddings()
        else:
            self.mean_embeddings = self._load_mean_embeddings()

    def _load_model(self):
        """Loads the Sentence Transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise ValueError(
                f"Error loading model '{self.model_name}': {e}"
            ) from e

    def _load_mean_embeddings(self) -> dict[str, np.ndarray]:
        """
        Load precomputed MEAN embeddings from a JSON file.
        Each intent maps to a single mean vector (NumPy array).
        """
        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"The embeddings file '{self.embeddings_path}' does not exist."
            )

        try:
            with self.embeddings_path.open("r", encoding="utf-8") as f:
                embeddings_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON file '{self.embeddings_path}': {e}") from e
        except Exception as e:
            raise ValueError(f"Error opening or reading file '{self.embeddings_path}': {e}") from e

        if not isinstance(embeddings_data, dict):
            raise ValueError(f"Invalid format in embeddings file '{self.embeddings_path}'. Expected a dictionary.")

        mean_embeddings = {}
        intents_skipped = 0
        for intent, mean_vector_list in embeddings_data.items():
            if not isinstance(mean_vector_list, list):
                intents_skipped += 1
                continue
            try:
                mean_vector = np.array(
                    mean_vector_list, dtype=np.float32
                ).reshape(1, -1)
                mean_embeddings[intent] = mean_vector
            except Exception as e:
                logging.error(
                    """Error processing mean embedding for intent
                    '%s' from '%s': %s. Skipping this intent.""",
                    intent, self.embeddings_path, e
                )
                intents_skipped += 1

        if not mean_embeddings:
            raise ValueError(
                f"""No valid embeddings loaded from '{self.embeddings_path}'.
                Check file content and format."""
            )

        return mean_embeddings

    def _calculate_similarities_to_mean(
        self, incoming_text: str
    ) -> dict[str, float]:
        """
        Calculate cosine similarity between incoming text embedding and
        the precomputed MEAN embedding for each intent. Returns empty
        dict on critical error.
        """
        if self.model is None:
            raise RuntimeError(
                "Model is not loaded. Cannot calculate similarities."
            )

        try:
            incoming_embedding = self.model.encode([incoming_text])
        except Exception as e:
            raise ValueError(f"Error encoding incoming text: {e}") from e

        similarities = {}
        for intent_, mean_intent_embedding in self.mean_embeddings.items():
            try:
                similarity_score = cosine_similarity(
                    incoming_embedding, mean_intent_embedding
                )[0][0]
                similarities[intent_] = float(similarity_score)
            except Exception as e:
                raise ValueError(
                    f"Error calculating similarity for intent '{intent_}': {e}"
                ) from e

        return similarities

    def classify(
        self, incoming_text: str, margin_threshold: float = 0.009
    ) -> tuple[str, float]:
        """
        Classify the intent using similarity to mean embeddings and a
        confidence margin. Includes a rule to prioritize 'Baby Loss' if
        confusion with 'Opt out' is below margin.

        Args:
            incoming_text (str): The user input text.
            margin_threshold (float): Minimum difference required between the
                top score and the second-best score for general confidence.
                Set to 0 or negative to disable margin check.

        Returns:
            tuple[str, float]: (intent, score). Intent may be 'out_of_scope'
                or 'Unclassified'. Score is the score of the returned intent.
        """

        if not isinstance(incoming_text, str) or not incoming_text.strip():
            return "Unclassified", 0.0

        log_text = (
            incoming_text[:50] + '...' if len(incoming_text)
            > 50 else incoming_text
        )
        logging.debug("Classifying text: '%s'", log_text)

        similarities_to_mean = self._calculate_similarities_to_mean(
            incoming_text
        )

        if not similarities_to_mean:
            return "Unclassified", 0.0

        # Handle case with only one possible intent
        if len(similarities_to_mean) < 2:
            best_intent = next(iter(similarities_to_mean.keys()))
            intent_confidence = next(iter(similarities_to_mean.values()))
            return best_intent, intent_confidence

        # --- Get Top 2 Intents and Scores ---
        try:
            # Sort items by score (descending) to easily get top 2
            sorted_items = sorted(
                similarities_to_mean.items(),
                key=lambda item: item[1], reverse=True
            )
            best_intent, intent_confidence = sorted_items[0]
            second_intent, second_score = sorted_items[1]
            margin = intent_confidence - second_score

        except Exception as e:
            raise ValueError(
                f"""Error during score sorting/margin calculation
                for '{log_text}': '{e}'"""
            ) from e

        # --- Apply Specific Business Rule: Baby Loss vs Opt out ---
        # When these two intents are close to each other, in score,
        # we want to prioritize 'Baby Loss'
        is_babyloss_optout_pair = (
            {best_intent, second_intent} ==
            {"Baby Loss", "Opt out"}
        )
        if is_babyloss_optout_pair:
            # If confusion is between Baby Loss and Opt out,
            # always prioritize Baby Loss
            baby_loss_score = similarities_to_mean.get("Baby Loss", -1.0)
            return "Baby Loss", baby_loss_score

        # Apply General Margin Thresholding (if not is_babyloss_optout_pair)
        elif margin_threshold > 0 and margin < margin_threshold:
            # If margin is too small for any other pair,
            # classify as Unclassified
            return "Unclassified", intent_confidence
        else:
            return best_intent, intent_confidence


def read_yaml(yaml_file_path: Path) -> dict[str, list[str]]:
    """
    Read intents and examples from a YAML file, ignoring synonym entries,
    and removing leading '-' from each example.

    Args:
        yaml_file_path (str or Path): Path to the YAML file.

    Returns:
        dict: A dictionary containing intents as keys and examples as values.
    """
    yaml_file_path = Path(yaml_file_path)

    if not yaml_file_path.exists():
        raise FileNotFoundError(f"The file '{yaml_file_path}' does not exist.")

    try:
        with yaml_file_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(
            f"Error parsing YAML file '{yaml_file_path}': {e}"
        ) from e

    intents_data = {}
    for entry in data.get("nlu", []):
        if "intent" in entry:
            intent = entry["intent"]
            examples = entry["examples"].splitlines()
            examples = [
                example.lstrip("-").strip()
                for example in examples if example.strip()
            ]
            intents_data[intent] = examples

    if not intents_data:
        raise ValueError("No intents found in the provided YAML file.")

    return intents_data


def compute_mean_embeddings(
    labeled_data: dict[str, list[str]],
    model: SentenceTransformer
) -> dict[str, list[float]]:
    """
    Compute the MEAN embedding for each intent's examples using the provided
    model.

    Args:
        labeled_data: Dictionary mapping intents to lists of examples.
        model: The loaded sentence transformer model.

    Returns:
        Dictionary mapping intents to their mean embedding vector (as a list).
        Returns empty dict if no embeddings could be computed.
    """
    mean_embeddings = {}
    intents_processed = 0

    for intent, texts in labeled_data.items():
        try:
            example_embeddings = model.encode(texts, show_progress_bar=False)

            if example_embeddings is None or example_embeddings.shape[0] == 0:
                continue

            mean_vector = np.mean(example_embeddings, axis=0)

            if mean_vector is None or not isinstance(mean_vector, np.ndarray):
                continue

            mean_embeddings[intent] = mean_vector.astype(float).tolist()
            intents_processed += 1

        except Exception as e:
            raise ValueError(
                f"Error computing embeddings for intent '{intent}': {e}"
            ) from e

    return mean_embeddings


def save_embeddings_to_file(embeddings: dict[str, list[float]], output_file: Path):
    """
    Save computed mean embeddings to a JSON file.

    Args:
        embeddings: The dictionary of mean embeddings.
        output_file: The path to save the JSON file.
    """

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            # Use compact separators for production file size
            json.dump(embeddings, f, separators=(',', ':'))
    except OSError as e:
        raise OSError(
            f"Error saving embeddings to '{output_file}': {e}"
        ) from e
    except TypeError as e:
        raise TypeError(f"Data format error saving embeddings: {e}") from e
    except Exception as e:
        raise RuntimeError(
            f"An unexpected error occurred while saving embeddings: {e}"
        ) from e
