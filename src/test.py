import csv
from datetime import datetime

from intent_classifier import IntentClassifier

# Optional: Configure basic logging if you want to see INFO logs from the script itself,
# though the classifier might have its own logging setup.
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_intent_classification() -> None:
    """
    Initializes the IntentClassifier and classifies a predefined list of queries.
    """

    queries = [
        # Variations for "Mom (response to What is your name?)" - with varied familial/nicknames
        "Mom",
        "Dad",
        "It's Mom",
        "I am mom",
        "Call me Mom",
        "My name is Mom",
        # Variations for "Cramp in the hands via ASK" (pregnancy-related)
        "Cramp in the hands",
        "My hands are cramping.",
        "I'm getting cramps in my hands during pregnancy.",
        "Experiencing hand cramps, is this pregnancy related?",
        "What to do for cramping hands while pregnant?",
        # Variations for "Pain on the leftside" (pregnancy-related)
        "Pain on the leftside",
        "My left side hurts.",
        "I have a pain on my left, is that normal in pregnancy?",
        "Feeling soreness on the left side of my body (pregnant).",
        "Left-side discomfort during pregnancy.",
        # Variations for "In 2 days I was there on yesterday" (confused timing/presence)
        "In 2 days I was there on yesterday",
        "Yesterday I am there in two days.",
        "I go there yesterday and for 2 days.",
        "Was there 2 days after yesterday.",
        "My visit was yesterday for 2 days time.",
        # Variations for "Pamela (via question What is your name?)" - with varied given names
        "Pamela",
        "It's Pamela.",
        "My name is Pamela.",
        "I'm Pamela.",
        "Just Pamela, thanks.",
    ]

    # Output filename
    output_filename = "classification_results.txt"

    # Initialize classifier (paths from your example)
    # IMPORTANT: Ensure these paths are correct for your environment.
    embeddings_file_path = "src/data/intent_embeddings_path.json"
    nlu_file_path = "src/data/nlu.yaml"
    classifier = None

    try:
        print("Initializing IntentClassifier...")
        print(f"  Embeddings path: {embeddings_file_path}")
        print(f"  NLU data path: {nlu_file_path}")

        classifier = IntentClassifier(
            embeddings_path=embeddings_file_path, nlu_path=nlu_file_path
        )
        print("IntentClassifier initialized successfully.")
    except FileNotFoundError:
        print(
            "ERROR: Could not initialize classifier. Ensure paths are correct and files exist:"
        )
        print(f"  Attempted embeddings path: {embeddings_file_path}")
        print(f"  Attempted NLU data path: {nlu_file_path}")
        return  # Exit if classifier fails to initialize
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while initializing classifier: {e}")
        return  # Exit if classifier fails to initialize

    # Set thresholds (from your example, using the correct parameter name for margin)
    s_threshold_value = 0.5
    margin_threshold_value = 0.002

    # --- Classification and Writing to CSV File ---
    try:
        with open(output_filename, "w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write CSV Header
            csv_writer.writerow(["Query", "Predicted Intent", "Reported Score"])

            # Print run information to console
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\nClassification Run at {timestamp_str}")
            print("Classifier Initialized With:")
            print(f"  Embeddings Path: {embeddings_file_path}")
            print(f"  NLU Data Path: {nlu_file_path}")
            print("Thresholds Used:")
            print(f"  s_thresh: {s_threshold_value}")
            print(f"  margin_threshold: {margin_threshold_value}")
            print("----------------------------------------------------")
            print(f"Processing queries and writing to {output_filename}...")

            for i, query in enumerate(queries):
                intent_result = "ERROR_IN_CLASSIFICATION"
                score_result = "N/A"
                error_message_for_csv = ""

                try:
                    intent, score = classifier.classify(
                        incoming_text=query,
                        s_thresh=s_threshold_value,
                        m_thresh=margin_threshold_value,
                    )
                    intent_result = intent
                    score_result = f"{score:.4f}"  # Format score for CSV
                    if (
                        i + 1
                    ) % 5 == 0 or i == 0:  # Print progress to console periodically
                        print(
                            f'  Processed query {i+1}/{len(queries)}: "{query[:50]}..." -> {intent_result} ({score_result})'
                        )

                except Exception as e:
                    print(f'  ERROR classifying query "{query}": {e}')
                    # Sanitize error message for CSV: remove newlines and commas
                    error_message_for_csv = str(e).replace("\n", " ").replace(",", ";")
                    score_result = (
                        error_message_for_csv  # Put error in score column for this row
                    )

                csv_writer.writerow([query, intent_result, score_result])

        print("----------------------------------------------------")
        print(
            f"Classification complete. Output successfully written to: {output_filename}"
        )

    except OSError as e:
        print(f"ERROR: Could not write results to file {output_filename}: {e}")
    except Exception as e:
        print(
            f"ERROR: An unexpected error occurred during classification or writing: {e}"
        )


if __name__ == "__main__":
    run_intent_classification()
