## MomConnect Intent Classifier

A machine learning service to classify inbound user messages for the *MomConnect*. This service identifies user intents such as service feedback and sensitive topics like baby loss, enabling the platform to provide appropriate and timely responses.


### How It Works: A Hybrid Approach to Intent Classification
This classifier uses a modern, two-stage process to understand user messages with both accuracy and precision. This design ensures that broad categories are handled robustly by a data-driven model, while critical, specific cases are managed with transparent and reliable rules.

_A simplified flowchart of the classification process._

1. Broad Understanding (Machine Learning): First, the user's message is converted into a sophisticated numerical representation (an "embedding") using a SentenceTransformer model. A trained machine learning model then reads this embedding to classify the message into one of four broad parent categories: `FEEDBACK`, `SENSITIVE_EXIT`, `NOISE_SPAM`, or `OTHER`.

2. Specific Details (*Enrichment Rules*): Once the main category is known, the system applies a set of precise rules to determine the specific sub-intent.

    - If the message is `FEEDBACK`, a sentiment analysis model determines if it's a `COMPLIMENT` or `COMPLAINT`.

    - If it's `SENSITIVE_EXIT`, carefully curated patterns detect if it's about `BABY_LOSS` or an `OPTOUT` request.



### Model Development & Training
This section is for anyone involved in managing the data, training the model, or evaluating its performance.

**Data & Schema**

The ground truth for this model lives in `src/data/nlu.yaml`. This file contains all the training examples, which are then processed by `src/data/build_datasets.py` to create the final training files.

  - To regenerate the training data from the source YAML files, run
    ```bash
    make datasets  # Which runs: poetry run python src/data/build_datasets.py --emit-jsonl
    ```
    This will create clean .jsonl files (e.g., samples.train.jsonl) in the src/mapped_data/ directory.
    
  - To train a new version of the model, run
    ```bash
      make train  # Which runs: poetry run python src/train_model.py
      ```
    This script executes the complete pipeline: it loads the processed data, encodes the text, trains the classifier, and saves all model artifacts to the `src/artifacts/ directory`.


### Evaluation & Threshold Tuning

Model performance is measured against a hold-out test set (`test.yaml`). Confidence thresholds are defined in `src/artifacts/thresholds.json` and can be tuned by analyzing the model's performance on the validation set to balance the needs of each intent (e.g., prioritizing recall for `SENSITIVE_EXIT`).


### API, Deployment & Integration
This section is for engineers responsible for deploying and integrating the service, and for QA who need to test the API.

**Setup and Installation**

- Install dependencies:
    ```bash
    make install # Which runs: poetry install
    ```

- Activate the virtual environment:
    ```
    poetry shell
    ```


#### Running the API Service
The application is a standard Flask service. Do not use the built-in Flask development server for production.

- For production or staging, use a WSGI server like Gunicorn:

    ```
    gunicorn --workers 2 --bind 0.0.0.0:5001 src.application:app
    ```

- For local development, you can use the Flask dev server:

    ```
     poetry run flask --app src.application run 
    ```

    _You will need to set the `NLU_USERNAME` and `NLU_PASSWORD` environment variables for authentication._


#### API Endpoints
The service provides two specific, authenticated endpoints. Authentication is handled via HTTP Basic Auth.

1. **Baby Loss Detection**

    This endpoint analyzes a message to determine if it relates to baby loss. It is optimized for high recall to ensure sensitive cases are not missed.

    - Request: GET /nlu/babyloss/
    - Query Parameters
  
          Parameter  Type   Required  Description
          question   string  Yes      The user's message text to be classified.
   

    - Responses:

        - `200 OK` (Success): The babyloss key will be true if the intent is detected, and false otherwise.

              {
                "babyloss": true,
                "model_version": "2025-09-29-v1",
                "parent_label": "SENSITIVE_EXIT",
                "sub_intent": "BABY_LOSS",
                "probability": 0.98,
                "review_status": "CLASSIFIED"
              }
        - `400 Bad Request`: Returned if the question parameter is missing.

2. **Feedback Detection**

    This endpoint analyzes a message to determine if it is a compliment or a complaint.

    - Request: GET /nlu/feedback/
    - Query Parameters
  
          Parameter  Type   Required  Description
          question   string  Yes      The user's message text to be classified.
   

    - Responses:

        - `200 OK`: The primary intent key will be COMPLIMENT, COMPLAINT, or None.

              {
                "intent": "COMPLIMENT",
                "model_version": "2025-09-29-v1",
                "parent_label": "FEEDBACK",
                "probability": 0.99,
                "review_status": "CLASSIFIED"
              }

        - `401 Unauthorized`: Returned for any endpoint if authentication is incorrect.


###  Testing and Code Quality
Run these commands to ensure code quality and correctness.

- Run Unit Tests:
  ```
  make test  # Which runs: poetry run pytest -vv
  ```

- Run Static Type Checking:
    ```
    make typecheck  # Which runs: poetry run mypy .
    ```

- Run Linter/Formatter:
    ```
    make lint  # Which runs: poetry run ruff check --fix . && poetry run ruff format .
    ```


### Production Deployment
For production, the Dockerfile should be configured to run the application using the Gunicorn command. The number of workers (--workers 4) should be adjusted based on the resources of the environment. The `src/artifacts directory`, which contains the trained model, must be included in the final container image.