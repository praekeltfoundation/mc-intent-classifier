## MomConnect Intent Classifier

A machine learning service to classify inbound user messages for *MomConnect*. This service identifies user intents such as service feedback and sensitive topics like baby loss, enabling the platform to provide appropriate and timely responses.


### How It Works: A Hybrid Approach to Intent Classification
This classifier uses a modern, two-stage process combined with endpoint-specific logic to understand user messages with both accuracy and precision. This design ensures that broad categories for sensitive topics are handled robustly by a data-driven model, while solicited feedback is analyzed directly for sentiment.

**General Classification (e.g., for `/nlu/babyloss/`)**

1.  **Broad Understanding (Machine Learning)**: First, the user's message is converted into a sophisticated numerical representation (an "embedding") using a SentenceTransformer model (`BAAI/bge-m3`). A trained machine learning model (`clf_parent.pkl`) then reads this embedding to classify the message into one of four broad parent categories: `FEEDBACK`, `SENSITIVE_EXIT`, `NOISE_SPAM`, or `OTHER`. Classification confidence is determined by tuned thresholds.

2.  **Specific Details (*Enrichment*)**: Once the main category is known, the system determines the specific sub-intent.
    * If the parent model predicts `SENSITIVE_EXIT` with sufficient confidence, a second, specialized machine learning model (`clf_sensitive_exit.pkl`) determines if it's about `BABY_LOSS` or an `OPTOUT` request.
    * *(Note: The general classifier also uses a sentiment model if FEEDBACK is detected, but this is secondary to the dedicated feedback endpoint's logic)*.

**Dedicated Feedback Analysis (for `/nlu/feedback/`)**

This endpoint assumes the chatbot has already solicited feedback and **bypasses the parent classification model entirely**.

1.  **Direct Sentiment Analysis**: The user's message is fed directly into a pre-trained multilingual sentiment analysis model (`cardiffnlp/twitter-xlm-roberta-base-sentiment`).
2.  **Sentiment Mapping**: The sentiment model's output (`positive`, `negative`, `neutral`) is mapped to `COMPLIMENT`, `COMPLAINT`, or `None` respectively.
3.  **Review Logic**: Predictions are flagged as `NEEDS_REVIEW` if the sentiment model output is `neutral` OR if the model's confidence score falls below a data-driven threshold (`sentiment_review_band`) tuned specifically for this sentiment model.

### Model Development Workflow
This section is for anyone involved in managing the data, training the model, or evaluating its performance.

#### The Data Source of Truth
The primary source of truth for this model lives in the consolidated YAML files within the `src/mapped_data/` directory, for example:
- `src/mapped_data/nlu.yaml`
- `src/mapped_data/validation.yaml`
- `src/mapped_data/test.yaml`

**All new examples and changes should be made directly to these files.** The original legacy files in `src/data/` are kept for historical purposes and are only used for migration.

#### Workflow for Updating Data
This is the standard, safe workflow for improving the model with new data.

1.  **Edit the Mapped YAML Files**: Manually add or modify examples in the `ymal` files under the `mapped_data` folder. You will work directly with the four parent intents (`FEEDBACK`, `SENSITIVE_EXIT`, etc.) and their sub-intents.

2.  **Build the JSONL Files**: After saving your YAML changes, run the following command:
    ```bash
    make build-jsonl
    ```
    This safe command reads your updated YAML files and generates the corresponding `.jsonl` files that the model training scripts consume. It will **never** overwrite your YAML files.

3.  **Train and Evaluate**: Once the JSONL files are up-to-date, you can run the full pipeline or individual steps as needed.
    ```bash
    # Run the entire process: build, train, tune, and evaluate
    make all

    # Or run individual steps
    make train
    make tune-thresholds
    ```

#### Legacy Data Migration (Advanced & Destructive)
> **CAUTION**: This is a destructive operation. Only run this command if you want to discard all manual changes in `src/mapped_data` and regenerate them from the original legacy data in `src/data`.

To perform a full migration from the legacy files, run:
```bash
make migrate-legacy
```


### Evaluation & Threshold Tuning

This is a two-step process to find the optimal confidence thresholds for the parent model and the sentiment model, and then get an unbiased measure of the final system's performance.

  - **Tune Thresholds on the Validation Set**

    Run the evaluation script in "tune" mode. This uses the `samples.validation.jsonl` dataset to find the best confidence threshold for each *parent* intent category and a separate `sentiment_review_band` threshold for the *sentiment model*.

    ```bash
    make tune-thresholds
    ```
   

    This generates the `src/artifacts/thresholds.json` file, which contains both parent thresholds and the sentiment review band. This file is required for the model to run. It also produces a detailed text report and performance plots in the `src/evaluations/` directory.

  - **Evaluate Final Performance on the Test Set**

    Once the models are trained and the thresholds are tuned, run the final performance report. This uses the **hold-out** `samples.test.jsonl` dataset to provide an unbiased measure of how the model will perform on new, unseen data.

    ```bash
    make evaluate
    ```
   

    The output of this command is the definitive performance report for the model version. It evaluates the parent model performance and the sub-intent performance (simulating the separate logic used by each API endpoint).


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

1.  **Baby Loss Detection (`/nlu/babyloss/`)**

    This endpoint analyzes a message using the full classification pipeline to determine if it relates to baby loss. It is optimized for high recall on the `SENSITIVE_EXIT` parent intent to ensure sensitive cases are not missed.

    -   Request: `GET /nlu/babyloss/`
    -   Query Parameters:
        -   `question` (string, required): The user's message text.
    -   Responses:
        -   `200 OK`: Returns whether baby loss was detected, along with model details. The `babyloss` key is `true` only if the parent intent is `SENSITIVE_EXIT` AND the sub-intent is `BABY_LOSS`.
            ```json
            {
              "babyloss": true,
              "model_version": "2025-10-28-v1",
              "parent_label": "SENSITIVE_EXIT",
              "sub_intent": "BABY_LOSS",
              "probability": 0.98,
              "review_status": "CLASSIFIED"
            }
            ```
        -   `400 Bad Request`: If the `question` parameter is missing.
        -   `401 Unauthorized`: If authentication fails.
        -   `503 Service Unavailable`: If the classifier failed to load.

2.  **Feedback Analysis (`/nlu/feedback/`)**

    This endpoint analyzes a message **assuming it is solicited feedback**. It **bypasses the parent model** and directly uses the sentiment model to determine if it is a `COMPLIMENT`, `COMPLAINT`, or `None` (for neutral sentiment).

    -   Request: `GET /nlu/feedback/`
    -   Query Parameters:
        -   `question` (string, required): The user's message text.
    -   Responses:
        -   `200 OK`: Returns the detected sentiment intent (`COMPLIMENT`/`COMPLAINT`/`None`), along with model details. `parent_label` will always be `"FEEDBACK"` for this endpoint. `review_status` becomes `NEEDS_REVIEW` if sentiment is `neutral` or confidence is below `sentiment_review_band`.
            ```json
            {
              "intent": "COMPLIMENT",
              "model_version": "2025-10-28-v1",
              "parent_label": "FEEDBACK",
              "probability": 0.99,
              "review_status": "CLASSIFIED",
              "sentiment_label": "positive"
            }
            ```
            ```json
            {
              "intent": "None",
              "model_version": "2025-10-28-v1",
              "parent_label": "FEEDBACK",
              "probability": 0.65,
              "review_status": "NEEDS_REVIEW",
              "sentiment_label": "neutral"
            }
            ```
        -   `400 Bad Request`: If the `question` parameter is missing.
        -   `401 Unauthorized`: If authentication fails.
        -   `503 Service Unavailable`: If the classifier failed to load.

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