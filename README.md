# Heart Health AI Assistant

This is a small assignment-ready AI application built around `data/heart.csv`.
It keeps the scope simple while covering the required parts: multiple AI
sub-tasks, fine tuning, Hugging Face integration, and RAG.

## Unified Objective

The project acts as a heart health decision-support demo. A user enters patient
measurements, receives a heart disease prediction, sees similar historical
patient records from the dataset, and can ask cardiovascular health questions
answered from local CDC/WHO documents using RAG and a Hugging Face model.

This is for academic demonstration only and is not a medical diagnosis tool.

For a full objective-by-objective explanation, see `PROJECT_SUMMARY.md`.

## Selected Sub-tasks

1. **Classification**: predict `HeartDisease` from tabular patient attributes.
2. **Similar Patient Retrieval**: find historical records in `heart.csv` that
   are closest to the current patient input after model preprocessing.
3. **Question Answering with Hugging Face + RAG**: retrieve CVD context from
   `rag_docs` and pass it to a Hugging Face text generation model for a grounded
   answer.

## Models Used

- **Fine-tuned small ML model**: Random Forest classifier with preprocessing and
  GridSearchCV hyperparameter tuning on `heart.csv`.
- **Hugging Face LLM/SLM**: default model is `google/flan-t5-small`, a free
  public instruction-tuned model that runs locally through `transformers`.
- **Similarity retriever**: nearest-neighbor matching over preprocessed patient
  features to show comparable records and their outcomes.
- **RAG retriever**: TF-IDF vector search over local cardiovascular disease text
  and PDF documents.

If the Hugging Face model or SDK is unavailable, the RAG chatbot shows a local
extractive fallback response so the demo does not break.

## Hugging Face Free Model Setup

The app uses the free public Hugging Face model:

```text
google/flan-t5-small
```

This model runs locally using the `transformers` library, so no paid hosted API
key is required. The first run may download the model from Hugging Face and
cache it on your machine.

Optional `.env` configuration:

```text
HF_MODEL=google/flan-t5-small
```

Alternative PowerShell setup for the current terminal:

```powershell
$env:HF_MODEL="google/flan-t5-small"
```

You can replace `HF_MODEL` with another compatible Hugging Face text-to-text
generation model if needed.

## Project Structure

```text
.
|-- app.py
|-- train_model.py
|-- requirements.txt
|-- data/
|   `-- heart.csv
`-- rag_docs/
    |-- CVD_CDC.txt
    |-- CVD_INDICATORS_CDC.pdf
    `-- CVD_WHO.txt
```

## Run

Install dependencies if needed:

```bash
pip install -r requirements.txt
```

Fine tune and save the model:

```bash
python train_model.py
```

Start the interactive app:

```bash
streamlit run app.py
```

## Assignment Mapping

- **Part I**: interactive Streamlit app with three cohesive AI sub-tasks.
- **Part II**: fine tuning via `GridSearchCV` on the heart disease dataset.
- **Part III**: RAG chatbot retrieves local CVD text/PDF knowledge documents and
  uses Hugging Face to generate the final answer.
