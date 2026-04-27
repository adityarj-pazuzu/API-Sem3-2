# Project Summary: Heart Health AI Assistant

## 1. Domain and Problem

This project is in the healthcare AI domain, focused on cardiovascular disease
(CVD) risk understanding and educational support.

The problem addressed is: how to combine predictive analytics and document-grounded
question answering in one simple application so users can:

1. Estimate heart disease risk from patient attributes.
2. Compare a patient with similar historical cases.
3. Ask CVD questions and receive answers grounded in trusted documents.

The app is for educational and assignment demonstration purposes only. It is not
a diagnostic or treatment system.

## 2. Dataset

Primary dataset:

```text
data/heart.csv
```

Target variable:

```text
HeartDisease
```

Class meaning:

- 1: heart disease present
- 0: normal

Main feature groups:

- Demographic and baseline: Age, Sex
- Clinical measurements: RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak
- Categorical clinical signals: ChestPainType, RestingECG, ExerciseAngina, ST_Slope

## 3. Unified Objective and Subtasks

All subtasks support one unified objective: practical heart health decision support
from tabular data plus evidence-grounded text knowledge.

### Subtask 1: Heart Disease Classification

Goal:

Predict the likelihood of heart disease for the current input patient profile.

Method:

- Preprocess numeric features with StandardScaler.
- Encode categorical features with OneHotEncoder.
- Train a RandomForestClassifier in a pipeline.
- Output both class label and probability.

User-facing output in app:

- Predicted class (Heart disease or Normal)
- Estimated likelihood
- Evaluation metrics (Accuracy, F1, ROC-AUC)

### Subtask 2: Similar Patient Retrieval

Goal:

Retrieve historical patient records that are most similar to the current input.

Method:

- Reuse the same trained preprocessing pipeline from classification.
- Transform both dataset records and current patient input.
- Compute pairwise distances.
- Return top-k nearest patient records with similarity scores.

User-facing output in app:

- Similar cases table
- Similarity value per case
- Outcome label per case
- Heart disease rate among retrieved similar cases

### Subtask 3: RAG Question Answering

Goal:

Answer CVD questions using local reference documents, then generate a readable
final response.

Method:

- Ingest and chunk local CVD documents.
- Retrieve relevant chunks with TF-IDF vector search.
- Build a prompt from question + retrieved context.
- Generate answer with a Hugging Face model.
- Use local extractive fallback if generation is unavailable.

## 4. Models Used

### A. Fine-tuned ML model (tabular prediction)

- Model family: RandomForestClassifier
- Pipeline: ColumnTransformer + model
- Training script: train_model.py
- Saved artifact: models/heart_rf_tuned.joblib

### B. Similarity model (case retrieval)

- Representation: preprocessed feature vectors from the same classification pipeline
- Distance metric: pairwise distances from scikit-learn

### C. RAG generation model

- Framework: Hugging Face Transformers
- Default model: google/flan-t5-small (configurable via HF_MODEL)
- Loading strategy: AutoConfig + AutoTokenizer + AutoModelForSeq2SeqLM or
  AutoModelForCausalLM, then generate()

## 5. How NLP Is Used

NLP is used in the RAG subsystem in two stages:

1. Retrieval NLP:
- The local CVD corpus is split into paragraph chunks.
- TfidfVectorizer converts chunks and user query into vector space.
- Similarity scoring identifies the most relevant context chunks.

2. Generation NLP:
- Retrieved chunks are assembled into a grounded prompt.
- A Hugging Face language model generates the final natural-language answer.
- Prompt constraints guide concise, context-based output and discourage unsupported claims.

This combination is Retrieval-Augmented Generation, where retrieval improves
factual grounding and reduces hallucination risk.

## 6. Tools and Libraries Used in the Application

Core application:

- Streamlit: interactive web UI and deployment-ready app structure

Data handling:

- pandas: data loading and tabular manipulation
- numpy: numerical operations

Machine learning:

- scikit-learn:
  - ColumnTransformer
  - StandardScaler
  - OneHotEncoder
  - RandomForestClassifier
  - GridSearchCV
  - TfidfVectorizer
  - pairwise_distances
  - train_test_split and evaluation metrics

Model persistence:

- joblib: serialize and load tuned model

Document processing:

- PyPDF2: optional PDF text extraction support

LLM integration:

- transformers: tokenizer/model loading and text generation
- torch: backend runtime for model inference

Configuration:

- python-dotenv: optional environment variable loading from .env

## 7. Model Fine Tuning

Fine tuning is implemented as hyperparameter optimization of the classical ML model.

Approach:

- Baseline estimator: RandomForestClassifier(class_weight="balanced")
- Search method: GridSearchCV with 5-fold cross-validation
- Optimized score: F1
- Tuned parameters:
  - classifier__n_estimators
  - classifier__max_depth
  - classifier__min_samples_leaf

After search, the best estimator is saved and reused by the Streamlit app to avoid
retraining on every run.

## 8. RAG Model and Documents Used

### Retrieval documents

Local knowledge base path:

```text
rag_docs/
```

Current CVD reference files:

```text
rag_docs/CVD_CDC.txt
rag_docs/CVD_WHO.txt
```

### RAG model behavior

1. Retrieve top relevant chunks from local CVD documents.
2. Compose prompt with explicit context and user question.
3. Generate answer with Hugging Face model.
4. If model load/inference fails, return local fallback based on retrieved text.

This design keeps answers domain-relevant while preserving robustness when model
dependencies or downloads are unavailable.

## 9. Key Files in This Project

```text
app.py                    # Streamlit app: prediction + retrieval + RAG QA
train_model.py            # Training and fine-tuning script
data/heart.csv            # Tabular heart disease dataset
models/heart_rf_tuned.joblib
rag_docs/CVD_CDC.txt
rag_docs/CVD_WHO.txt
requirements.txt
```
