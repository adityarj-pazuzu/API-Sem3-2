from __future__ import annotations

import re
import os
from pathlib import Path

# Avoid a known Streamlit watcher interaction with transformers lazy imports.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_PATH = BASE_DIR / "data" / "heart.csv"
DOCS_DIR = BASE_DIR / "rag_docs"
MODEL_PATH = BASE_DIR / "models" / "heart_rf_tuned.joblib"
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-small")
CATEGORICAL_COLUMNS = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
]
NUMERIC_COLUMNS = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "Oldpeak",
]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42, class_weight="balanced")),
        ]
    )


@st.cache_resource
def train_or_load_model() -> tuple[Pipeline, dict[str, float | str]]:
    df = load_data()
    x = df[FEATURE_COLUMNS]
    y = df["HeartDisease"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        source = "saved fine-tuned model"
    else:
        model = build_model()
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [4, 8, None],
            "classifier__min_samples_leaf": [1, 3, 5],
        }
        search = GridSearchCV(model, param_grid, cv=5, scoring="f1", n_jobs=1)
        search.fit(x_train, y_train)
        model = search.best_estimator_
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        source = "trained and fine-tuned now"

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "source": source,
    }
    return model, metrics


@st.cache_data
def load_rag_chunks() -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for idx, paragraph in enumerate(paragraphs):
            if len(paragraph) > 80:
                chunks.append({"source": path.name, "chunk": str(idx + 1), "text": paragraph})
    for path in sorted(DOCS_DIR.glob("*.pdf")):
        try:
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
                for idx, paragraph in enumerate(paragraphs[:8], start=1):
                    if len(paragraph) > 80:
                        chunks.append(
                            {
                                "source": f"{path.name}, page {page_number}",
                                "chunk": str(idx),
                                "text": paragraph,
                            }
                        )
        except Exception:
            continue
    return chunks


@st.cache_resource
def build_retriever() -> tuple[TfidfVectorizer, np.ndarray, list[dict[str, str]]]:
    chunks = load_rag_chunks()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
    return vectorizer, matrix, chunks


def ask_rag(question: str, top_k: int = 3) -> tuple[str, list[dict[str, str]]]:
    if not question.strip():
        return "Ask a cardiovascular health question to search the knowledge base.", []

    vectorizer, matrix, chunks = build_retriever()
    query_vector = vectorizer.transform([question])
    scores = (matrix @ query_vector.T).toarray().ravel()
    best_indices = scores.argsort()[::-1][:top_k]
    retrieved = [chunks[i] | {"score": f"{scores[i]:.3f}"} for i in best_indices if scores[i] > 0]

    if not retrieved:
        return "I could not find a relevant answer in the local CVD documents.", []

    answer_parts = []
    for item in retrieved:
        sentences = re.split(r"(?<=[.!?])\s+", item["text"])
        answer_parts.append(sentences[0])

    answer = " ".join(answer_parts)
    return answer, retrieved


@st.cache_resource
def load_huggingface_generator():
    try:
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )
    except Exception as exc:
        return None, f"Hugging Face transformers is not available: {exc}"

    try:
        config = AutoConfig.from_pretrained(HF_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

        if getattr(config, "is_encoder_decoder", False):
            model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL)
            model_type = "seq2seq"
        else:
            model = AutoModelForCausalLM.from_pretrained(HF_MODEL)
            model_type = "causal"

        model.eval()
        return (tokenizer, model, model_type), None
    except Exception as exc:
        return None, f"Hugging Face model could not be loaded: {exc}"


def call_huggingface(prompt: str) -> tuple[str | None, str | None]:
    loaded, error = load_huggingface_generator()
    if error:
        return None, error

    try:
        import torch

        tokenizer, model, model_type = loaded
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=140, do_sample=False)

        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if model_type == "causal" and decoded.startswith(prompt):
            text = decoded[len(prompt) :].strip()
        else:
            text = decoded

        if text:
            return text, None
        return None, "Hugging Face model returned an empty response."
    except Exception as exc:
        return None, f"Hugging Face generation failed: {exc}"


def make_patient_input() -> pd.DataFrame:
    with st.sidebar:
        st.header("Patient Inputs")
        age = st.slider("Age", 18, 90, 52)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest pain type", ["ASY", "NAP", "ATA", "TA"])
        resting_bp = st.slider("Resting blood pressure", 80, 220, 130)
        cholesterol = st.slider("Cholesterol", 0, 650, 220)
        fasting_bs = st.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Maximum heart rate", 60, 202, 145)
        exercise_angina = st.selectbox("Exercise angina", ["N", "Y"])
        oldpeak = st.slider("Oldpeak ST depression", 0.0, 7.0, 1.0, 0.1)
        st_slope = st.selectbox("ST slope", ["Up", "Flat", "Down"])

    return pd.DataFrame(
        [
            {
                "Age": age,
                "Sex": sex,
                "ChestPainType": chest_pain,
                "RestingBP": resting_bp,
                "Cholesterol": cholesterol,
                "FastingBS": fasting_bs,
                "RestingECG": resting_ecg,
                "MaxHR": max_hr,
                "ExerciseAngina": exercise_angina,
                "Oldpeak": oldpeak,
                "ST_Slope": st_slope,
            }
        ]
    )


def find_similar_patients(
    model: Pipeline, patient_df: pd.DataFrame, top_k: int = 5
) -> pd.DataFrame:
    df = load_data()
    transformed_dataset = model.named_steps["preprocess"].transform(df[FEATURE_COLUMNS])
    transformed_patient = model.named_steps["preprocess"].transform(patient_df[FEATURE_COLUMNS])
    distances = pairwise_distances(transformed_dataset, transformed_patient).ravel()
    nearest_indices = distances.argsort()[:top_k]

    similar = df.iloc[nearest_indices].copy()
    similar.insert(0, "Similarity", 1 / (1 + distances[nearest_indices]))
    similar["Outcome"] = similar["HeartDisease"].map({0: "Normal", 1: "Heart disease"})
    display_columns = [
        "Similarity",
        "Age",
        "Sex",
        "ChestPainType",
        "RestingBP",
        "Cholesterol",
        "MaxHR",
        "ExerciseAngina",
        "Oldpeak",
        "ST_Slope",
        "Outcome",
    ]
    return similar[display_columns]


def generate_huggingface_rag_answer(question: str, sources: list[dict[str, str]]) -> tuple[str, str]:
    if not sources:
        return "I could not find a relevant answer in the local CVD documents.", "Local retriever"

    context = "\n\n".join(
        f"Source: {item['source']} chunk {item['chunk']}\n{item['text']}" for item in sources
    )
    prompt = f"""
Answer this cardiovascular disease question using only the context.
Give a direct answer, then add one short expanded explanation.
Do not give diagnosis or personalized treatment.

Question: {question}

Context:
{context}

Answer:
"""
    text, error = call_huggingface(prompt)
    if text:
        return text, f"Hugging Face local model ({HF_MODEL}) with local RAG context"

    fallback_answer = " ".join(re.split(r"(?<=[.!?])\s+", item["text"])[0] for item in sources)
    return fallback_answer, f"Local fallback ({error})"


st.set_page_config(page_title="Heart Health AI Assistant", page_icon="heart", layout="wide")
st.title("Heart Health AI Assistant")
st.caption("Prediction, similar-case retrieval, and RAG question answering in one small AI application.")

model, metrics = train_or_load_model()
patient_df = make_patient_input()
probability = float(model.predict_proba(patient_df)[0, 1])
prediction = int(probability >= 0.5)

left, right = st.columns([1, 1])
with left:
    st.subheader("Sub-task 1: Heart Disease Classification")
    st.metric("Predicted class", "Heart disease" if prediction else "Normal")
    st.metric("Estimated likelihood", f"{probability:.1%}")
    st.progress(probability)

    st.subheader("Fine-tuned Model Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    c2.metric("F1 score", f"{metrics['f1']:.3f}")
    c3.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    st.caption(f"Model source: {metrics['source']}")

with right:
    st.subheader("Sub-task 2: Similar Patient Retrieval")
    st.write(
        "This local retrieval task finds dataset records most similar to the current patient input "
        "using the same preprocessing features used by the classifier."
    )
    similar_patients = find_similar_patients(model, patient_df)
    heart_disease_rate = similar_patients["Outcome"].eq("Heart disease").mean()
    st.metric("Heart disease rate among similar cases", f"{heart_disease_rate:.0%}")
    st.dataframe(
        similar_patients.style.format({"Similarity": "{:.3f}"}),
        hide_index=True,
        use_container_width=True,
    )

st.divider()
st.subheader("Sub-task 3: RAG Question Answering")
st.markdown(
    """
    This chatbot can answer questions grounded in the local CDC/WHO cardiovascular
    disease documents, such as:

    - What are the major risk factors for cardiovascular disease?
    - What symptoms can occur during a heart attack?
    - How can cardiovascular disease risk be reduced?
    - What is cardiac rehabilitation?
    - Why is early detection of cardiovascular disease important?
    """
)
question = st.text_input(
    "Ask about heart disease symptoms, risk factors, prevention, or rehabilitation",
    value="What are the major risk factors for cardiovascular disease?",
)
answer, sources = ask_rag(question)
if sources:
    answer, answer_source = generate_huggingface_rag_answer(question, sources)
else:
    answer_source = "Local retriever"
st.write(answer)
st.caption(f"Answered by: {answer_source}")

if sources:
    with st.expander("Retrieved context"):
        for item in sources:
            st.markdown(f"**{item['source']} - chunk {item['chunk']} - score {item['score']}**")
            st.write(item["text"])

st.info(
    "Assignment mapping: classification uses a fine-tuned Random Forest SLM-style model, "
    "Sub-task 2 retrieves similar patient records, and Hugging Face is used for the RAG chatbot answer."
)
