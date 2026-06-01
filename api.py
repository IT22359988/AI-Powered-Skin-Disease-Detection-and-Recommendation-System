import io
import os
import json
import pickle
import re
from typing import Optional

import faiss
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

# ---------------- CONFIG ----------------
IMG_SIZE = 224
MODEL_PATH = "model/skinfusionnet_final.keras"

diseaseLabels = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevus",
    "Vascular Lesion",
]

stageLabels = [
    "Mild",
    "Moderate",
    "Severe",
]

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

app = FastAPI(title="Skin Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD MODEL ----------------
try:
    model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")


# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ---------------- PREDICTION ----------------
def predict_image(image: Image.Image):
    img = preprocess_image(image)
    preds = model.predict({"input_image": img}, verbose=0)

    disease_probs = preds["disease"][0]
    stage_probs = preds["stage"][0]

    disease_idx = int(np.argmax(disease_probs))
    stage_idx = int(np.argmax(stage_probs))

    return {
        "disease": diseaseLabels[disease_idx],
        "disease_confidence": round(float(disease_probs[disease_idx]) * 100, 2),
        "stage": stageLabels[stage_idx],
        "stage_confidence": round(float(stage_probs[stage_idx]) * 100, 2),
        "all_disease_probs": {
            diseaseLabels[i]: round(float(disease_probs[i]) * 100, 2)
            for i in range(len(diseaseLabels))
        },
        "all_stage_probs": {
            stageLabels[i]: round(float(stage_probs[i]) * 100, 2)
            for i in range(len(stageLabels))
        },
    }


# ---------------- VECTOR DB ----------------
def load_vector_db():
    index_path = "RAG/derm_vectors.index"
    chunks_path = "RAG/derm_chunks.pkl"

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return None, None, None

    index = faiss.read_index(index_path)

    with open(chunks_path, "rb") as f:
        data = pickle.load(f)

    chunks = data.get("chunks", [])
    metadata = data.get("metadata", [])

    return index, chunks, metadata


def retrieve_context(question: str, top_k: int = 5):
    index, chunks, metadata = load_vector_db()

    if index is None or not chunks:
        return None, []

    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    q_vec = np.array(emb.data[0].embedding, dtype=np.float32).reshape(1, -1)

    _, idxs = index.search(q_vec, top_k)

    selected_chunks = []
    selected_meta = []

    for i in idxs[0]:
        if 0 <= i < len(chunks):
            selected_chunks.append(chunks[i])
            if i < len(metadata):
                selected_meta.append(metadata[i])

    context = "\n\n".join(selected_chunks)
    return context, selected_meta


# ---------------- RAG QA ----------------
def ask_question_silent(question: str):
    context, _ = retrieve_context(question, top_k=5)

    if not context:
        return "Dermatology knowledge base is not available."

    reply = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an educational dermatology assistant for general users. "
                    "You must not provide diagnosis, prescriptions, medication dosages, "
                    "or treatment instructions. "
                    "For moderate or severe conditions, emphasize the importance of "
                    "consulting a qualified dermatologist. "
                    "Answer only using the provided context. "
                    "If the context is insufficient, say so clearly."
                )
            },
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n{context}\n\n"
                    f"QUESTION: {question}\n\n"
                    "Provide an educational, non-prescriptive answer."
                )
            }
        ]
    )

    return reply.choices[0].message.content.strip()


# ---------------- RAW INSIGHTS GENERATION ----------------
def generate_raw_insights(disease: str, stage: str, stage_conf: float):
    raw = {
        "summary": ask_question_silent(f"What is {disease} in dermatology?"),
        "definition": ask_question_silent(f"Explain {disease} in simple educational terms."),
        "causes": ask_question_silent(f"What causes {disease}?"),
        "symptoms": ask_question_silent(f"What are the common symptoms of {disease}?"),
        "red_flags": ask_question_silent(
            f"What warning signs or red flags should people watch for in {disease}?"
        ),
        "next_steps": ask_question_silent(
            f"What are sensible next steps for someone with possible {disease} "
            f"for educational purposes only?"
        ),
    }

    if stage == "Mild":
        raw["self_care"] = ask_question_silent(
            f"What are general safe self-care and observation tips for {disease} "
            f"for educational purposes only?"
        )
        raw["when_to_seek_care"] = ask_question_silent(
            f"When should someone seek professional dermatology evaluation for {disease}?"
        )
    else:
        raw["self_care"] = ask_question_silent(
            f"What are safe educational self-monitoring and skin-protection suggestions "
            f"for {disease}, without giving treatment instructions?"
        )
        raw["when_to_seek_care"] = ask_question_silent(
            f"Why is dermatologist evaluation important for {disease} at {stage} stage?"
        )

    return {
        "disease": disease,
        "stage": stage,
        "stage_confidence": stage_conf,
        "raw_sections": raw,
        "rag_used": True,
    }


# ---------------- FALLBACK FORMATTERS ----------------
def split_into_points(text: str):
    if not text:
        return []

    # split by lines first
    parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*\u2022\d\.\)\(]+\s*", "", line).strip()
        if line:
            parts.append(line)

    if len(parts) >= 2:
        return parts

    # fallback sentence split
    sentence_parts = re.split(r"(?<=[.!?])\s+", text.strip())
    sentence_parts = [s.strip(" -•\n\t") for s in sentence_parts if s.strip()]
    if len(sentence_parts) >= 2:
        return sentence_parts[:6]

    return [text.strip()]


def ensure_string(value: Optional[str], fallback: str):
    if value is None:
        return fallback
    value = value.strip()
    return value if value else fallback


# ---------------- DISPLAY REFINEMENT ----------------
def refine_insights_for_display(raw_payload: dict, disease_confidence: Optional[float]):
    raw_sections = raw_payload["raw_sections"]
    disease = raw_payload["disease"]
    stage = raw_payload["stage"]
    stage_confidence = raw_payload["stage_confidence"]

    prompt = f"""
You are converting raw dermatology educational content into a clean mobile-app reading format.

Rules:
- Educational only
- Do NOT provide diagnosis
- Do NOT provide medication dosages
- Do NOT provide treatment instructions
- Use short readable sentences
- Preserve meaning from the source text
- If information is uncertain or incomplete, keep it cautious
- Return valid JSON only, with no markdown fences
- "summary", "definition", and "causes" must be short readable paragraphs
- "symptoms", "self_care", "red_flags", "when_to_seek_care", and "next_steps" must be clean bullet-style arrays
- Keep list items concise and user-friendly
- Do not invent facts not supported by the raw content

Return this exact JSON schema:
{{
  "summary": "string",
  "definition": "string",
  "causes": "string",
  "symptoms": ["string", "string"],
  "self_care": ["string", "string"],
  "red_flags": ["string", "string"],
  "when_to_seek_care": ["string", "string"],
  "next_steps": ["string", "string"]
}}

Disease: {disease}
Stage: {stage}
Stage confidence: {stage_confidence}
Disease confidence: {disease_confidence}

RAW CONTENT:
{json.dumps(raw_sections, ensure_ascii=False)}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You format educational dermatology content for a mobile app. "
                        "Return JSON only."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = response.choices[0].message.content.strip()
        formatted = json.loads(content)

        return {
            "disease": disease,
            "disease_confidence": disease_confidence,
            "stage": stage,
            "stage_confidence": stage_confidence,
            "summary": ensure_string(formatted.get("summary"), raw_sections.get("summary", "")),
            "definition": ensure_string(formatted.get("definition"), raw_sections.get("definition", "")),
            "causes": ensure_string(formatted.get("causes"), raw_sections.get("causes", "")),
            "symptoms": formatted.get("symptoms", []),
            "self_care": formatted.get("self_care", []),
            "red_flags": formatted.get("red_flags", []),
            "when_to_seek_care": formatted.get("when_to_seek_care", []),
            "next_steps": formatted.get("next_steps", []),
            "rag_used": True,
        }

    except Exception:
        # Safe fallback if JSON parsing or model formatting fails
        return {
            "disease": disease,
            "disease_confidence": disease_confidence,
            "stage": stage,
            "stage_confidence": stage_confidence,
            "summary": ensure_string(raw_sections.get("summary"), "No summary available."),
            "definition": ensure_string(raw_sections.get("definition"), "No definition available."),
            "causes": ensure_string(raw_sections.get("causes"), "No cause information available."),
            "symptoms": split_into_points(raw_sections.get("symptoms", "")),
            "self_care": split_into_points(raw_sections.get("self_care", "")),
            "red_flags": split_into_points(raw_sections.get("red_flags", "")),
            "when_to_seek_care": split_into_points(raw_sections.get("when_to_seek_care", "")),
            "next_steps": split_into_points(raw_sections.get("next_steps", "")),
            "rag_used": True,
        }


# ---------------- ROUTES ----------------
@app.get("/")
def root():
    return {"message": "Skin Disease Detection API is running"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        prediction = predict_image(image)

        raw_insights = generate_raw_insights(
            disease=prediction["disease"],
            stage=prediction["stage"],
            stage_conf=prediction["stage_confidence"],
        )

        refined_insights = refine_insights_for_display(
            raw_payload=raw_insights,
            disease_confidence=prediction["disease_confidence"],
        )

        return {
            "disease": prediction["disease"],
            "disease_confidence": prediction["disease_confidence"],
            "stage": prediction["stage"],
            "stage_confidence": prediction["stage_confidence"],
            "all_disease_probs": prediction["all_disease_probs"],
            "all_stage_probs": prediction["all_stage_probs"],
            "insights_ready": True,
            "insights": refined_insights,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analyze failed: {str(e)}")