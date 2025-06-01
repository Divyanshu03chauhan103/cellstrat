# endpoint.py

from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form
from app.services.extractor import extract_text_from_pdf
from app.services.analyzer import analyze_medical_report
from app.services.medical_search_tool import medical_web_search
from app.services.query_faiss import query_faiss
from app.models.schemas import SearchRequest, SearchResponse, AnalysisResponse
import json

router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_pdf(
    files: List[UploadFile] = File(None),
    symptoms: Optional[str] = Form(None),
    medical_history: Optional[str] = Form(None),
):
    """
    - Expects multiple files under "files"
    - Expects "symptoms" as a JSON‐encoded list of strings
    - Expects "medical_history" as a plain string
    """
    # 1. Read each uploaded file’s bytes and extract text
    extracted_texts = []
    if files:
        for f in files:
            content = await f.read()
            text = extract_text_from_pdf(content)
            extracted_texts.append(text)

    # 2. Combine all PDF texts into one big string
    combined_text = "\n\n".join(extracted_texts).strip()

    # 3. Parse symptoms JSON (if provided)
    symptom_list = []
    if symptoms:
        try:
            symptom_list = json.loads(symptoms)  # expecting something like '["cough","fever"]'
        except json.JSONDecodeError:
            symptom_list = []

    # 4. Call your analyzer — adapt its signature if needed to accept symptoms & medical_history
    #    For example, if analyze_medical_report(text, symptoms, history) is your function:
    return analyze_medical_report(
        text=combined_text,
        symptoms=symptom_list,
        medical_history=medical_history or ""
    )


@router.post("/search", response_model=SearchResponse)
async def search_medical_info(req: SearchRequest):
    faiss = query_faiss(req.query)
    web = medical_web_search(req.query)
    return {"faiss_results": faiss, "web_result": web}
