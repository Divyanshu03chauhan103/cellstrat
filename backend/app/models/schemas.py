from pydantic import BaseModel
from typing import List, Dict

class SearchRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    faiss_results: List[str]
    web_result: str

class AnalysisResponse(BaseModel):
    extracted_values: Dict[str, str]
    findings: List[str]
    recommendations: List[str]
    faiss_context: List[str]
    additional_info: List[str]
    raw_text_preview: str
