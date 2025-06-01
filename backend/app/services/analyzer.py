# analyzer.py

import re
from typing import Dict, List
from app.services.query_faiss import query_faiss
from app.services.medical_search_tool import medical_web_search

def extract_medical_values(text: str) -> Dict:
    patterns = {
        "blood_pressure": [r'(?:bp|blood pressure)[\s:]*?(\d{2,3})[/-](\d{2,3})'],
        "glucose": [r'(?:glucose|sugar|fbs|rbs)[\s:]*(\d+\.?\d*)'],
        "cholesterol": [r'(?:cholesterol|chol)[\s:]*(\d+\.?\d*)'],
        "hemoglobin": [r'(?:hemoglobin|hb)[\s:]*(\d+\.?\d*)'],
        "temperature": [r'(?:temp|temperature)[\s:]*(\d+\.?\d*)'],
        "heart_rate": [r'(?:hr|pulse|heart rate)[\s:]*(\d+)']
    }
    results = {}
    lower = text.lower()

    for key, regexes in patterns.items():
        for pattern in regexes:
            match = re.search(pattern, lower)
            if match:
                results[key] = "/".join(match.groups()) if len(match.groups()) > 1 else match.group(1)
                break
    return results

def interpret_medical_values(values: Dict) -> Dict:
    findings: List[str] = []
    recommendations: List[str] = []

    bp = values.get("blood_pressure", "")
    if bp:
        # Expect "systolic/diastolic"
        try:
            s, d = map(int, bp.split('/'))
            if s > 140 or d > 90:
                findings.append(f"High BP: {bp}")
                recommendations.append("Check for hypertension")
            else:
                findings.append(f"BP Normal: {bp}")
        except ValueError:
            # If parsing fails, just record the raw BP
            findings.append(f"Blood pressure reading: {bp}")

    if "glucose" in values:
        try:
            g = float(values["glucose"])
            if g > 126:
                findings.append(f"High Glucose: {g}")
                recommendations.append("Consider diabetes assessment")
            else:
                findings.append(f"Glucose Normal: {g}")
        except ValueError:
            findings.append(f"Glucose reading: {values['glucose']}")

    if "cholesterol" in values:
        try:
            c = float(values["cholesterol"])
            if c > 200:
                findings.append(f"High Cholesterol: {c}")
                recommendations.append("Assess lipid profile and lifestyle")
            else:
                findings.append(f"Cholesterol Normal: {c}")
        except ValueError:
            findings.append(f"Cholesterol reading: {values['cholesterol']}")

    if "hemoglobin" in values:
        try:
            h = float(values["hemoglobin"])
            if h < 12:
                findings.append(f"Low Hemoglobin: {h}")
                recommendations.append("Evaluate for anemia")
            else:
                findings.append(f"Hemoglobin Normal: {h}")
        except ValueError:
            findings.append(f"Hemoglobin reading: {values['hemoglobin']}")

    if "temperature" in values:
        try:
            t = float(values["temperature"])
            if t > 100.4:
                findings.append(f"Fever detected: {t}°F")
                recommendations.append("Check for infection or inflammation")
            else:
                findings.append(f"Temperature Normal: {t}°F")
        except ValueError:
            findings.append(f"Temperature reading: {values['temperature']}")

    if "heart_rate" in values:
        try:
            hr = int(values["heart_rate"])
            if hr > 100:
                findings.append(f"High Heart Rate: {hr} bpm")
                recommendations.append("Assess for tachycardia causes")
            else:
                findings.append(f"Heart Rate Normal: {hr} bpm")
        except ValueError:
            findings.append(f"Heart rate reading: {values['heart_rate']}")

    return {"findings": findings, "recommendations": recommendations}


def analyze_medical_report(
    text: str,
    symptoms: List[str],
    medical_history: str
) -> Dict:
    """
    Analyze the given medical report text, incorporate reported symptoms and medical history,
    and return a dictionary with:
      - extracted_values
      - findings
      - recommendations
      - faiss_context
      - additional_info
      - raw_text_preview
      - reported_symptoms
      - medical_history
    """

    # 1. Extract numeric/clinical values (BP, glucose, etc.)
    values = extract_medical_values(text)
    interpretations = interpret_medical_values(values)

    # 2. Print out recommendations to console (for debugging/logging)
    if interpretations["recommendations"]:
        print("recommendations:\n" + "\n".join(interpretations["recommendations"]))

    # 3. Run a FAISS query on the first 500 chars of text (to get local context)
    faiss_context = query_faiss(text[:500])

    # 4. Check for a few high-level conditions in the text
    conditions = re.findall(r'(diabetes|hypertension|anemia|asthma|infection)', text.lower())
    additional_info: List[str] = []
    for c in conditions[:2]:
        result = medical_web_search(c.strip())
        if "No reliable" not in result:
            snippet = result[:200] + "..." if len(result) > 200 else result
            additional_info.append(f"About {c}: {snippet}")

    # 5. Incorporate reported symptoms into "findings" if any
    if symptoms:
        for symptom in symptoms:
            interpretations["findings"].append(f"Patient reports symptom: {symptom}")

    # 6. If medical history was provided, append a note under additional_info
    if medical_history and medical_history.strip():
        additional_info.append(f"Medical history provided: {medical_history.strip()[:200]}{'...' if len(medical_history) > 200 else ''}")

    # 7. Build and return the final dictionary
    return {
        "extracted_values": values,
        "findings": interpretations["findings"],
        "recommendations": interpretations["recommendations"],
        "faiss_context": faiss_context[:2],       # return top‐2 FAISS contexts
        "additional_info": additional_info,
        "raw_text_preview": text[:300] + "..." if len(text) > 300 else text,
        "reported_symptoms": symptoms,
        "medical_history": medical_history or ""
    }
