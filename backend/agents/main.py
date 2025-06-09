from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any, Union
import asyncio
import json
import os
import tempfile
import logging
from pathlib import Path
import uvicorn
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Import your diagnostic agent
from diagnostic_agent import MedicalDiagnosticAgent, DiagnosticInput, DiagnosticResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize diagnostic agent (global variable)
diagnostic_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the diagnostic agent on startup"""
    global diagnostic_agent
    
    try:
        gemini_key = os.getenv("GEMINI_API_KEY")
        google_key = os.getenv("GOOGLE_API_KEY")
        google_cse_id = os.getenv("GOOGLE_CSE_ID")
        
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        diagnostic_agent = MedicalDiagnosticAgent(
            gemini_api_key=gemini_key,
            google_api_key=google_key,
            google_cse_id=google_cse_id
        )
        
        logger.info("Medical Diagnostic Agent initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize diagnostic agent: {e}")
        raise
    
    yield  # This is where the app runs
    
    # Cleanup code would go here if needed
    logger.info("Application shutting down")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="MediMind AI Diagnostic API",
    description="AI-powered medical diagnostic assistant API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class DiagnosticRequest(BaseModel):
    symptoms: List[str] = []
    medical_history: str = ""
    age: Optional[int] = None
    gender: Optional[str] = None
    vital_signs: Optional[Dict[str, float]] = None

class SourceInfo(BaseModel):
    """Model for source information"""
    title: str
    url: Optional[str] = None
    source: Optional[str] = None

class DiagnosticResponse(BaseModel):
    primary_diagnosis: str
    confidence_score: float
    differential_diagnoses: List[Dict[str, Any]]
    severity: str
    findings: List[str]
    recommendations: List[str]
    follow_up_questions: List[str]
    first_aid_steps: List[str]
    medications: List[Dict[str, str]]
    emergency_indicators: List[str]
    follow_up: Optional[str] = None
    sources: Optional[List[Union[str, SourceInfo]]] = None  # Allow both strings and source objects

class HealthResponse(BaseModel):
    status: str
    message: str
    agent_initialized: bool

# Helper function to process sources
def process_sources(sources: Optional[List[Any]]) -> Optional[List[str]]:
    """Convert sources to list of strings for the response"""
    if not sources:
        return None
    
    processed_sources = []
    for source in sources:
        if isinstance(source, str):
            processed_sources.append(source)
        elif isinstance(source, dict):
            # Extract meaningful information from dictionary sources
            if 'title' in source and 'url' in source:
                processed_sources.append(f"{source['title']} - {source['url']}")
            elif 'title' in source:
                processed_sources.append(source['title'])
            elif 'url' in source:
                processed_sources.append(source['url'])
            else:
                # If it's a dict but doesn't have expected fields, convert to string
                processed_sources.append(str(source))
        else:
            # For any other type, convert to string
            processed_sources.append(str(source))
    
    return processed_sources

# Helper function to save uploaded files temporarily
async def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file to temporary location and return path"""
    try:
        # Create temporary file
        suffix = Path(file.filename).suffix if file.filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"Saved uploaded file: {file.filename} -> {temp_file_path}")
        return temp_file_path
        
    except Exception as e:
        logger.error(f"Error saving uploaded file {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}"
        )

# Helper function to clean up temporary files
def cleanup_temp_files(file_paths: List[str]):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file {file_path}: {e}")

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="MediMind AI Diagnostic API is running",
        agent_initialized=diagnostic_agent is not None
    )

@app.post("/analyze", response_model=DiagnosticResponse)
async def analyze_diagnostics(
    files: List[UploadFile] = File(default=[]),
    symptoms: str = Form(default="[]"),
    medical_history: str = Form(default=""),
    age: Optional[int] = Form(default=None),
    gender: Optional[str] = Form(default=None),
    vital_signs: str = Form(default="{}")
):
    """
    Analyze medical reports and symptoms to provide diagnostic insights
    """
    if diagnostic_agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Diagnostic agent not initialized"
        )
    
    temp_file_paths = []
    
    try:
        # Parse JSON strings from form data
        try:
            symptoms_list = json.loads(symptoms) if symptoms else []
            vital_signs_dict = json.loads(vital_signs) if vital_signs else {}
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON format: {str(e)}"
            )
        
        # Validate input
        if not files and not symptoms_list and not medical_history.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please provide at least one of: medical reports, symptoms, or medical history"
            )
        
        # Save uploaded files temporarily
        uploaded_report_paths = []
        for file in files:
            if file.filename:  # Skip empty files
                temp_path = await save_uploaded_file(file)
                temp_file_paths.append(temp_path)
                uploaded_report_paths.append(temp_path)
        
        # Create diagnostic input
        diagnostic_input = DiagnosticInput(
            symptoms=symptoms_list,
            medical_history=medical_history,
            uploaded_reports=uploaded_report_paths,
            age=age,
            gender=gender,
            vital_signs=vital_signs_dict if vital_signs_dict else None
        )
        
        logger.info(f"Processing diagnostic request with {len(uploaded_report_paths)} files, "
                   f"{len(symptoms_list)} symptoms, and medical history: {bool(medical_history.strip())}")
        
        # Run diagnosis
        result = await diagnostic_agent.diagnose(diagnostic_input)
        
        # Process sources to ensure they're strings
        processed_sources = process_sources(result.sources)
        
        # Convert result to response format
        response = DiagnosticResponse(
            primary_diagnosis=result.primary_diagnosis,
            confidence_score=result.confidence_score,
            differential_diagnoses=result.differential_diagnoses,
            severity=result.severity,
            findings=result.findings,
            recommendations=result.recommendations,
            follow_up_questions=result.follow_up_questions,
            first_aid_steps=result.first_aid_steps,
            medications=result.medications,
            emergency_indicators=result.emergency_indicators,
            follow_up=result.follow_up,
            sources=processed_sources  # Use processed sources
        )
        
        logger.info(f"Diagnostic analysis completed successfully. Primary diagnosis: {result.primary_diagnosis}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during diagnostic analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during analysis: {str(e)}"
        )
    finally:
        # Clean up temporary files
        cleanup_temp_files(temp_file_paths)

@app.post("/analyze-symptoms", response_model=DiagnosticResponse)
async def analyze_symptoms_only(request: DiagnosticRequest):
    """
    Analyze symptoms without file uploads (for quick symptom checking)
    """
    if diagnostic_agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Diagnostic agent not initialized"
        )
    
    try:
        # Validate input
        if not request.symptoms and not request.medical_history.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please provide symptoms or medical history"
            )
        
        # Create diagnostic input
        diagnostic_input = DiagnosticInput(
            symptoms=request.symptoms,
            medical_history=request.medical_history,
            uploaded_reports=[],  # No files for this endpoint
            age=request.age,
            gender=request.gender,
            vital_signs=request.vital_signs
        )
        
        logger.info(f"Processing symptom-only analysis with {len(request.symptoms)} symptoms")
        
        # Run diagnosis
        result = await diagnostic_agent.diagnose(diagnostic_input)
        
        # Process sources to ensure they're strings
        processed_sources = process_sources(result.sources)
        
        # Convert result to response format
        response = DiagnosticResponse(
            primary_diagnosis=result.primary_diagnosis,
            confidence_score=result.confidence_score,
            differential_diagnoses=result.differential_diagnoses,
            severity=result.severity,
            findings=result.findings,
            recommendations=result.recommendations,
            follow_up_questions=result.follow_up_questions,
            first_aid_steps=result.first_aid_steps,
            medications=result.medications,
            emergency_indicators=result.emergency_indicators,
            follow_up=result.follow_up,
            sources=processed_sources  # Use processed sources
        )
        
        logger.info(f"Symptom analysis completed successfully. Primary diagnosis: {result.primary_diagnosis}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during symptom analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during analysis: {str(e)}"
        )

@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats"""
    return {
        "supported_formats": [
            {
                "extension": ".pdf",
                "description": "PDF documents",
                "mime_types": ["application/pdf"]
            },
            {
                "extension": ".txt",
                "description": "Text files",
                "mime_types": ["text/plain"]
            },
            {
                "extension": ".doc",
                "description": "Microsoft Word documents",
                "mime_types": ["application/msword"]
            },
            {
                "extension": ".docx",
                "description": "Microsoft Word documents (newer format)",
                "mime_types": ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
            },
            {
                "extension": ".jpg/.jpeg",
                "description": "JPEG images",
                "mime_types": ["image/jpeg"]
            },
            {
                "extension": ".png",
                "description": "PNG images",
                "mime_types": ["image/png"]
            }
        ],
        "max_file_size": "10MB",
        "max_files": 10
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred",
            "type": "internal_server_error"
        }
    )

if __name__ == "__main__":
    # For development - use uvicorn command in production
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )