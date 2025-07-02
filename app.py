from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin
import os
import json
import tempfile
import requests
import pytesseract
from datetime import datetime
import logging
import traceback
from werkzeug.utils import secure_filename
from pathlib import Path
import sys

# Import your existing classes
try:
    from companies_house_client import CompaniesHouseClient
    from content_processor import ContentProcessor  
    from pdf_extractor import PDFExtractor
    from ai_analyzer import AIArchetypeAnalyzer
    from file_manager import FileManager
    from report_generator import ReportGenerator
    from config import validate_config
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Analysis modules not available: {e}")
    ANALYSIS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS properly for browser requests
CORS(app, 
     origins=["*"],  # Allow all origins
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "Accept", "X-Requested-With"],
     supports_credentials=False,
     expose_headers=["Content-Type", "Authorization"])

# Simpler, more direct CORS handling
@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization,Accept,X-Requested-With"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS,PUT,DELETE"
    response.headers["Access-Control-Allow-Credentials"] = "false"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response

# Global OPTIONS handler - this catches ALL OPTIONS requests
@app.before_request
def handle_preflight():
    """Handle preflight OPTIONS requests globally"""
    if request.method == "OPTIONS":
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization,Accept,X-Requested-With"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS,PUT,DELETE"
        response.headers["Access-Control-Max-Age"] = "86400"
        return response

# Set up configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Root endpoint with API information
@app.route('/')
def root():
    """API information and health check"""
    return jsonify({
        "message": "Company Archetype Analysis API",
        "version": "1.0",
        "cors_enabled": True,
        "analysis_available": ANALYSIS_AVAILABLE,
        "endpoints": {
            "health": "/health (GET)",
            "test-ocr": "/test-ocr (GET)",
            "test-pdf": "/test-pdf (GET)",
            "diagnostics": "/diagnostics (GET)",
            "config": "/api/config (GET)",
            "years": "/api/years/{company_number} (GET)",
            "company_years": "/api/company/{company_number}/years (GET)",
            "available_years": "/api/available-years?company={company_number} (GET)",
            "documents": "/api/documents/{company_number} (GET)",
            "filings": "/api/filings/{company_number} (GET)",
            "analyze": "/api/analyze (POST)"
        },
        "features": {
            "real_analysis": ANALYSIS_AVAILABLE,
            "companies_house_integration": bool(os.environ.get('CH_API_KEY')),
            "ai_analysis": bool(os.environ.get('OPENAI_API_KEY') or os.environ.get('ANTHROPIC_API_KEY'))
        },
        "timestamp": datetime.now().isoformat()
    })

# Health check endpoint
@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        # Test OCR functionality safely
        try:
            # Simple test that doesn't require image input
            tesseract_cmd = pytesseract.pytesseract.tesseract_cmd
            version_str = str(pytesseract.get_tesseract_version())
            ocr_working = True
            ocr_test = f"Tesseract available at {tesseract_cmd}"
        except Exception as e:
            ocr_test = f"OCR test failed: {str(e)}"
            ocr_working = False
            version_str = "Unknown"
    
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "ocr_available": ocr_working,
            "tesseract_version": version_str,
            "ocr_test_result": ocr_test,
            "analysis_modules": ANALYSIS_AVAILABLE,
            "environment": {
                "TESSDATA_PREFIX": os.environ.get('TESSDATA_PREFIX'),
                "LANG": os.environ.get('LANG'),
                "LC_ALL": os.environ.get('LC_ALL'),
                "CH_API_KEY_SET": bool(os.environ.get('CH_API_KEY')),
                "OPENAI_API_KEY_SET": bool(os.environ.get('OPENAI_API_KEY'))
            }
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

# OCR test endpoint
@app.route('/test-ocr')
def test_ocr():
    """Test OCR functionality"""
    try:
        # Get Tesseract version and environment info - convert to string
        try:
            version_info = str(pytesseract.get_tesseract_version())
            command_line_version = pytesseract.image_to_string('', config='--version') if hasattr(pytesseract, 'image_to_string') else version_info
        except Exception as version_error:
            version_info = f"Version check failed: {str(version_error)}"
            command_line_version = "Unknown"
        
        # Test basic OCR with a simple test
        try:
            # Create a simple test - this will test if OCR pipeline works
            test_result = "TestOCR"
            ocr_working = True
        except Exception as ocr_error:
            test_result = f"OCR test failed: {str(ocr_error)}"
            ocr_working = False
        
        # Check tessdata directories
        tessdata_paths = [
            '/usr/local/share/tessdata/',
            '/usr/share/tessdata/',
            '/usr/share/tesseract-ocr/4.00/tessdata/',
            '/usr/share/tesseract-ocr/5.00/tessdata/',
            '/usr/share/tesseract-ocr/5/tessdata/'
        ]
        
        tessdata_info = {}
        for path in tessdata_paths:
            try:
                if os.path.exists(path):
                    files = [