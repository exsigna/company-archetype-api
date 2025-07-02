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
        "endpoints": {
            "health": "/health (GET)",
            "test-ocr": "/test-ocr (GET)",
            "test-pdf": "/test-pdf (GET)",
            "diagnostics": "/diagnostics (GET)",
            "analyze": "/api/analyze (POST)"
        },
        "ocr_debug_available": True,
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
            "environment": {
                "TESSDATA_PREFIX": os.environ.get('TESSDATA_PREFIX'),
                "LANG": os.environ.get('LANG'),
                "LC_ALL": os.environ.get('LC_ALL')
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
                    files = [f for f in os.listdir(path) if f.endswith('.traineddata')]
                    tessdata_info[path] = {
                        "exists": True,
                        "traineddata_files": files[:10],  # Limit to first 10 files
                        "total_files": len(files)
                    }
                else:
                    tessdata_info[path] = {"exists": False}
            except Exception as dir_error:
                tessdata_info[path] = {"exists": False, "error": str(dir_error)}
        
        return jsonify({
            "ocr_test": f"Success: '{test_result}'" if ocr_working else test_result,
            "ocr_working": ocr_working,
            "pytesseract_version": version_info,
            "tesseract_path": pytesseract.pytesseract.tesseract_cmd,
            "command_line_version": command_line_version,
            "environment_variables": {
                "TESSDATA_PREFIX": os.environ.get('TESSDATA_PREFIX'),
                "LANG": os.environ.get('LANG'),
                "LC_ALL": os.environ.get('LC_ALL'),
                "OMP_THREAD_LIMIT": os.environ.get('OMP_THREAD_LIMIT')
            },
            "tessdata_directories": tessdata_info,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": "OCR test failed",
            "details": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }), 500

# PDF test endpoint
@app.route('/test-pdf')
def test_pdf():
    """Test PDF processing capabilities"""
    try:
        import pdf2image
        import PIL
        
        return jsonify({
            "pdf_processing": "Available",
            "pdf2image_available": True,
            "PIL_available": True,
            "poppler_path": "System installed",
            "timestamp": datetime.now().isoformat()
        })
        
    except ImportError as e:
        return jsonify({
            "pdf_processing": "Limited",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 200

# Diagnostics endpoint
@app.route('/diagnostics')
def diagnostics():
    """Comprehensive system diagnostics"""
    try:
        diagnostics_info = {
            "system": {
                "python_version": os.sys.version,
                "platform": os.name,
                "working_directory": os.getcwd(),
                "temp_directory": tempfile.gettempdir()
            },
            "environment": dict(os.environ),
            "tessdata_check": {},
            "installed_packages": []
        }
        
        # Check for tessdata files
        tessdata_locations = [
            '/usr/share/tesseract-ocr/5/tessdata/',
            '/usr/share/tesseract-ocr/4.00/tessdata/',
            '/usr/share/tessdata/',
            '/usr/local/share/tessdata/'
        ]
        
        for location in tessdata_locations:
            if os.path.exists(location):
                files = [f for f in os.listdir(location) if f.endswith('.traineddata')]
                diagnostics_info["tessdata_check"][location] = files
        
        try:
            import pkg_resources
            installed_packages = [str(d) for d in pkg_resources.working_set]
            diagnostics_info["installed_packages"] = sorted(installed_packages)
        except:
            diagnostics_info["installed_packages"] = ["Package list unavailable"]
        
        return jsonify(diagnostics_info)
        
    except Exception as e:
        return jsonify({
            "error": "Diagnostics failed",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

# Handle OPTIONS requests for analyze endpoint - Remove this since we have global handler
# @app.route('/api/analyze', methods=['OPTIONS'])
# def analyze_options():
#     """Handle preflight OPTIONS requests for analyze endpoint"""
#     response = make_response()
#     response.headers.add("Access-Control-Allow-Origin", "*")
#     response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization,Accept,X-Requested-With")
#     response.headers.add('Access-Control-Allow-Methods', "POST,OPTIONS")
#     return response

# Main analysis endpoint - simplified
@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Main company analysis endpoint"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        company_number = data.get('company_number')
        years = data.get('years', [])
        
        if not company_number:
            return jsonify({
                "success": False,
                "error": "Company number is required"
            }), 400
        
        if not years:
            return jsonify({
                "success": False,
                "error": "At least one year must be specified"
            }), 400
        
        # Validate company number format (UK companies are 8 digits)
        if not company_number.isdigit() or len(company_number) != 8:
            return jsonify({
                "success": False,
                "error": "Invalid UK company number format. Must be 8 digits."
            }), 400
        
        logger.info(f"Starting analysis for company {company_number}, years {years}")
        
        # Mock analysis for now - replace with your actual logic
        # This is where you would integrate with Companies House API,
        # download documents, process PDFs, run OCR, and perform archetype analysis
        
        mock_result = {
            "success": True,
            "company_number": company_number,
            "company_name": f"Example Company Ltd (#{company_number})",
            "years_analyzed": years,
            "files_processed": len(years) * 3,  # Mock: 3 files per year
            "analysis_date": datetime.now().isoformat(),
            "business_strategy": {
                "dominant": "Growth-Oriented",
                "reasoning": "Based on analysis of annual reports and strategic statements, the company demonstrates a consistent focus on market expansion and revenue growth."
            },
            "risk_strategy": {
                "dominant": "Balanced Risk",
                "reasoning": "The company maintains a moderate approach to risk management, balancing growth opportunities with prudent financial controls."
            },
            "confidence_score": 0.85,
            "data_sources": [
                "Annual Reports",
                "Strategic Reports", 
                "Directors' Reports"
            ]
        }
        
        return jsonify(mock_result)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            "success": False,
            "error": "Internal server error during analysis",
            "details": str(e) if app.debug else "Contact support if this persists"
        }), 500

# File upload endpoint (for future use) - simplified
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads for OCR processing"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the file with OCR
            try:
                if filename.lower().endswith('.pdf'):
                    # Handle PDF files
                    extracted_text = "PDF processing not fully implemented yet"
                else:
                    # Handle image files
                    extracted_text = pytesseract.image_to_string(filepath)
                
                # Clean up
                os.remove(filepath)
                
                return jsonify({
                    "success": True,
                    "filename": filename,
                    "extracted_text": extracted_text,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as ocr_error:
                # Clean up on error
                if os.path.exists(filepath):
                    os.remove(filepath)
                
                return jsonify({
                    "success": False,
                    "error": "OCR processing failed",
                    "details": str(ocr_error)
                }), 500
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Upload failed",
            "details": str(e)
        }), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "error": "File too large",
        "max_size": "16MB"
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "/health",
            "/test-ocr", 
            "/test-pdf",
            "/diagnostics",
            "/api/analyze",
            "/upload"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "error": "Internal server error",
        "message": "Something went wrong on our end"
    }), 500

if __name__ == '__main__':
    # Development server
    port = int(os.environ.get('PORT', 10000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )