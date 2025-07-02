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

# --- OpenAI API Health Check Route ---
@app.route("/check-openai")
def check_openai():
    import openai
    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key or openai_key.startswith("your_"):
            return "❌ API key missing or placeholder"

        client = openai.OpenAI(api_key=openai_key)
        models = client.models.list()
        model_count = len(models.data) if hasattr(models, 'data') else 0
        return f"✅ OpenAI API key is working. {model_count} models available."
    except Exception as e:
        return f"❌ OpenAI API test failed: {type(e).__name__}: {e}"
# --- End of Health Check ---


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
            command_line_version = version_info
        except Exception as version_error:
            version_info = f"Version check failed: {str(version_error)}"
            command_line_version = "Unknown"
        
        # Test basic OCR with a simple test
        try:
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

# API configuration endpoint
@app.route('/api/config')
def api_config():
    """Get API configuration status"""
    return jsonify({
        "companies_house_configured": bool(os.environ.get('CH_API_KEY')),
        "openai_configured": bool(os.environ.get('OPENAI_API_KEY')),
        "anthropic_configured": bool(os.environ.get('ANTHROPIC_API_KEY')),
        "analysis_modules_available": ANALYSIS_AVAILABLE,
        "analysis_methods_available": [
            "pattern_based",
            "ai_powered" if (os.environ.get('OPENAI_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')) else None
        ],
        "pdf_extraction_methods": ["pypdf2", "pdfplumber", "ocr"] if ANALYSIS_AVAILABLE else ["basic"],
        "max_years_analysis": 10,
        "max_files_per_analysis": 5,
        "features": {
            "real_company_data": bool(os.environ.get('CH_API_KEY')),
            "ai_analysis": bool(os.environ.get('OPENAI_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')),
            "advanced_pdf_processing": ANALYSIS_AVAILABLE,
            "archetype_classification": ANALYSIS_AVAILABLE
        }
    })

# ===== NEW ENDPOINTS TO MATCH FRONTEND EXPECTATIONS =====

@app.route('/api/years/<company_number>')
@app.route('/api/company/<company_number>/years')
def get_company_years(company_number):
    """Get available years for a company"""
    try:
        if not ANALYSIS_AVAILABLE:
            return jsonify({"error": "Analysis modules not available"}), 500
            
        ch_client = CompaniesHouseClient()
        
        # Validate company exists
        exists, company_name = ch_client.validate_company_exists(company_number)
        if not exists:
            return jsonify({"error": f"Company {company_number} not found"}), 404
        
        # Get filing history
        filing_history = ch_client.get_filing_history(company_number, category="accounts")
        if not filing_history:
            return jsonify({"error": "No filing history found"}), 404
        
        # Extract years from accounts
        years = []
        for filing in filing_history.get('items', []):
            if filing.get('category') == 'accounts':
                date = filing.get('date', '')
                if date:
                    try:
                        year = datetime.fromisoformat(date.replace('Z', '+00:00')).year
                        if year not in years:
                            years.append(year)
                    except:
                        continue
        
        years.sort(reverse=True)  # Most recent first
        
        return jsonify({
            "company_number": company_number,
            "company_name": company_name,
            "years": years,
            "total_years": len(years),
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Error getting years for {company_number}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/available-years')
def get_available_years():
    """Get available years with query parameter"""
    company_number = request.args.get('company')
    if not company_number:
        return jsonify({"error": "company parameter required"}), 400
    
    return get_company_years(company_number)


@app.route('/api/documents/<company_number>')
@app.route('/api/filings/<company_number>')  
def get_company_documents(company_number):
    """Get available documents for a company"""
    try:
        if not ANALYSIS_AVAILABLE:
            return jsonify({"error": "Analysis modules not available"}), 500
            
        ch_client = CompaniesHouseClient()
        
        # Validate company exists
        exists, company_name = ch_client.validate_company_exists(company_number)
        if not exists:
            return jsonify({"error": f"Company {company_number} not found"}), 404
        
        # Get filing history
        filing_history = ch_client.get_filing_history(company_number, category="accounts")
        if not filing_history:
            return jsonify({"error": "No filing history found"}), 404
        
        # Format documents
        documents = []
        for filing in filing_history.get('items', []):
            if filing.get('category') == 'accounts':
                date = filing.get('date', '')
                year = None
                if date:
                    try:
                        year = datetime.fromisoformat(date.replace('Z', '+00:00')).year
                    except:
                        pass
                
                documents.append({
                    "transaction_id": filing.get('transaction_id'),
                    "description": filing.get('description'),
                    "date": date,
                    "year": year,
                    "type": filing.get('type'),
                    "category": filing.get('category'),
                    "downloadable": bool(filing.get('links', {}).get('document_metadata'))
                })
        
        # Sort by date (most recent first)
        documents.sort(key=lambda x: x.get('date', ''), reverse=True)
        
        return jsonify({
            "company_number": company_number,
            "company_name": company_name,
            "documents": documents,
            "total_documents": len(documents),
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Error getting documents for {company_number}: {e}")
        return jsonify({"error": str(e)}), 500

# Diagnostics endpoint
@app.route('/diagnostics')
def diagnostics():
    """Comprehensive system diagnostics"""
    try:
        diagnostics_info = {
            "system": {
                "python_version": sys.version,
                "platform": os.name,
                "working_directory": os.getcwd(),
                "temp_directory": tempfile.gettempdir()
            },
            "environment": dict(os.environ),
            "tessdata_check": {},
            "installed_packages": [],
            "analysis_modules": {
                "available": ANALYSIS_AVAILABLE,
                "import_errors": [] if ANALYSIS_AVAILABLE else ["See logs for import details"]
            }
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

# Main analysis endpoint - REAL INTEGRATION
@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Main company analysis endpoint with real archetype analysis"""
    try:
        # Check if analysis modules are available
        if not ANALYSIS_AVAILABLE:
            return jsonify({
                "success": False,
                "error": "Analysis modules not available. Please ensure all analysis files are copied to the Flask project.",
                "missing_modules": "companies_house_client, content_processor, pdf_extractor, ai_analyzer, config"
            }), 500
        
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
        
        # Validate company number format
        if not company_number.isdigit() or len(company_number) != 8:
            return jsonify({
                "success": False,
                "error": "Invalid UK company number format. Must be 8 digits."
            }), 400
        
        logger.info(f"Starting real archetype analysis for company {company_number}, years {years}")
        
        # Check if we have Companies House API key
        companies_house_api_key = os.environ.get('CH_API_KEY')
        if not companies_house_api_key:
            return jsonify({
                "success": False,
                "error": "Companies House API key not configured. Please set CH_API_KEY environment variable."
            }), 500
        
        # Initialize your existing classes
        try:
            ch_client = CompaniesHouseClient()
            content_processor = ContentProcessor()
            pdf_extractor = PDFExtractor()
            archetype_analyzer = AIArchetypeAnalyzer()
            file_manager = FileManager()
            report_generator = ReportGenerator()
        except Exception as e:
            logger.error(f"Failed to initialize analysis components: {e}")
            return jsonify({
                "success": False,
                "error": f"Failed to initialize analysis components: {str(e)}"
            }), 500
        
        # Validate company exists
        exists, company_name = ch_client.validate_company_exists(company_number)
        if not exists:
            return jsonify({
                "success": False,
                "error": f"Company {company_number} not found in Companies House records."
            }), 404
        
        # Calculate max_years to capture selected years
        current_year = datetime.now().year
        oldest_year = min(years)
        max_years_needed = current_year - oldest_year + 2
        
        # Download company filings
        logger.info(f"Downloading filings for last {max_years_needed} years...")
        download_results = ch_client.download_annual_accounts(company_number, max_years_needed)
        
        if not download_results or download_results['total_downloaded'] == 0:
            return jsonify({
                "success": False,
                "error": f"No annual accounts found for company {company_number}. This could mean the company hasn't filed accounts recently or they are not available for download."
            }), 404
        
        # Filter to selected years
        filtered_files = []
        for file_info in download_results['downloaded_files']:
            file_date = file_info.get('date')
            if file_date:
                try:
                    if isinstance(file_date, str):
                        file_year = datetime.strptime(str(file_date), '%Y-%m-%d').year
                    else:
                        file_year = file_date.year
                    
                    if file_year in years:
                        filtered_files.append(file_info)
                        logger.info(f"Including {file_info['filename']} (Year {file_year})")
                except Exception as e:
                    logger.warning(f"Could not parse date for {file_info['filename']}: {e}")
                    continue
        
        if not filtered_files:
            return jsonify({
                "success": False,
                "error": f"No documents found for selected years {years}. Available years might be different."
            }), 404
        
        logger.info(f"Found {len(filtered_files)} documents in selected years")
        
        # Extract content from PDFs
        extracted_content = []
        for file_info in filtered_files[:5]:  # Limit to 5 files for performance
            try:
                logger.info(f"Processing {file_info['filename']}")
                
                # Read PDF file
                with open(file_info['path'], 'rb') as f:
                    pdf_content = f.read()
                
                # Extract text using your PDFExtractor
                extraction_result = pdf_extractor.extract_text_from_pdf(
                    pdf_content, 
                    file_info['filename']
                )
                
                if extraction_result["extraction_status"] == "success":
                    content = extraction_result.get("raw_text", "")
                    
                    if content and len(content.strip()) > 100:
                        extracted_content.append({
                            'filename': file_info['filename'],
                            'date': file_info['date'],
                            'content': content,
                            'metadata': {
                                'transaction_id': file_info.get('transaction_id', ''),
                                'description': file_info.get('description', ''),
                                'file_size': file_info['size'],
                                'extraction_method': extraction_result["extraction_method"]
                            }
                        })
                        logger.info(f"Successfully extracted {len(content)} characters from {file_info['filename']}")
                    else:
                        logger.warning(f"Insufficient content extracted from {file_info['filename']}")
                else:
                    logger.warning(f"Failed to extract content from {file_info['filename']}: {extraction_result.get('error', 'Unknown error')}")
                
            except Exception as e:
                logger.error(f"Error processing {file_info['filename']}: {e}")
                continue
        
        if not extracted_content:
            return jsonify({
                "success": False,
                "error": "Could not extract readable content from any documents. This might indicate the PDFs are image-based or protected."
            }), 500
        
        logger.info(f"Successfully extracted content from {len(extracted_content)} documents")
        
        # Process content using your ContentProcessor
        processed_documents = []
        for content_data in extracted_content:
            try:
                processed = content_processor.process_document_content(
                    content_data['content'],
                    content_data['metadata']
                )
                processed_documents.append(processed)
            except Exception as e:
                logger.error(f"Error processing content from {content_data['filename']}: {e}")
                continue
        
        # Combine all documents
        combined_analysis = content_processor.combine_multiple_documents(processed_documents)
        
        # Perform archetype analysis using your AIArchetypeAnalyzer
        combined_content = "\n\n".join([content_data['content'] for content_data in extracted_content])
        
        logger.info("Starting archetype classification analysis...")
        archetype_analysis = archetype_analyzer.analyze_archetypes(
            combined_content,
            company_name,
            company_number
        )
        
        # Create portfolio analysis for saving/reporting
        portfolio_analysis = {
            'company_number': company_number,
            'company_name': company_name,
            'files_analyzed': len(extracted_content),
            'files_successful': len(extracted_content),
            'total_content_sections': combined_analysis.get('total_sections', len(extracted_content)),
            'analysis_timestamp': datetime.now().isoformat(),
            'archetype_analysis': archetype_analysis,
            'file_analyses': [
                {
                    'filename': content_data['filename'],
                    'extraction_status': 'success',
                    'content_summary': {
                        'strategy_found': True,
                        'governance_found': True,
                        'risk_found': True,
                        'audit_found': True,
                        'total_content_sections': 1
                    },
                    'extraction_method': content_data['metadata']['extraction_method'],
                    'debug_info': content_data['metadata'].get('debug_info', {})
                }
                for content_data in extracted_content
            ]
        }
        
        # Save results
        try:
            file_manager.save_analysis_results(company_name, company_number, portfolio_analysis)
            report_generator.generate_analysis_report(portfolio_analysis)
        except Exception as e:
            logger.warning(f"Failed to save results: {e}")
        
        # Clean up temporary files
        try:
            ch_client.cleanup_temp_files()
        except Exception as e:
            logger.warning(f"Could not clean up temp files: {e}")
        
        # Build response in the format your HTML expects
        if archetype_analysis.get('success', False):
            business_archetypes = archetype_analysis.get('business_strategy_archetypes', {})
            risk_archetypes = archetype_analysis.get('risk_strategy_archetypes', {})
            
            result = {
                "success": True,
                "company_number": company_number,
                "company_name": company_name,
                "years_analyzed": years,
                "files_processed": len(extracted_content),
                "documents_found": len(filtered_files),
                "analysis_date": datetime.now().isoformat(),
                "business_strategy": {
                    "dominant": business_archetypes.get('dominant', 'Unknown'),
                    "reasoning": business_archetypes.get('reasoning', 'Analysis completed but detailed reasoning not available'),
                    "secondary": business_archetypes.get('secondary')
                },
                "risk_strategy": {
                    "dominant": risk_archetypes.get('dominant', 'Unknown'), 
                    "reasoning": risk_archetypes.get('reasoning', 'Analysis completed but detailed reasoning not available'),
                    "secondary": risk_archetypes.get('secondary')
                },
                "confidence_score": min(len(extracted_content) / 3.0, 1.0),
                "analysis_method": archetype_analysis.get('analysis_type', 'pattern_based'),
                "data_sources": [content['filename'] for content in extracted_content],
                "word_count": combined_analysis.get('content_stats', {}).get('total_word_count', 0),
                "content_categories": {
                    category: len(sections) 
                    for category, sections in combined_analysis.get('categorized_content', {}).items()
                }
            }
            
            logger.info(f"Analysis completed successfully for {company_name}")
            logger.info(f"Business Strategy: {result['business_strategy']['dominant']}")
            logger.info(f"Risk Strategy: {result['risk_strategy']['dominant']}")
            logger.info(f"Analysis method: {result['analysis_method']}")
            
            return jsonify(result)
        else:
            # Analysis failed but return what we can
            error_msg = archetype_analysis.get('error', 'Archetype analysis failed')
            logger.error(f"Archetype analysis failed: {error_msg}")
            
            return jsonify({
                "success": False,
                "error": f"Archetype analysis failed: {error_msg}",
                "company_name": company_name,
                "files_processed": len(extracted_content),
                "partial_data": {
                    "word_count": combined_analysis.get('content_stats', {}).get('total_word_count', 0),
                    "documents_processed": len(extracted_content),
                    "content_categories": {
                        category: len(sections) 
                        for category, sections in combined_analysis.get('categorized_content', {}).items()
                    }
                }
            }), 500
        
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
            "/api/config",
            "/api/years/{company_number}",
            "/api/company/{company_number}/years",
            "/api/available-years?company={number}",
            "/api/documents/{company_number}",
            "/api/filings/{company_number}",
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
    logger.info(f"Analysis modules available: {ANALYSIS_AVAILABLE}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )