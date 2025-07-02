#!/usr/bin/env python3
"""
Flask API for Exsigna Integration - Company Archetype Analysis
Deployment version for Render.com
"""

from flask import Flask, request, jsonify, make_response
import logging
import os
import tempfile
import subprocess
import sys
import shutil
from datetime import datetime
from pathlib import Path

# Import your existing modules
from companies_house_client import CompaniesHouseClient
from content_processor import ContentProcessor
from pdf_extractor import PDFExtractor
from ai_analyzer import AIArchetypeAnalyzer
from file_manager import FileManager
from report_generator import ReportGenerator
from config import validate_config, MIN_EXTRACTION_LENGTH, validate_company_number

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure Flask for production
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# CORS configuration - handle manually for better control
@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept, Authorization, X-Requested-With'
    response.headers['Access-Control-Max-Age'] = '86400'
    return response

@app.route('/', methods=['GET'])
def home():
    """Basic home endpoint with API information"""
    return jsonify({
        'service': 'Company Archetype Analysis API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'analyze': '/api/analyze (POST)',
            'health': '/health (GET)',
            'test_ocr': '/test-ocr (GET)'
        },
        'usage': 'Send POST to /api/analyze with {"company_number": "12345678", "years": [2020, 2021, 2022]}'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Basic configuration check
        config_valid = validate_config()
        return jsonify({
            'status': 'healthy' if config_valid else 'configuration_error',
            'timestamp': datetime.now().isoformat(),
            'config_valid': config_valid
        }), 200 if config_valid else 503
    except Exception as e:
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 500

@app.route('/test-ocr', methods=['GET'])
def test_ocr():
    """Test OCR functionality on Render"""
    diagnostics = {}
    
    try:
        # Test 1: Check if tesseract binary is available
        result = subprocess.run(['tesseract', '--version'], 
                               capture_output=True, text=True, timeout=10)
        diagnostics['tesseract_binary'] = {
            'available': result.returncode == 0,
            'version': result.stdout if result.returncode == 0 else result.stderr,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except FileNotFoundError:
        diagnostics['tesseract_binary'] = {
            'available': False,
            'error': 'tesseract command not found',
            'return_code': -1
        }
    except Exception as e:
        diagnostics['tesseract_binary'] = {'error': str(e), 'available': False}
    
    try:
        # Test 2: Check pytesseract import and config
        import pytesseract
        diagnostics['pytesseract'] = {
            'import_success': True,
            'version': pytesseract.__version__ if hasattr(pytesseract, '__version__') else 'unknown',
            'module_path': pytesseract.__file__
        }
        
        # Test pytesseract.get_tesseract_version()
        try:
            tess_version = pytesseract.get_tesseract_version()
            diagnostics['pytesseract']['tesseract_version'] = str(tess_version)
        except Exception as e:
            diagnostics['pytesseract']['tesseract_version_error'] = str(e)
            
        # Test creating a simple image and OCR
        try:
            from PIL import Image
            import io
            
            # Create a simple test image with text
            img = Image.new('RGB', (200, 50), color='white')
            
            # Try OCR on test image
            test_text = pytesseract.image_to_string(img, timeout=10)
            diagnostics['pytesseract']['simple_ocr_test'] = {
                'success': True,
                'result': test_text or "(no text detected)",
                'result_length': len(test_text) if test_text else 0
            }
        except Exception as e:
            diagnostics['pytesseract']['simple_ocr_test'] = {
                'success': False,
                'error': str(e)
            }
            
    except ImportError as e:
        diagnostics['pytesseract'] = {'import_error': str(e), 'import_success': False}
    except Exception as e:
        diagnostics['pytesseract'] = {'other_error': str(e), 'import_success': False}
    
    try:
        # Test 3: Check pdf2image and poppler
        from pdf2image import convert_from_bytes
        diagnostics['pdf2image'] = {
            'import_success': True,
            'module_path': convert_from_bytes.__module__
        }
        
        # Test poppler utilities
        poppler_commands = ['pdftoppm', 'pdfinfo', 'pdftocairo']
        poppler_results = {}
        
        for cmd in poppler_commands:
            try:
                result = subprocess.run([cmd, '-h'], 
                                       capture_output=True, text=True, timeout=5)
                poppler_results[cmd] = {
                    'available': result.returncode in [0, 1],  # Some commands return 1 for help
                    'return_code': result.returncode
                }
            except FileNotFoundError:
                poppler_results[cmd] = {
                    'available': False,
                    'error': f'{cmd} command not found'
                }
            except Exception as e:
                poppler_results[cmd] = {
                    'available': False,
                    'error': str(e)
                }
        
        diagnostics['poppler_utils'] = poppler_results
        
    except ImportError as e:
        diagnostics['pdf2image'] = {'import_error': str(e), 'import_success': False}
    except Exception as e:
        diagnostics['pdf2image'] = {'other_error': str(e), 'import_success': False}
    
    # Test 4: Environment info
    diagnostics['environment'] = {
        'python_version': sys.version,
        'tessdata_prefix': os.environ.get('TESSDATA_PREFIX', 'not set'),
        'path_dirs': os.environ.get('PATH', '').split(':'),
        'ld_library_path': os.environ.get('LD_LIBRARY_PATH', 'not set'),
        'render_service': os.environ.get('RENDER_SERVICE_NAME', 'not set'),
        'render_env': os.environ.get('RENDER', 'not set')
    }
    
    # Test 5: Check executable locations
    executables = ['tesseract', 'pdftoppm', 'python3']
    executable_paths = {}
    
    for exe in executables:
        try:
            path = shutil.which(exe)
            executable_paths[exe] = path or 'not found in PATH'
        except Exception as e:
            executable_paths[exe] = f'error: {e}'
    
    diagnostics['executable_paths'] = executable_paths
    
    # Test 6: System package check
    try:
        # Check if apt packages are installed
        result = subprocess.run(['dpkg', '-l'], 
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            installed_packages = result.stdout
            ocr_packages = {}
            for pkg in ['tesseract-ocr', 'poppler-utils', 'libtesseract-dev']:
                ocr_packages[pkg] = pkg in installed_packages
            diagnostics['installed_packages'] = ocr_packages
        else:
            diagnostics['installed_packages'] = {'error': 'dpkg command failed'}
    except Exception as e:
        diagnostics['installed_packages'] = {'error': str(e)}
    
    # Test 7: Memory usage
    try:
        import psutil
        memory = psutil.virtual_memory()
        diagnostics['memory'] = {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'percent_used': memory.percent
        }
    except ImportError:
        diagnostics['memory'] = {'error': 'psutil not available'}
    except Exception as e:
        diagnostics['memory'] = {'error': str(e)}
    
    return jsonify(diagnostics)

# Handle OPTIONS requests globally for all /api/* routes
@app.route('/api/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    """Handle CORS preflight requests for all API endpoints"""
    response = make_response('', 200)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept, Authorization, X-Requested-With'
    response.headers['Access-Control-Max-Age'] = '86400'
    return response

@app.route('/api/analyze', methods=['POST'])
def analyze_company():
    """
    Main API endpoint for Exsigna website integration
    
    Expected payload:
    {
        "company_number": "02613335",
        "years": [2020, 2021, 2022]
    }
    
    Returns:
    {
        "success": true,
        "company_name": "COMPANY NAME LIMITED",
        "business_strategy": {...},
        "risk_strategy": {...}
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract and validate parameters
        company_number = data.get('company_number', '').strip()
        selected_years = data.get('years', [])
        
        if not company_number:
            return jsonify({'error': 'company_number is required'}), 400
        
        if not validate_company_number(company_number):
            return jsonify({'error': 'Invalid UK company number format'}), 400
        
        if not selected_years or not isinstance(selected_years, list):
            return jsonify({'error': 'years must be a non-empty array'}), 400
        
        # Validate years
        current_year = datetime.now().year
        for year in selected_years:
            if not isinstance(year, int) or year < 1990 or year > current_year:
                return jsonify({'error': f'Invalid year: {year}. Must be between 1990 and {current_year}'}), 400
        
        logger.info(f"Starting analysis for company {company_number}, years: {selected_years}")
        
        # Validate configuration
        if not validate_config():
            logger.error("Configuration validation failed")
            return jsonify({'error': 'Server configuration invalid'}), 500
        
        # Initialize components
        try:
            ch_client = CompaniesHouseClient()
            content_processor = ContentProcessor()
            pdf_extractor = PDFExtractor()
            archetype_analyzer = AIArchetypeAnalyzer()
            file_manager = FileManager()
            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            return jsonify({'error': 'Failed to initialize analysis components'}), 500
        
        # Validate company exists
        try:
            exists, company_name = ch_client.validate_company_exists(company_number)
            if not exists:
                logger.warning(f"Company {company_number} not found")
                return jsonify({'error': f'Company {company_number} not found in Companies House records'}), 404
            
            logger.info(f"Company validated: {company_name}")
        except Exception as e:
            logger.error(f"Company validation failed: {e}")
            return jsonify({'error': 'Failed to validate company'}), 500
        
        # Download and filter files for selected years
        try:
            # Calculate how many years back we need to look
            oldest_year = min(selected_years)
            max_years_needed = current_year - oldest_year + 2  # Add buffer
            
            logger.info(f"Downloading accounts (scanning back {max_years_needed} years)")
            download_results = ch_client.download_annual_accounts(company_number, max_years_needed)
            
            if not download_results or download_results['total_downloaded'] == 0:
                logger.warning("No annual accounts available for download")
                return jsonify({'error': 'No annual accounts available for download'}), 404
            
            logger.info(f"Downloaded {download_results['total_downloaded']} documents")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return jsonify({'error': 'Failed to download company filings'}), 500
        
        # Filter to selected years
        try:
            filtered_files = []
            for file_info in download_results['downloaded_files']:
                file_date = file_info.get('date', '')
                if file_date:
                    try:
                        # Handle different date formats
                        if isinstance(file_date, str):
                            if 'T' in file_date:
                                file_year = datetime.fromisoformat(file_date.replace('Z', '+00:00')).year
                            else:
                                file_year = datetime.strptime(file_date, '%Y-%m-%d').year
                        else:
                            file_year = file_date.year
                        
                        if file_year in selected_years:
                            filtered_files.append(file_info)
                            logger.info(f"Included file for year {file_year}: {file_info['filename']}")
                    except Exception as date_e:
                        logger.warning(f"Could not parse date for {file_info.get('filename', 'unknown')}: {date_e}")
                        continue
            
            if not filtered_files:
                logger.warning(f"No files found for selected years: {selected_years}")
                return jsonify({'error': f'No files found for selected years: {selected_years}'}), 404
            
            logger.info(f"Filtered to {len(filtered_files)} files matching selected years")
        except Exception as e:
            logger.error(f"File filtering failed: {e}")
            return jsonify({'error': 'Failed to filter files by year'}), 500
        
        # Extract content from PDFs
        try:
            extracted_content = []
            for i, file_info in enumerate(filtered_files, 1):
                logger.info(f"Extracting content from file {i}/{len(filtered_files)}: {file_info['filename']}")
                
                try:
                    with open(file_info['path'], 'rb') as f:
                        pdf_content = f.read()
                    
                    extraction_result = pdf_extractor.extract_text_from_pdf(
                        pdf_content, file_info['filename']
                    )
                    
                    if extraction_result["extraction_status"] == "success":
                        content = extraction_result.get("raw_text", "")
                        if content and len(content.strip()) > MIN_EXTRACTION_LENGTH:
                            extracted_content.append({
                                'filename': file_info['filename'],
                                'date': file_info['date'],
                                'content': content,
                                'metadata': {
                                    'transaction_id': file_info.get('transaction_id', ''),
                                    'file_size': file_info['size'],
                                    'extraction_method': extraction_result["extraction_method"]
                                }
                            })
                            logger.info(f"Successfully extracted {len(content)} characters using {extraction_result['extraction_method']}")
                        else:
                            logger.warning(f"Insufficient content extracted from {file_info['filename']}")
                    else:
                        logger.warning(f"Extraction failed for {file_info['filename']}: {extraction_result.get('error', 'Unknown error')}")
                        
                except Exception as extract_e:
                    logger.error(f"Error processing {file_info['filename']}: {extract_e}")
                    continue
            
            if not extracted_content:
                logger.error("No readable content extracted from any PDFs")
                return jsonify({'error': 'No readable content could be extracted from the PDF files'}), 422
            
            logger.info(f"Successfully extracted content from {len(extracted_content)} files")
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            return jsonify({'error': 'Failed to extract content from PDFs'}), 500
        
        # Process content
        try:
            processed_documents = []
            for content_data in extracted_content:
                processed = content_processor.process_document_content(
                    content_data['content'], content_data['metadata']
                )
                processed_documents.append(processed)
            
            combined_analysis = content_processor.combine_multiple_documents(processed_documents)
            total_words = sum(len(content_data['content'].split()) for content_data in extracted_content)
            logger.info(f"Content processing completed. Total words: {total_words:,}")
        except Exception as e:
            logger.error(f"Content processing failed: {e}")
            return jsonify({'error': 'Failed to process document content'}), 500
        
        # Perform archetype analysis
        try:
            combined_content = "\n\n".join([content_data['content'] for content_data in extracted_content])
            
            logger.info("Starting archetype analysis...")
            archetype_analysis = archetype_analyzer.analyze_archetypes(
                combined_content, company_name, company_number
            )
            
            if not archetype_analysis.get('success', False):
                error_msg = archetype_analysis.get('error', 'Archetype analysis failed')
                logger.error(f"Archetype analysis failed: {error_msg}")
                return jsonify({'error': f'Archetype analysis failed: {error_msg}'}), 500
            
            logger.info(f"Archetype analysis completed using {archetype_analysis.get('analysis_type', 'unknown')} method")
        except Exception as e:
            logger.error(f"Archetype analysis error: {e}")
            return jsonify({'error': 'Failed to perform archetype analysis'}), 500
        
        # Clean up temporary files
        try:
            ch_client.cleanup_temp_files()
            logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Could not clean up temp files: {e}")
        
        # Prepare response
        business_archetypes = archetype_analysis.get('business_strategy_archetypes', {})
        risk_archetypes = archetype_analysis.get('risk_strategy_archetypes', {})
        
        response_data = {
            'success': True,
            'company_number': company_number,
            'company_name': company_name,
            'analysis_date': datetime.now().isoformat(),
            'years_analyzed': sorted(selected_years),
            'files_processed': len(extracted_content),
            'total_words_analyzed': total_words,
            'business_strategy': {
                'dominant': business_archetypes.get('dominant', ''),
                'secondary': business_archetypes.get('secondary', ''),
                'reasoning': business_archetypes.get('reasoning', '')
            },
            'risk_strategy': {
                'dominant': risk_archetypes.get('dominant', ''),
                'secondary': risk_archetypes.get('secondary', ''),
                'reasoning': risk_archetypes.get('reasoning', '')
            },
            'analysis_method': archetype_analysis.get('analysis_type', ''),
            'model_used': archetype_analysis.get('model_used', 'pattern_based')
        }
        
        logger.info(f"Analysis completed successfully for {company_name}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Unexpected error in analysis: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # For local development
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)