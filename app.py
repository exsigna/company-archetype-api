#!/usr/bin/env python3
"""
Flask API for Exsigna Integration - Company Archetype Analysis
Deployment version for Render.com with CORS support and enhanced OCR debugging
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import tempfile
import subprocess
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

# Enable CORS for all domains and all routes
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

# Configure Flask for production
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

@app.route('/', methods=['GET'])
def home():
    """Basic home endpoint with API information"""
    return jsonify({
        'service': 'Company Archetype Analysis API',
        'version': '1.0.1',
        'status': 'running',
        'cors_enabled': True,
        'ocr_debug_available': True,
        'endpoints': {
            'analyze': '/api/analyze (POST)',
            'health': '/health (GET)',
            'diagnostics': '/diagnostics (GET)',
            'test-ocr': '/test-ocr (GET)',
            'test-pdf': '/test-pdf (GET)'
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
            'config_valid': config_valid,
            'cors_enabled': True,
            'ocr_available': check_ocr_availability()
        }), 200 if config_valid else 503
    except Exception as e:
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'cors_enabled': True
        }), 500

def check_ocr_availability():
    """Quick check if OCR is available"""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except:
        return False

@app.route('/test-ocr', methods=['GET'])
def test_ocr():
    """Test OCR configuration and capabilities"""
    try:
        import pytesseract
        from pdf2image import convert_from_path
        import subprocess
        import os
        
        # Test Tesseract installation
        try:
            version = pytesseract.get_tesseract_version()
            tesseract_path = pytesseract.pytesseract.tesseract_cmd
        except Exception as e:
            version = f"Error: {e}"
            tesseract_path = "Not found"
        
        # Test command line tesseract
        try:
            result = subprocess.run(['tesseract', '--version'], 
                                 capture_output=True, text=True, timeout=10)
            cmd_version = result.stdout + result.stderr
        except Exception as e:
            cmd_version = f"Command failed: {e}"
        
        # Check tessdata directory
        tessdata_paths = [
            '/usr/share/tesseract-ocr/5.00/tessdata/',
            '/usr/share/tesseract-ocr/4.00/tessdata/',
            '/usr/share/tessdata/',
            '/usr/local/share/tessdata/'
        ]
        
        tessdata_info = {}
        for path in tessdata_paths:
            if os.path.exists(path):
                try:
                    files = os.listdir(path)
                    tessdata_info[path] = {
                        'exists': True,
                        'file_count': len(files),
                        'eng_data': 'eng.traineddata' in files,
                        'files': files[:10]  # First 10 files
                    }
                except Exception as e:
                    tessdata_info[path] = {'exists': True, 'error': str(e)}
            else:
                tessdata_info[path] = {'exists': False}
        
        # Environment variables
        env_vars = {
            'TESSDATA_PREFIX': os.environ.get('TESSDATA_PREFIX', 'Not set'),
            'OMP_THREAD_LIMIT': os.environ.get('OMP_THREAD_LIMIT', 'Not set'),
            'LC_ALL': os.environ.get('LC_ALL', 'Not set'),
            'LANG': os.environ.get('LANG', 'Not set')
        }
        
        # Test simple OCR
        ocr_test_result = "Not tested"
        try:
            # Create a simple test image with text
            from PIL import Image, ImageDraw, ImageFont
            import io
            
            # Create a simple image with text
            img = Image.new('RGB', (200, 50), color='white')
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "Test OCR", fill='black')
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Test OCR
            ocr_result = pytesseract.image_to_string(img)
            ocr_test_result = f"Success: '{ocr_result.strip()}'"
            
        except Exception as e:
            ocr_test_result = f"Failed: {str(e)}"
        
        return jsonify({
            'pytesseract_version': str(version),
            'tesseract_path': tesseract_path,
            'command_line_version': cmd_version,
            'tessdata_directories': tessdata_info,
            'environment_variables': env_vars,
            'ocr_test': ocr_test_result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/test-pdf', methods=['GET'])
def test_pdf_processing():
    """Test PDF processing capabilities"""
    try:
        # Test PDF libraries
        pdf_status = {}
        
        # Test PyPDF2
        try:
            import PyPDF2
            pdf_status['PyPDF2'] = {
                'available': True,
                'version': PyPDF2.__version__ if hasattr(PyPDF2, '__version__') else 'Unknown'
            }
        except ImportError as e:
            pdf_status['PyPDF2'] = {'available': False, 'error': str(e)}
        
        # Test pdfplumber
        try:
            import pdfplumber
            pdf_status['pdfplumber'] = {
                'available': True,
                'version': pdfplumber.__version__ if hasattr(pdfplumber, '__version__') else 'Unknown'
            }
        except ImportError as e:
            pdf_status['pdfplumber'] = {'available': False, 'error': str(e)}
        
        # Test pdf2image
        try:
            import pdf2image
            pdf_status['pdf2image'] = {
                'available': True,
                'version': pdf2image.__version__ if hasattr(pdf2image, '__version__') else 'Unknown'
            }
        except ImportError as e:
            pdf_status['pdf2image'] = {'available': False, 'error': str(e)}
        
        # Test poppler utilities
        poppler_status = {}
        poppler_commands = ['pdftoppm', 'pdfinfo', 'pdftocairo']
        
        for cmd in poppler_commands:
            try:
                result = subprocess.run([cmd, '-v'], capture_output=True, text=True, timeout=5)
                poppler_status[cmd] = {
                    'available': True,
                    'version': result.stderr.strip() if result.stderr else result.stdout.strip()
                }
            except Exception as e:
                poppler_status[cmd] = {'available': False, 'error': str(e)}
        
        return jsonify({
            'pdf_libraries': pdf_status,
            'poppler_utilities': poppler_status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/diagnostics', methods=['GET'])
def diagnostics():
    """Detailed system diagnostics"""
    try:
        import sys
        import platform
        import psutil
        
        # Get memory info
        memory = psutil.virtual_memory()
        
        # Check if required packages are importable
        packages_status = {}
        packages_to_check = [
            'PyPDF2', 'pdfplumber', 'pytesseract', 'pdf2image', 
            'PIL', 'requests', 'openai', 'anthropic', 'pandas', 'numpy'
        ]
        
        for package in packages_to_check:
            try:
                __import__(package)
                packages_status[package] = 'available'
            except ImportError:
                packages_status[package] = 'missing'

        return jsonify({
            'system': {
                'platform': platform.platform(),
                'python_version': sys.version,
                'memory_total_gb': round(memory.total / (1024**3), 2),
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'memory_percent_used': memory.percent
            },
            'packages': packages_status,
            'environment': {
                'render_env': os.environ.get('RENDER', 'false'),
                'port': os.environ.get('PORT', 'not set'),
                'tessdata_prefix': os.environ.get('TESSDATA_PREFIX', 'not set'),
                'omp_thread_limit': os.environ.get('OMP_THREAD_LIMIT', 'not set')
            },
            'cors_enabled': True,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Diagnostics error: {e}")
        return jsonify({
            'error': str(e),
            'cors_enabled': True,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
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
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
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
        
        # Initialize components with enhanced PDF extractor
        try:
            ch_client = CompaniesHouseClient()
            content_processor = ContentProcessor()
            pdf_extractor = EnhancedPDFExtractor()  # Use enhanced version
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
        
        # Extract content from PDFs with enhanced methods
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
                return jsonify({
                    'error': 'No readable content could be extracted from the PDF files',
                    'details': 'The PDFs may be image-based or protected. Try different years or another company.',
                    'suggestions': [
                        'Try company 00445790 (Tesco PLC)',
                        'Try company 00000006 (Bank of England)', 
                        'Try different years (2022, 2021, 2020)'
                    ],
                    'debug_url': f'{request.host_url}test-ocr'
                }), 422
            
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
            'model_used': archetype_analysis.get('model_used', 'pattern_based'),
            'extraction_methods': [content['metadata']['extraction_method'] for content in extracted_content]
        }
        
        logger.info(f"Analysis completed successfully for {company_name}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Unexpected error in analysis: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

class EnhancedPDFExtractor(PDFExtractor):
    """Enhanced PDF extractor with additional fallback methods"""
    
    def extract_text_from_pdf(self, pdf_content, filename):
        """Extract text with multiple fallback methods"""
        
        # Try original methods first
        result = super().extract_text_from_pdf(pdf_content, filename)
        
        if result["extraction_status"] == "success":
            return result
        
        # Try enhanced fallback methods
        logger.info(f"Trying enhanced extraction methods for {filename}")
        
        # Method 1: Basic PyPDF2 with different settings
        try:
            fallback_result = self._extract_with_basic_pypdf2(pdf_content, filename)
            if fallback_result["extraction_status"] == "success":
                return fallback_result
        except Exception as e:
            logger.warning(f"Basic PyPDF2 fallback failed: {e}")
        
        # Method 2: Try with different OCR settings
        try:
            ocr_result = self._extract_with_enhanced_ocr(pdf_content, filename)
            if ocr_result["extraction_status"] == "success":
                return ocr_result
        except Exception as e:
            logger.warning(f"Enhanced OCR fallback failed: {e}")
        
        # If all methods fail, return detailed error
        return {
            "extraction_status": "failed",
            "error": "All extraction methods failed including enhanced fallbacks",
            "methods_tried": ["pypdf2", "pdfplumber", "ocr", "basic_pypdf2_fallback", "enhanced_ocr"],
            "suggestion": "PDF may be image-based, password-protected, or corrupted"
        }
    
    def _extract_with_basic_pypdf2(self, pdf_content, filename):
        """Fallback extraction method for problematic PDFs"""
        try:
            import io
            from PyPDF2 import PdfReader
            
            pdf_file = io.BytesIO(pdf_content)
            reader = PdfReader(pdf_file)
            
            text = ""
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and len(page_text.strip()) > 20:
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text + "\n"
                except Exception as page_e:
                    logger.warning(f"Failed to extract page {page_num}: {page_e}")
                    continue
            
            if len(text.strip()) > 100:
                return {
                    "extraction_status": "success",
                    "raw_text": text,
                    "extraction_method": "basic_pypdf2_fallback",
                    "pages_extracted": len(reader.pages)
                }
            else:
                return {
                    "extraction_status": "failed",
                    "error": f"Insufficient text extracted: {len(text)} characters"
                }
                
        except Exception as e:
            return {
                "extraction_status": "failed", 
                "error": f"Basic PyPDF2 extraction failed: {str(e)}"
            }
    
    def _extract_with_enhanced_ocr(self, pdf_content, filename):
        """Enhanced OCR with different settings"""
        try:
            import pytesseract
            from pdf2image import convert_from_bytes
            import tempfile
            import os
            
            # Convert PDF to images with different settings
            try:
                # Try with higher DPI for better OCR
                images = convert_from_bytes(
                    pdf_content, 
                    dpi=300,  # Higher DPI
                    fmt='png',
                    thread_count=1
                )
            except Exception as e:
                logger.warning(f"High DPI conversion failed: {e}")
                # Fallback to lower DPI
                images = convert_from_bytes(
                    pdf_content, 
                    dpi=150,
                    fmt='png',
                    thread_count=1
                )
            
            text = ""
            for i, image in enumerate(images):
                try:
                    # Try different OCR configurations
                    config_options = [
                        '--psm 1',  # Automatic page segmentation with OSD
                        '--psm 3',  # Fully automatic page segmentation (default)
                        '--psm 6',  # Assume a single uniform block of text
                        '--psm 11', # Sparse text
                    ]
                    
                    page_text = ""
                    for config in config_options:
                        try:
                            page_text = pytesseract.image_to_string(image, config=config)
                            if len(page_text.strip()) > 50:
                                break
                        except:
                            continue
                    
                    if page_text and len(page_text.strip()) > 20:
                        text += f"\n--- Page {i + 1} ---\n"
                        text += page_text + "\n"
                        
                except Exception as page_e:
                    logger.warning(f"OCR failed for page {i}: {page_e}")
                    continue
            
            if len(text.strip()) > 100:
                return {
                    "extraction_status": "success",
                    "raw_text": text,
                    "extraction_method": "enhanced_ocr_fallback",
                    "pages_processed": len(images)
                }
            else:
                return {
                    "extraction_status": "failed",
                    "error": f"Enhanced OCR extracted insufficient text: {len(text)} characters"
                }
                
        except Exception as e:
            return {
                "extraction_status": "failed",
                "error": f"Enhanced OCR failed: {str(e)}"
            }

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Handle CORS preflight requests globally
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({'status': 'ok'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

if __name__ == '__main__':
    # For local development
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)