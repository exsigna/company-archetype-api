#!/usr/bin/env python3
"""
Flask API for Strategic Analysis Tool with Database Integration
Fixed for Flask 2.3+ compatibility and Gunicorn deployment
Updated for ExecutiveAIAnalyzer (Board-Grade Analysis)
"""

import time
import os
import sys
import json
import logging
import uuid
import gc
import psutil
from datetime import datetime, date
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import your existing modules with better error handling
missing_modules = []
try:
    from config import validate_config, validate_company_number
except ImportError as e:
    missing_modules.append(f"config: {e}")

try:
    from companies_house_client import CompaniesHouseClient
except ImportError as e:
    missing_modules.append(f"companies_house_client: {e}")

try:
    from content_processor import ContentProcessor
except ImportError as e:
    missing_modules.append(f"content_processor: {e}")

try:
    from pdf_extractor import PDFExtractor, ParallelPDFExtractor, extract_multiple_files_parallel, extract_files_in_batches
except ImportError as e:
    missing_modules.append(f"pdf_extractor: {e}")

try:
    from ai_analyzer import ExecutiveAIAnalyzer
except ImportError as e:
    missing_modules.append(f"ai_analyzer: {e}")

try:
    from file_manager import FileManager
except ImportError as e:
    missing_modules.append(f"file_manager: {e}")

try:
    from report_generator import ReportGenerator
except ImportError as e:
    missing_modules.append(f"report_generator: {e}")

try:
    from database import AnalysisDatabase
except ImportError as e:
    missing_modules.append(f"database: {e}")
    # Create a dummy AnalysisDatabase class to prevent errors
    class AnalysisDatabase:
        def __init__(self):
            pass
        def test_connection(self):
            return False

# Report missing modules but don't crash immediately
if missing_modules:
    print("⚠️  Warning: Some modules could not be imported:")
    for module in missing_modules:
        print(f"   - {module}")
    print("   The API will start but some features may not work.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
MAX_PARALLEL_WORKERS = int(os.environ.get('MAX_PARALLEL_WORKERS', 4))
PDF_EXTRACTION_TIMEOUT = int(os.environ.get('PDF_EXTRACTION_TIMEOUT', 300))
MAX_FILES_PER_ANALYSIS = int(os.environ.get('MAX_FILES_PER_ANALYSIS', 20))
MEMORY_WARNING_THRESHOLD = int(os.environ.get('MEMORY_WARNING_THRESHOLD', 85))

# Global helper function for JSON serialization
def make_json_serializable(obj):
    """Convert dates and other non-serializable objects to strings"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif hasattr(obj, 'isoformat'):  # Handle other date-like objects
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

def check_memory_usage():
    """Check memory usage and log warnings if high"""
    try:
        memory = psutil.virtual_memory()
        if memory.percent > MEMORY_WARNING_THRESHOLD:
            logger.warning(f"⚠️ High memory usage: {memory.percent:.1f}%")
            gc.collect()
            return True
        return False
    except Exception as e:
        logger.debug(f"Could not check memory usage: {e}")
        return False

# Initialize components with better error handling
components_status = {}

def safe_init_component(name, init_func):
    """Safely initialize a component and track its status"""
    try:
        component = init_func()
        components_status[name] = {'status': 'ok', 'instance': component}
        logger.info(f"✅ {name} initialized successfully")
        return component
    except Exception as e:
        components_status[name] = {'status': 'error', 'error': str(e)}
        logger.error(f"❌ {name} initialization failed: {e}")
        return None

# Initialize components
db = None  # Initialize to None first
try:
    ch_client = safe_init_component('CompaniesHouseClient', CompaniesHouseClient)
    content_processor = safe_init_component('ContentProcessor', ContentProcessor)
    pdf_extractor = safe_init_component('PDFExtractor', PDFExtractor)
    parallel_pdf_extractor = safe_init_component('ParallelPDFExtractor', lambda: ParallelPDFExtractor(max_workers=MAX_PARALLEL_WORKERS))
    archetype_analyzer = safe_init_component('ExecutiveAIAnalyzer', ExecutiveAIAnalyzer)
    file_manager = safe_init_component('FileManager', FileManager)
    report_generator = safe_init_component('ReportGenerator', ReportGenerator)
    
    # Handle AnalysisDatabase separately since it might not be available
    if 'database' not in [m.split(':')[0] for m in missing_modules]:
        db = safe_init_component('AnalysisDatabase', AnalysisDatabase)
    else:
        logger.warning("⚠️ Database module not available - database features will be disabled")
        components_status['AnalysisDatabase'] = {'status': 'error', 'error': 'Module not found'}
    
    # Log overall status
    successful_components = sum(1 for comp in components_status.values() if comp['status'] == 'ok')
    total_components = len(components_status)
    logger.info(f"Component initialization: {successful_components}/{total_components} successful")
    
except Exception as e:
    logger.error(f"Error during component initialization: {e}")
    # Ensure db is set to None if there was an error
    if 'db' not in locals():
        db = None

def initialize_app():
    """Initialize app and test database connection"""
    logger.info("🚀 Initializing Strategic Analysis API...")
    
    try:
        # Validate configuration if available
        if 'config' not in [m.split(':')[0] for m in missing_modules]:
            if not validate_config():
                logger.error("❌ Configuration validation failed")
                return
            else:
                logger.info("✅ Configuration validated")
        
        # Test database connection if available
        if db and components_status.get('AnalysisDatabase', {}).get('status') == 'ok':
            success = db.test_connection()
            if success:
                logger.info("✅ Database connected successfully")
            else:
                logger.error("❌ Database connection failed")
        
        logger.info("✅ Strategic Analysis API initialized successfully")
    except Exception as e:
        logger.error(f"❌ Error during initialization: {e}")

def create_app():
    """
    Application factory pattern for Flask
    Better for production deployment with Gunicorn
    """
    app = Flask(__name__)
    CORS(app)
    
    # Initialize the app
    initialize_app()
    
    @app.route('/')
    def home():
        """Enhanced home page with component status"""
        component_statuses = {}
        for name, info in components_status.items():
            component_statuses[name] = info['status']
        
        return jsonify({
            'service': 'Strategic Analysis API',
            'status': 'running',
            'version': '3.0.0',  # Updated version for board-grade analysis
            'component_status': component_statuses,
            'features': [
                'Board-grade strategic analysis with executive insights',
                'Multi-file analysis with individual file processing',
                'Executive AI archetype classification',
                'Strategic risk heatmap generation',
                'Board action items and recommendations',
                'Intelligent content sampling (15K chars)',
                'File-by-file synthesis and confidence scoring',
                'Comprehensive evidence-based reasoning',
                'Parallel PDF processing with fallback',
                'Memory usage monitoring and optimization'
            ],
            'configuration': {
                'max_parallel_workers': MAX_PARALLEL_WORKERS,
                'pdf_extraction_timeout': PDF_EXTRACTION_TIMEOUT,
                'max_files_per_analysis': MAX_FILES_PER_ANALYSIS,
                'memory_warning_threshold': f"{MEMORY_WARNING_THRESHOLD}%"
            },
            'endpoints': {
                'analyze': '/api/analyze',
                'years': '/api/years/<company_number>',
                'lookup_company': '/api/company/lookup/<company_name_or_number>',
                'check_company': '/api/company/check',
                'history': '/api/analysis/history/<company_number>',
                'recent': '/api/analysis/recent',
                'search': '/api/analysis/search/<term>',
                'test_db': '/api/database/test',
                'preview_cleanup': '/api/database/preview-cleanup/<company_number>',
                'cleanup_analysis': '/api/database/cleanup/<company_number>/<analysis_id>',
                'cleanup_invalid': '/api/database/cleanup/invalid/<company_number>',
                'database_stats': '/api/database/stats',
                'system_status': '/api/system/status',
                'validate_request': '/api/validate/request'
            },
            'safety_note': 'Always use preview-cleanup before cleanup-invalid to see what will be deleted'
        })

    @app.route('/health')
    def health_check():
        """Enhanced health check with component status and memory usage"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Check critical components
            critical_components = ['CompaniesHouseClient', 'PDFExtractor', 'ParallelPDFExtractor']
            critical_status = all(
                components_status.get(comp, {}).get('status') == 'ok' 
                for comp in critical_components
            )
            
            health_status = 'healthy' if critical_status else 'degraded'
            
            return jsonify({
                'status': health_status,
                'timestamp': datetime.now().isoformat(),
                'components': {name: info['status'] for name, info in components_status.items()},
                'system': {
                    'memory_usage_percent': round(memory.percent, 1),
                    'memory_available_gb': round(memory.available / (1024**3), 1),
                    'disk_usage_percent': round(disk.percent, 1),
                    'disk_free_gb': round(disk.free / (1024**3), 1)
                },
                'configuration': {
                    'max_parallel_workers': MAX_PARALLEL_WORKERS,
                    'extraction_timeout': PDF_EXTRACTION_TIMEOUT
                }
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/company/lookup/<company_identifier>')
    def lookup_company_analysis(company_identifier):
        """Look up previous analyses for a company by name or number"""
        if not db or components_status.get('AnalysisDatabase', {}).get('status') != 'ok':
            return jsonify({
                'success': False,
                'error': 'Database not available'
            }), 503
        
        try:
            logger.info(f"Looking up previous analyses for: {company_identifier}")
            
            # First, try to find by company number (exact match)
            if validate_company_number(company_identifier):
                results = db.get_analysis_by_company(company_identifier)
                if results:
                    analysis_metadata = []
                    for result in results:
                        metadata = {
                            'analysis_id': result.get('id'),
                            'company_number': result.get('company_number'),
                            'company_name': result.get('company_name'),
                            'analysis_date': result.get('analysis_date'),
                            'years_analyzed': result.get('years_analyzed', []),
                            'files_processed': result.get('files_processed', 0),
                            'business_strategy': result.get('business_strategy_dominant'),
                            'risk_strategy': result.get('risk_strategy_dominant'),
                            'status': result.get('status'),
                            'analysis_type': result.get('analysis_type', 'unknown')
                        }
                        analysis_metadata.append(metadata)
                    
                    return jsonify({
                        'success': True,
                        'found': True,
                        'search_term': company_identifier,
                        'search_type': 'company_number',
                        'company_number': results[0].get('company_number'),
                        'company_name': results[0].get('company_name'),
                        'total_analyses': len(results),
                        'analyses': analysis_metadata
                    })
            
            # If not a company number or no results, search by company name
            search_results = db.search_companies(company_identifier)
            
            if search_results:
                first_match = search_results[0]
                detailed_analyses = db.get_analysis_by_company(first_match['company_number'])
                
                analysis_metadata = []
                for result in detailed_analyses:
                    metadata = {
                        'analysis_id': result.get('id'),
                        'company_number': result.get('company_number'),
                        'company_name': result.get('company_name'),
                        'analysis_date': result.get('analysis_date'),
                        'years_analyzed': result.get('years_analyzed', []),
                        'files_processed': result.get('files_processed', 0),
                        'business_strategy': result.get('business_strategy_dominant'),
                        'risk_strategy': result.get('risk_strategy_dominant'),
                        'status': result.get('status'),
                        'analysis_type': result.get('analysis_type', 'unknown')
                    }
                    analysis_metadata.append(metadata)
                
                return jsonify({
                    'success': True,
                    'found': True,
                    'search_term': company_identifier,
                    'search_type': 'company_name',
                    'company_number': first_match['company_number'],
                    'company_name': first_match['company_name'],
                    'total_analyses': len(detailed_analyses),
                    'analyses': analysis_metadata,
                    'other_matches': len(search_results) - 1 if len(search_results) > 1 else 0
                })
            
            return jsonify({
                'success': True,
                'found': False,
                'search_term': company_identifier,
                'message': 'No previous analyses found for this company',
                'suggestion': 'Use /api/analyze to create a new analysis'
            })
            
        except Exception as e:
            logger.error(f"Error looking up company {company_identifier}: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/company/check', methods=['POST'])
    def check_company_before_analysis():
        """Check if a company has previous analyses before running new analysis"""
        try:
            data = request.get_json()
            if not data or not data.get('company_identifier'):
                return jsonify({
                    'success': False,
                    'error': 'company_identifier is required'
                }), 400
            
            company_identifier = data.get('company_identifier', '').strip()
            return lookup_company_analysis(company_identifier)
            
        except Exception as e:
            logger.error(f"Error checking company: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/years/<company_number>')
    def get_available_years(company_number):
        """Get available filing years for a company"""
        if not ch_client or components_status.get('CompaniesHouseClient', {}).get('status') != 'ok':
            return jsonify({
                'success': False,
                'error': 'Companies House client not available'
            }), 503
        
        try:
            if not validate_company_number(company_number):
                return jsonify({
                    'success': False,
                    'error': 'Invalid company number format'
                }), 400
            
            logger.info(f"Getting available years for company {company_number}")
            
            filing_history = ch_client.get_filing_history(company_number)
            
            if not filing_history:
                return jsonify({
                    'success': False,
                    'error': 'Could not retrieve filing history'
                }), 404
            
            # Extract available years from filings
            available_years = []
            for filing in filing_history.get('items', []):
                if filing.get('category') == 'accounts':
                    date = filing.get('date', '')
                    if date:
                        try:
                            year = datetime.fromisoformat(date.replace('Z', '+00:00')).year
                            if year not in available_years:
                                available_years.append(year)
                        except:
                            continue
            
            # Sort years (most recent first) and limit to 20
            available_years.sort(reverse=True)
            available_years = available_years[:20]
            
            return jsonify({
                'success': True,
                'company_number': company_number,
                'years': available_years,
                'count': len(available_years)
            })
            
        except Exception as e:
            logger.error(f"Error getting years for {company_number}: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/analyze', methods=['POST'])
    def analyze_company():
        """Enhanced main analysis endpoint with board-grade analysis"""
        
        # Check if critical components are available
        required_components = ['CompaniesHouseClient', 'ContentProcessor', 'ExecutiveAIAnalyzer']
        missing_components = [
            comp for comp in required_components 
            if components_status.get(comp, {}).get('status') != 'ok'
        ]
        
        if missing_components:
            return jsonify({
                'success': False,
                'error': f'Required components not available: {", ".join(missing_components)}'
            }), 503
        
        # Generate unique request ID to track this analysis
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"🆔 Analysis request {request_id} started")
        
        # Check memory usage at start
        check_memory_usage()
        
        try:
            # Get request data
            data = request.get_json()
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'No JSON data provided'
                }), 400
            
            company_number = data.get('company_number', '').strip()
            years = data.get('years', [])
            analysis_context = data.get('analysis_context', 'Strategic Review')  # New parameter for board context
            
            # Validate inputs
            if not company_number:
                return jsonify({
                    'success': False,
                    'error': 'Company number is required'
                }), 400
            
            if not validate_company_number(company_number):
                return jsonify({
                    'success': False,
                    'error': 'Invalid company number format'
                }), 400
            
            if not years or not isinstance(years, list):
                return jsonify({
                    'success': False,
                    'error': 'Years array is required'
                }), 400
            
            if len(years) > MAX_FILES_PER_ANALYSIS:
                return jsonify({
                    'success': False,
                    'error': f'Too many years requested. Maximum: {MAX_FILES_PER_ANALYSIS}'
                }), 400
            
            logger.info(f"🚀 Request {request_id}: Starting board-grade analysis for company {company_number}, years: {years}")
            
            # Validate company exists
            exists, company_name = ch_client.validate_company_exists(company_number)
            if not exists:
                return jsonify({
                    'success': False,
                    'error': f'Company {company_number} not found'
                }), 404
            
            logger.info(f"✅ Company validated: {company_name}")
            
            # Download filings
            logger.info("📥 Downloading filings...")
            max_years_needed = max(datetime.now().year - min(years) + 2, 6)
            download_results = download_company_filings(company_number, max_years_needed)
            
            if not download_results or download_results['total_downloaded'] == 0:
                return jsonify({
                    'success': False,
                    'error': 'No annual accounts could be downloaded'
                }), 404
            
            # Filter files to selected years
            logger.info(f"🔍 Filtering files to selected years: {years}")
            filtered_files = filter_files_by_years(download_results['downloaded_files'], years)
            
            if not filtered_files:
                return jsonify({
                    'success': False,
                    'error': 'No files found for the selected years'
                }), 404
            
            logger.info(f"📋 Found {len(filtered_files)} files matching selected years")
            
            # Check memory before extraction
            check_memory_usage()
            
            # Extract content from PDFs with improved method
            logger.info(f"📄 Extracting content from {len(filtered_files)} files...")
            extracted_content = extract_content_from_files(filtered_files)
            
            if not extracted_content:
                return jsonify({
                    'success': False,
                    'error': 'No readable content could be extracted from the files'
                }), 500
            
            logger.info(f"✅ Successfully extracted content from {len(extracted_content)} files")
            
            # Check memory before analysis
            check_memory_usage()
            
            # Enhanced content processing and board-grade analysis
            logger.info("🧠 Processing and analyzing content for board presentation...")
            analysis_results = process_and_analyze_content_for_board(
                extracted_content, company_name, company_number, analysis_context
            )
            
            if not analysis_results:
                return jsonify({
                    'success': False,
                    'error': 'Content analysis failed'
                }), 500
            
            # Prepare enhanced response data with board-grade insights
            board_analysis = analysis_results.get('board_analysis', {})
            
            response_data = {
                'success': True,
                'company_number': company_number,
                'company_name': company_name,
                'years_analyzed': years,
                'files_processed': len(extracted_content),
                'analysis_context': analysis_context,
                
                # Board-grade analysis results
                'executive_summary': board_analysis.get('executive_summary', ''),
                'business_strategy_analysis': board_analysis.get('business_strategy_analysis', {}),
                'risk_strategy_analysis': board_analysis.get('risk_strategy_analysis', {}),
                'strategic_recommendations': board_analysis.get('strategic_recommendations', []),
                'executive_dashboard': board_analysis.get('executive_dashboard', {}),
                'board_presentation_summary': board_analysis.get('board_presentation_summary', {}),
                
                # Legacy compatibility fields
                'business_strategy': board_analysis.get('business_strategy_analysis', {}),
                'risk_strategy': board_analysis.get('risk_strategy_analysis', {}),
                
                'analysis_date': datetime.now().isoformat(),
                'analysis_type': board_analysis.get('analysis_metadata', {}).get('analysis_type', 'board_grade_executive'),
                'confidence_level': board_analysis.get('analysis_metadata', {}).get('confidence_level', 'medium'),
                'processing_stats': {
                    'parallel_extraction_used': len(filtered_files) > 1,
                    'total_content_length': sum(len(content.get('content', '')) for content in extracted_content),
                    'extraction_methods': list(set(
                        content.get('metadata', {}).get('extraction_method', 'unknown') 
                        for content in extracted_content
                    )),
                    'board_grade_analysis': True
                }
            }
            
            # Store in database if available
            if db and components_status.get('AnalysisDatabase', {}).get('status') == 'ok':
                try:
                    logger.info(f"💾 Request {request_id}: Storing board-grade analysis results in database...")
                    serializable_response = make_json_serializable(response_data)
                    record_id = db.store_analysis_result(serializable_response)
                    response_data['database_id'] = record_id
                    logger.info(f"✅ Request {request_id}: Board-grade analysis stored in database with ID: {record_id}")
                    
                except Exception as db_error:
                    logger.error(f"❌ Request {request_id}: Database storage failed: {str(db_error)}")
                    response_data['database_warning'] = 'Analysis completed but database storage had issues'
            else:
                response_data['database_warning'] = 'Database not available - results not stored'
            
            # Clean up temporary files
            try:
                if ch_client:
                    ch_client.cleanup_temp_files()
                logger.info("🧹 Temporary files cleaned up")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Cleanup warning: {cleanup_error}")
            
            # Final memory cleanup
            gc.collect()
            
            logger.info(f"🎉 Request {request_id}: Board-grade analysis completed successfully for {company_number}")
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"❌ Request {request_id}: Analysis failed: {e}")
            # Ensure cleanup on error
            try:
                if ch_client:
                    ch_client.cleanup_temp_files()
                gc.collect()
            except:
                pass
            
            return jsonify({
                'success': False,
                'error': str(e),
                'request_id': request_id
            }), 500

    # Database endpoints
    @app.route('/api/analysis/history/<company_number>')
    def get_company_history(company_number):
        """Get analysis history for a company"""
        if not db or components_status.get('AnalysisDatabase', {}).get('status') != 'ok':
            return jsonify({
                'success': False,
                'error': 'Database not available'
            }), 503
        
        try:
            results = db.get_analysis_by_company(company_number)
            return jsonify({
                'success': True,
                'company_number': company_number,
                'results': results,
                'count': len(results)
            })
        except Exception as e:
            logger.error(f"Error getting history for {company_number}: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/analysis/recent')
    def get_recent():
        """Get recent analyses"""
        if not db or components_status.get('AnalysisDatabase', {}).get('status') != 'ok':
            return jsonify({
                'success': False,
                'error': 'Database not available'
            }), 503
        
        try:
            results = db.get_recent_analyses()
            return jsonify({
                'success': True,
                'results': results,
                'count': len(results)
            })
        except Exception as e:
            logger.error(f"Error getting recent analyses: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/analysis/search/<search_term>')
    def search_companies(search_term):
        """Search companies"""
        if not db or components_status.get('AnalysisDatabase', {}).get('status') != 'ok':
            return jsonify({
                'success': False,
                'error': 'Database not available'
            }), 503
        
        try:
            results = db.search_companies(search_term)
            return jsonify({
                'success': True,
                'search_term': search_term,
                'results': results,
                'count': len(results)
            })
        except Exception as e:
            logger.error(f"Error searching companies: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/database/test')
    def test_database_endpoint():
        """Test database connection"""
        if not db:
            return jsonify({
                'success': False,
                'message': 'Database component not initialized'
            }), 503
        
        try:
            success = db.test_connection()
            return jsonify({
                'success': success,
                'message': 'Database connection successful' if success else 'Database connection failed'
            })
        except Exception as e:
            logger.error(f"Database test failed: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/database/preview-cleanup/<company_number>')
    def preview_cleanup(company_number):
        """Preview what would be deleted without actually deleting - SAFETY FIRST"""
        if not db or components_status.get('AnalysisDatabase', {}).get('status') != 'ok':
            return jsonify({
                'success': False,
                'error': 'Database not available'
            }), 503
        
        try:
            logger.info(f"Previewing cleanup for company {company_number}")
            
            # Get all analyses for the company
            analyses = db.get_analysis_by_company(company_number)
            
            preview_results = []
            
            for analysis in analyses:
                issues = []
                will_delete = False
                
                # Parse raw_response if it's a string
                raw_response = analysis.get('raw_response')
                if isinstance(raw_response, str):
                    try:
                        raw_response = json.loads(raw_response)
                    except:
                        raw_response = {}
                
                # Check for HSBC specifically (wrong company)
                company_name = analysis.get('company_name', '')
                if 'HSBC' in company_name and company_number == '02613335':
                    issues.append("Wrong company name (HSBC for Together Personal Finance)")
                
                # Check for exact generic text matches (not partial)
                business_reasoning = analysis.get('business_strategy_reasoning', '')
                if business_reasoning == 'The company demonstrates strong growth-oriented strategies with focus on market expansion and innovation.':
                    issues.append("Exact generic business reasoning match")
                
                risk_reasoning = analysis.get('risk_strategy_reasoning', '')
                if risk_reasoning == 'Conservative risk management approach with emphasis on regulatory compliance and stable operations.':
                    issues.append("Exact generic risk reasoning match")
                
                # Check for the specific incomplete raw_response pattern
                if raw_response and isinstance(raw_response, dict):
                    if (raw_response.get('analysis_complete') == True and 
                        raw_response.get('success') == True and 
                        len(raw_response) == 2):  # Only has these 2 keys
                        issues.append("Incomplete raw_response (only analysis_complete and success)")
                
                # Only mark for deletion if multiple specific red flags (VERY CONSERVATIVE)
                if len(issues) >= 2:
                    will_delete = True
                
                preview_results.append({
                    'analysis_id': analysis.get('id'),
                    'company_name': analysis.get('company_name'),
                    'analysis_date': analysis.get('analysis_date'),
                    'years_analyzed': analysis.get('years_analyzed'),
                    'business_strategy': analysis.get('business_strategy_dominant'),
                    'risk_strategy': analysis.get('risk_strategy_dominant'),
                    'issues_found': issues,
                    'will_delete': will_delete,
                    'reasoning_length': {
                        'business': len(business_reasoning),
                        'risk': len(risk_reasoning)
                    },
                    'raw_response_keys': list(raw_response.keys()) if isinstance(raw_response, dict) else 'invalid'
                })
            
            to_delete = [r for r in preview_results if r['will_delete']]
            to_keep = [r for r in preview_results if not r['will_delete']]
            
            return jsonify({
                'success': True,
                'company_number': company_number,
                'total_analyses': len(preview_results),
                'will_delete': len(to_delete),
                'will_keep': len(to_keep),
                'to_delete': to_delete,
                'to_keep': to_keep,
                'all_analyses': preview_results,
                'warning': '⚠️  This is a preview only - no data has been deleted',
                'safety_note': 'Only analyses with multiple specific red flags will be deleted'
            })
            
        except Exception as e:
            logger.error(f"Error previewing cleanup for {company_number}: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/database/cleanup/<company_number>/<int:analysis_id>', methods=['DELETE'])
    def delete_specific_analysis(company_number, analysis_id):
        """Delete a specific analysis by ID"""
        if not db or components_status.get('AnalysisDatabase', {}).get('status') != 'ok':
            return jsonify({
                'success': False,
                'error': 'Database not available'
            }), 503
        
        try:
            logger.info(f"Attempting to delete analysis ID {analysis_id} for company {company_number}")
            
            success = db.delete_analysis_by_id(analysis_id, company_number)
            
            if success:
                logger.info(f"Successfully deleted analysis ID {analysis_id}")
                return jsonify({
                    'success': True,
                    'message': f'Deleted analysis {analysis_id} for company {company_number}'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Analysis not found or could not be deleted'
                }), 404
                
        except Exception as e:
            logger.error(f"Error deleting analysis {analysis_id}: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/database/cleanup/invalid/<company_number>', methods=['DELETE'])
    def cleanup_invalid_analyses(company_number):
        """Remove all invalid analysis entries for a company - USE WITH EXTREME CAUTION"""
        if not db or components_status.get('AnalysisDatabase', {}).get('status') != 'ok':
            return jsonify({
                'success': False,
                'error': 'Database not available'
            }), 503
        
        try:
            logger.warning(f"⚠️  DANGEROUS OPERATION: Cleaning up invalid analyses for company {company_number}")
            
            deleted_count = db.cleanup_invalid_analyses(company_number)
            
            return jsonify({
                'success': True,
                'message': f'Cleaned up {deleted_count} invalid analyses for company {company_number}',
                'deleted_count': deleted_count,
                'warning': 'This operation permanently deleted data. Use preview-cleanup first in future.'
            })
            
        except Exception as e:
            logger.error(f"Error cleaning up analyses for {company_number}: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/database/stats')
    def get_database_stats():
        """Get database statistics"""
        if not db or components_status.get('AnalysisDatabase', {}).get('status') != 'ok':
            return jsonify({
                'success': False,
                'error': 'Database not available'
            }), 503
        
        try:
            stats = db.get_analysis_statistics()
            return jsonify({
                'success': True,
                'statistics': stats
            })
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/analysis/summary/<company_number>/<int:analysis_id>')
    def get_analysis_summary(company_number, analysis_id):
        """Get detailed summary of a specific analysis"""
        if not db or components_status.get('AnalysisDatabase', {}).get('status') != 'ok':
            return jsonify({
                'success': False,
                'error': 'Database not available'
            }), 503
        
        try:
            analyses = db.get_analysis_by_company(company_number)
            target_analysis = None
            
            for analysis in analyses:
                if analysis.get('id') == analysis_id:
                    target_analysis = analysis
                    break
            
            if not target_analysis:
                return jsonify({
                    'success': False,
                    'error': 'Analysis not found'
                }), 404
            
            # Generate summary using the AI analyzer if available
            if archetype_analyzer and hasattr(archetype_analyzer, 'get_analysis_summary'):
                summary = archetype_analyzer.get_analysis_summary(target_analysis)
            else:
                # Fallback summary
                summary = {
                    'company_name': target_analysis.get('company_name', 'Unknown'),
                    'analysis_id': analysis_id,
                    'business_strategy': target_analysis.get('business_strategy_dominant', 'Unknown'),
                    'risk_strategy': target_analysis.get('risk_strategy_dominant', 'Unknown'),
                    'analysis_date': target_analysis.get('analysis_date', 'Unknown')
                }
            
            return jsonify({
                'success': True,
                'analysis_summary': summary
            })
            
        except Exception as e:
            logger.error(f"Error getting analysis summary: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/analysis/compare/<company_number>')
    def compare_analyses(company_number):
        """Compare multiple analyses for the same company to show evolution over time"""
        if not db or components_status.get('AnalysisDatabase', {}).get('status') != 'ok':
            return jsonify({
                'success': False,
                'error': 'Database not available'
            }), 503
        
        try:
            analyses = db.get_analysis_by_company(company_number)
            
            if len(analyses) < 2:
                return jsonify({
                    'success': False,
                    'error': 'Need at least 2 analyses to compare'
                }), 400
            
            # Sort by analysis date
            sorted_analyses = sorted(analyses, key=lambda x: x.get('analysis_date', ''), reverse=True)
            
            comparison_data = {
                'company_number': company_number,
                'company_name': sorted_analyses[0].get('company_name', 'Unknown'),
                'total_analyses': len(sorted_analyses),
                'comparison': []
            }
            
            for analysis in sorted_analyses:
                comparison_data['comparison'].append({
                    'analysis_id': analysis.get('id'),
                    'analysis_date': analysis.get('analysis_date'),
                    'years_analyzed': analysis.get('years_analyzed', []),
                    'files_processed': analysis.get('files_processed', 0),
                    'business_strategy': {
                        'dominant': analysis.get('business_strategy_dominant'),
                        'secondary': analysis.get('business_strategy_secondary')
                    },
                    'risk_strategy': {
                        'dominant': analysis.get('risk_strategy_dominant'),
                        'secondary': analysis.get('risk_strategy_secondary')
                    },
                    'analysis_type': analysis.get('analysis_type', 'unknown'),
                    'confidence_level': analysis.get('confidence_level', 'medium')
                })
            
            # Identify changes over time
            if len(sorted_analyses) >= 2:
                latest = sorted_analyses[0]
                previous = sorted_analyses[1]
                
                changes = {
                    'business_strategy_changed': latest.get('business_strategy_dominant') != previous.get('business_strategy_dominant'),
                    'risk_strategy_changed': latest.get('risk_strategy_dominant') != previous.get('risk_strategy_dominant'),
                    'analysis_improvement': latest.get('analysis_type', '').startswith('ai_') and not previous.get('analysis_type', '').startswith('ai_')
                }
                
                comparison_data['changes'] = changes
            
            return jsonify({
                'success': True,
                'comparison_data': comparison_data
            })
            
        except Exception as e:
            logger.error(f"Error comparing analyses for {company_number}: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/system/status')
    def system_status():
        """Get comprehensive system status"""
        try:
            # Get system memory info
            memory_info = {}
            try:
                memory = psutil.virtual_memory()
                memory_info = {
                    'total_gb': round(memory.total / (1024**3), 1),
                    'available_gb': round(memory.available / (1024**3), 1),
                    'usage_percent': round(memory.percent, 1),
                    'status': 'ok' if memory.percent < MEMORY_WARNING_THRESHOLD else 'warning'
                }
            except Exception as e:
                memory_info = {'error': str(e)}
            
            # Component capabilities
            capabilities = {
                'board_grade_analysis': archetype_analyzer is not None,
                'executive_insights': archetype_analyzer is not None,
                'strategic_recommendations': True,
                'multi_file_analysis': parallel_pdf_extractor is not None,
                'ai_archetype_classification': archetype_analyzer is not None,
                'enhanced_content_sampling': True,
                'individual_file_processing': True,
                'synthesis_and_confidence_scoring': True,
                'database_integration': db is not None
            }
            
            # Database status
            db_status = 'not_available'
            if db:
                try:
                    db_status = 'operational' if db.test_connection() else 'error'
                except:
                    db_status = 'error'
            
            status = {
                'service': 'Strategic Analysis API',
                'version': '3.0.0',  # Board-grade version
                'status': 'operational',
                'timestamp': datetime.now().isoformat(),
                'system': {
                    'memory': memory_info,
                    'cpu_count': os.cpu_count(),
                    'parallel_workers_configured': MAX_PARALLEL_WORKERS
                },
                'capabilities': capabilities,
                'components': {
                    'companies_house_client': components_status.get('CompaniesHouseClient', {}).get('status', 'error'),
                    'pdf_extractor': components_status.get('PDFExtractor', {}).get('status', 'error'),
                    'parallel_pdf_extractor': components_status.get('ParallelPDFExtractor', {}).get('status', 'error'),
                    'content_processor': components_status.get('ContentProcessor', {}).get('status', 'error'),
                    'executive_ai_analyzer': components_status.get('ExecutiveAIAnalyzer', {}).get('status', 'error'),
                    'database': db_status,
                    'file_manager': components_status.get('FileManager', {}).get('status', 'error'),
                    'report_generator': components_status.get('ReportGenerator', {}).get('status', 'error')
                },
                'component_errors': {
                    name: info.get('error', '') 
                    for name, info in components_status.items() 
                    if info.get('status') == 'error'
                }
            }
            
            return jsonify(status)
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return jsonify({
                'service': 'Strategic Analysis API',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @app.route('/api/validate/request', methods=['POST'])
    def validate_analysis_request():
        """Validate analysis request before processing"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({
                    'valid': False,
                    'errors': ['No JSON data provided']
                }), 400
            
            errors = []
            warnings = []
            
            # Check critical components
            required_components = ['CompaniesHouseClient', 'ContentProcessor', 'ExecutiveAIAnalyzer']
            missing_components = [
                comp for comp in required_components 
                if components_status.get(comp, {}).get('status') != 'ok'
            ]
            
            if missing_components:
                errors.append(f'Required components not available: {", ".join(missing_components)}')
            
            # Validate company number
            company_number = data.get('company_number', '').strip()
            if not company_number:
                errors.append('Company number is required')
            elif not validate_company_number(company_number):
                errors.append('Invalid company number format')
            
            # Validate years
            years = data.get('years', [])
            if not years:
                errors.append('Years array is required')
            elif not isinstance(years, list):
                errors.append('Years must be an array')
            elif len(years) > MAX_FILES_PER_ANALYSIS:
                errors.append(f'Too many years selected (max: {MAX_FILES_PER_ANALYSIS})')
            elif len(years) > 10:
                warnings.append('More than 10 years selected - this may take a long time')
            elif len(years) == 1:
                warnings.append('Only 1 year selected - multi-file analysis benefits require multiple years')
            
            # Check memory
            try:
                memory = psutil.virtual_memory()
                if memory.percent > 80:
                    warnings.append(f'High system memory usage ({memory.percent:.1f}%) - analysis may be slower')
            except:
                pass
            
            # Check if company exists (if no critical errors so far)
            company_exists = False
            company_name = 'Unknown'
            if not errors and company_number and ch_client:
                try:
                    company_exists, company_name = ch_client.validate_company_exists(company_number)
                    if not company_exists:
                        errors.append(f'Company {company_number} not found')
                except Exception as e:
                    warnings.append(f'Could not verify company existence: {str(e)}')
            
            # Check for previous analyses
            previous_analyses = []
            if company_exists and db and components_status.get('AnalysisDatabase', {}).get('status') == 'ok':
                try:
                    previous_analyses = db.get_analysis_by_company(company_number)
                    if previous_analyses:
                        warnings.append(f'Company has {len(previous_analyses)} previous analyses')
                except Exception as e:
                    warnings.append(f'Could not check previous analyses: {str(e)}')
            
            validation_result = {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'company_info': {
                    'company_number': company_number,
                    'company_name': company_name,
                    'exists': company_exists
                },
                'request_info': {
                    'years_count': len(years),
                    'years_selected': years,
                    'estimated_files': len(years),
                    'previous_analyses_count': len(previous_analyses)
                },
                'system_capabilities': {
                    'board_grade_analysis_available': archetype_analyzer is not None,
                    'ai_analysis_available': archetype_analyzer is not None,
                    'parallel_processing_available': parallel_pdf_extractor is not None,
                    'database_available': db is not None,
                    'max_recommended_years': 10,
                    'max_allowed_years': MAX_FILES_PER_ANALYSIS
                },
                'component_status': {name: info['status'] for name, info in components_status.items()}
            }
            
            return jsonify(validation_result)
            
        except Exception as e:
            logger.error(f"Error validating request: {e}")
            return jsonify({
                'valid': False,
                'errors': [f'Validation error: {str(e)}']
            }), 500

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({'error': 'Internal server error'}), 500

    @app.errorhandler(503)
    def service_unavailable(error):
        return jsonify({'error': 'Service temporarily unavailable'}), 503

    return app

# Helper functions for the analysis process
def download_company_filings(company_number, max_years):
    """Download company filings using existing method"""
    try:
        if not ch_client:
            raise Exception("Companies House client not available")
        
        results = ch_client.download_annual_accounts(company_number, max_years)
        logger.info(f"📥 Downloaded {results['total_downloaded']} files for {company_number}")
        return results
    except Exception as e:
        logger.error(f"❌ Error downloading filings for {company_number}: {e}")
        return None

def filter_files_by_years(downloaded_files, selected_years):
    """Filter downloaded files to only include selected years"""
    filtered_files = []
    
    for file_info in downloaded_files:
        file_date = file_info.get('date', '')
        if file_date:
            try:
                if isinstance(file_date, str):
                    if 'T' in file_date:
                        file_year = datetime.fromisoformat(file_date.replace('Z', '+00:00')).year
                    else:
                        file_year = datetime.strptime(file_date, '%Y-%m-%d').year
                else:
                    file_year = file_date.year
                
                if file_year in selected_years:
                    filtered_files.append(file_info)
                    logger.info(f"✅ Included file: {file_info['filename']} (Year {file_year})")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not determine year for {file_info['filename']}: {e}")
    
    return filtered_files

def extract_content_from_files(downloaded_files):
    """Extract content from downloaded files using parallel processing when beneficial"""
    try:
        logger.info(f"📄 Starting content extraction from {len(downloaded_files)} files")
        
        # Use parallel extraction for multiple files, sequential for single files
        if len(downloaded_files) > 1 and parallel_pdf_extractor:
            logger.info(f"🚀 Using parallel extraction for {len(downloaded_files)} files")
            extracted_content = extract_multiple_files_parallel(downloaded_files)
        else:
            logger.info("📄 Using sequential extraction")
            extracted_content = extract_content_from_files_legacy(downloaded_files)
        
        # Filter out None results and log statistics
        valid_content = [content for content in extracted_content if content is not None]
        
        logger.info(f"✅ Content extraction completed: {len(valid_content)}/{len(downloaded_files)} files successful")
        
        if len(valid_content) != len(downloaded_files):
            failed_count = len(downloaded_files) - len(valid_content)
            logger.warning(f"⚠️ {failed_count} files failed extraction")
        
        return valid_content
        
    except Exception as e:
        logger.error(f"❌ Error in content extraction: {e}")
        return []

def extract_content_from_files_legacy(downloaded_files):
    """Legacy sequential extraction method - kept for fallback"""
    if not pdf_extractor:
        logger.error("PDF extractor not available")
        return []
    
    extracted_content = []
    
    for file_info in downloaded_files:
        try:
            logger.info(f"📄 Extracting content from: {file_info['filename']}")
            
            # Read PDF content
            with open(file_info['path'], 'rb') as f:
                pdf_content = f.read()
            
            # Extract using PDFExtractor
            extraction_result = pdf_extractor.extract_text_from_pdf(
                pdf_content, file_info['filename']
            )
            
            if extraction_result["extraction_status"] == "success":
                content = extraction_result.get("raw_text", "")
                if content and len(content.strip()) > 100:
                    extracted_content.append({
                        'filename': file_info['filename'],
                        'date': file_info['date'],
                        'content': content,
                        'metadata': {
                            'file_size': file_info['size'],
                            'extraction_method': extraction_result["extraction_method"]
                        }
                    })
                    logger.info(f"✅ Successfully extracted {len(content):,} characters from {file_info['filename']}")
                else:
                    logger.warning(f"⚠️ Insufficient content extracted from {file_info['filename']}")
            else:
                logger.error(f"❌ Extraction failed for {file_info['filename']}")
                
        except Exception as e:
            logger.error(f"❌ Error extracting content from {file_info['filename']}: {e}")
    
    return extracted_content

def process_and_analyze_content_for_board(extracted_content, company_name, company_number, analysis_context):
    """Process and analyze content for board-grade insights"""
    try:
        if not content_processor or not archetype_analyzer:
            logger.error("Content processor or executive analyzer not available")
            return None
        
        logger.info(f"🧠 Starting board-grade content processing for {len(extracted_content)} files")
        
        # Process documents individually
        processed_documents = []
        for i, content_data in enumerate(extracted_content):
            logger.info(f"📋 Processing document {i+1}: {content_data['filename']}")
            processed = content_processor.process_document_content(
                content_data['content'], content_data['metadata']
            )
            processed_documents.append(processed)
        
        # Combine documents for overall analysis
        combined_analysis = content_processor.combine_multiple_documents(processed_documents)
        logger.info("✅ Document combination completed")
        
        # Prepare combined content
        combined_content = "\n\n".join([content_data['content'] for content_data in extracted_content])
        logger.info(f"📊 Combined content length: {len(combined_content):,} characters")
        
        # Board-grade archetype analysis with executive insights
        logger.info("🏛️ Performing board-grade archetype analysis with executive insights")
        
        # Ensure extracted_content dates are serializable before passing to analyzer
        serializable_extracted_content = []
        for content_data in extracted_content:
            serializable_content = {
                'filename': content_data['filename'],
                'date': make_json_serializable(content_data['date']),
                'content': content_data['content'],
                'metadata': content_data['metadata']
            }
            serializable_extracted_content.append(serializable_content)
        
        # Use the new board-grade analysis method
        board_analysis = archetype_analyzer.analyze_for_board(
            combined_content,
            company_name, 
            company_number,
            extracted_content=serializable_extracted_content,
            analysis_context=analysis_context
        )
        
        logger.info("✅ Board-grade archetype analysis completed")
        
        return {
            'processed_content': combined_analysis,
            'board_analysis': board_analysis,
            'document_count': len(extracted_content)
        }
        
    except Exception as e:
        logger.error(f"❌ Error in board-grade content processing: {e}")
        return None

# Legacy function for backward compatibility
def process_and_analyze_content_api(extracted_content, company_name, company_number):
    """Legacy function for backward compatibility - delegates to board analysis"""
    return process_and_analyze_content_for_board(extracted_content, company_name, company_number, 'Strategic Review')

# Create the Flask app
app = create_app()

if __name__ == '__main__':
    # Enhanced startup logging
    logger.info("=" * 70)
    logger.info("🚀 STRATEGIC ANALYSIS API v3.0.0 - BOARD-GRADE ANALYSIS")
    logger.info("=" * 70)
    logger.info("🔧 Features:")
    logger.info("   ✅ Board-grade strategic analysis with executive insights")
    logger.info("   ✅ Executive dashboard and strategic recommendations")
    logger.info("   ✅ Strategic risk heatmap for board oversight")
    logger.info("   ✅ Board presentation summary generation")
    logger.info("   ✅ Fixed Flask 2.3+ compatibility")
    logger.info("   ✅ Gunicorn deployment ready")
    logger.info("   ✅ Improved error handling and graceful degradation")
    logger.info("   ✅ Component status monitoring and health checks")
    logger.info("   ✅ Memory usage monitoring and optimization")
    logger.info("   ✅ Parallel PDF processing with fallback")
    logger.info("=" * 70)
    logger.info(f"📊 Configuration:")
    logger.info(f"   - Max parallel workers: {MAX_PARALLEL_WORKERS}")
    logger.info(f"   - PDF extraction timeout: {PDF_EXTRACTION_TIMEOUT}s")
    logger.info(f"   - Max files per analysis: {MAX_FILES_PER_ANALYSIS}")
    logger.info(f"   - Memory warning threshold: {MEMORY_WARNING_THRESHOLD}%")
    logger.info("=" * 70)
    
    # Final component status report
    successful_components = sum(1 for comp in components_status.values() if comp['status'] == 'ok')
    total_components = len(components_status)
    logger.info(f"🎯 Components ready: {successful_components}/{total_components}")
    
    if successful_components < total_components:
        logger.warning("⚠️  Some components failed to initialize - check logs above")
        failed_components = [name for name, info in components_status.items() if info['status'] == 'error']
        logger.warning(f"   Failed components: {', '.join(failed_components)}")
    
    logger.info("=" * 70)
    
    port = int(os.environ.get('PORT', 10000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🌐 Starting server on port {port} (debug: {debug_mode})")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)