#!/usr/bin/env python3
"""
Flask API for Strategic Analysis Tool with Database Integration
ENHANCED: Complete text preservation and improved field mapping
Fixed for Flask 2.3+ compatibility and Gunicorn deployment
Updated for ExecutiveAIAnalyzer (Board-Grade Analysis)
ENHANCED: Full reasoning text preservation throughout pipeline
FIXED: Confidence level extraction from AI analyzer
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

# ENHANCED: Extract archetype information with complete text preservation
def extract_archetype_from_analysis(result):
    """
    ENHANCED: Extract business and risk archetype names from analysis result with FULL TEXT PRESERVATION
    """
    try:
        logger.info(f"🔍 ENHANCED ARCHETYPE EXTRACTION from result keys: {list(result.keys())}")
        
        # Initialize with defaults
        business_strategy = 'Disciplined Specialist Growth'
        risk_strategy = 'Risk-First Conservative'
        
        # METHOD 1: Try direct field access first (from API response)
        if result.get('business_strategy_dominant'):
            business_strategy = result['business_strategy_dominant']
            logger.info(f"✅ Found business_strategy_dominant: {business_strategy}")
            
        if result.get('risk_strategy_dominant'):
            risk_strategy = result['risk_strategy_dominant']
            logger.info(f"✅ Found risk_strategy_dominant: {risk_strategy}")
        
        # METHOD 2: Try structured business/risk strategy objects (from AI analyzer)
        if 'business_strategy' in result and isinstance(result['business_strategy'], dict):
            business_obj = result['business_strategy']
            if business_obj.get('dominant'):
                business_strategy = business_obj['dominant']
                logger.info(f"✅ Found business_strategy.dominant: {business_strategy}")
                
        if 'risk_strategy' in result and isinstance(result['risk_strategy'], dict):
            risk_obj = result['risk_strategy']
            if risk_obj.get('dominant'):
                risk_strategy = risk_obj['dominant']
                logger.info(f"✅ Found risk_strategy.dominant: {risk_strategy}")
        
        # METHOD 3: Try raw_response as fallback
        if business_strategy == 'Disciplined Specialist Growth' or risk_strategy == 'Risk-First Conservative':
            raw_response = result.get('raw_response')
            if raw_response and isinstance(raw_response, str):
                try:
                    raw_data = json.loads(raw_response)
                    
                    if business_strategy == 'Disciplined Specialist Growth':
                        business_strategy = (raw_data.get('business_strategy', {}).get('dominant') or 
                                           raw_data.get('business_strategy_dominant') or 
                                           business_strategy)
                        
                    if risk_strategy == 'Risk-First Conservative':
                        risk_strategy = (raw_data.get('risk_strategy', {}).get('dominant') or 
                                       raw_data.get('risk_strategy_dominant') or 
                                       risk_strategy)
                        
                    logger.info(f"✅ Enhanced extraction from raw_response: {business_strategy}, {risk_strategy}")
                except Exception as parse_error:
                    logger.warning(f"Could not parse raw_response: {parse_error}")
        
        logger.info(f"🎯 FINAL EXTRACTED ARCHETYPES: Business='{business_strategy}', Risk='{risk_strategy}'")
        return business_strategy, risk_strategy
        
    except Exception as e:
        logger.error(f"❌ Error in enhanced archetype extraction: {e}")
        return 'Disciplined Specialist Growth', 'Risk-First Conservative'

# ENHANCED: Store analysis with complete text preservation
def store_analysis_with_enhanced_preservation(db, response_data):
    """
    ENHANCED: Store analysis ensuring complete text preservation
    """
    if not db or components_status.get('AnalysisDatabase', {}).get('status') != 'ok':
        logger.warning("Database not available for enhanced storage")
        return None
        
    try:
        logger.info(f"💾 ENHANCED STORAGE: Starting with {len(str(response_data))} chars of data")
        
        # CRITICAL: Make sure we preserve the COMPLETE reasoning texts
        if 'business_strategy' in response_data and isinstance(response_data['business_strategy'], dict):
            business_reasoning = response_data['business_strategy'].get('dominant_reasoning', '')
            logger.info(f"💾 Business reasoning to store: {len(business_reasoning)} characters")
            
        if 'risk_strategy' in response_data and isinstance(response_data['risk_strategy'], dict):
            risk_reasoning = response_data['risk_strategy'].get('dominant_reasoning', '')
            logger.info(f"💾 Risk reasoning to store: {len(risk_reasoning)} characters")
        
        # Ensure JSON serialization preserves unicode and full text
        serializable_response = make_json_serializable(response_data)
        
        # ENHANCED: Add explicit reasoning fields at top level for database storage
        if 'business_strategy' in response_data and isinstance(response_data['business_strategy'], dict):
            serializable_response['business_strategy_reasoning'] = response_data['business_strategy'].get('dominant_reasoning', '')
            serializable_response['business_strategy_definition'] = response_data['business_strategy'].get('dominant_definition', '')
            
        if 'risk_strategy' in response_data and isinstance(response_data['risk_strategy'], dict):
            serializable_response['risk_strategy_reasoning'] = response_data['risk_strategy'].get('dominant_reasoning', '')
            serializable_response['risk_strategy_definition'] = response_data['risk_strategy'].get('dominant_definition', '')
        
        logger.info(f"💾 ENHANCED: About to store with explicit reasoning fields")
        record_id = db.store_analysis_result(serializable_response)
        
        if record_id:
            logger.info(f"✅ ENHANCED STORAGE: Successfully stored analysis with ID: {record_id}")
            
            # VERIFICATION: Check what was actually stored
            stored_analyses = db.get_analysis_by_company(response_data.get('company_number', ''))
            if stored_analyses:
                latest = stored_analyses[0]  # Most recent
                business_len = len(latest.get('business_strategy_reasoning', ''))
                risk_len = len(latest.get('risk_strategy_reasoning', ''))
                logger.info(f"✅ VERIFICATION: Stored reasoning lengths - Business: {business_len}, Risk: {risk_len}")
        
        return record_id
        
    except Exception as e:
        logger.error(f"❌ Enhanced storage failed: {e}")
        return None

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
            'version': '3.1.0-CONFIDENCE-FIXED',  # Updated version for confidence fix
            'component_status': component_statuses,
            'features': [
                'FIXED: Confidence level calculation and extraction',
                'ENHANCED: Complete text preservation throughout pipeline',
                'Board-grade strategic analysis with executive insights',
                'Multi-file analysis with individual file processing',
                'Executive AI archetype classification',
                'Strategic risk heatmap generation',
                'Board action items and recommendations',
                'Intelligent content sampling (15K chars)',
                'File-by-file synthesis and confidence scoring',
                'Comprehensive evidence-based reasoning',
                'Parallel PDF processing with fallback',
                'Memory usage monitoring and optimization',
                'ENHANCED: Full reasoning text preservation and display'
            ],
            'configuration': {
                'max_parallel_workers': MAX_PARALLEL_WORKERS,
                'pdf_extraction_timeout': PDF_EXTRACTION_TIMEOUT,
                'max_files_per_analysis': MAX_FILES_PER_ANALYSIS,
                'memory_warning_threshold': f"{MEMORY_WARNING_THRESHOLD}%",
                'enhanced_text_preservation': True,
                'confidence_calculation_fixed': True
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
                    'extraction_timeout': PDF_EXTRACTION_TIMEOUT,
                    'enhanced_text_preservation': True,
                    'confidence_calculation_fixed': True
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
        """ENHANCED: Look up previous analyses with complete text retrieval"""
        if not db or components_status.get('AnalysisDatabase', {}).get('status') != 'ok':
            return jsonify({
                'success': False,
                'error': 'Database not available'
            }), 503
        
        try:
            logger.info(f"🔍 ENHANCED LOOKUP: Starting lookup for: {company_identifier}")
            
            # First, try to find by company number (exact match)
            if validate_company_number(company_identifier):
                results = db.get_analysis_by_company(company_identifier)
                if results:
                    analysis_metadata = []
                    for result in results:
                        # ENHANCED: Extract archetype info with better fallback handling
                        business_strategy, risk_strategy = extract_archetype_from_analysis(result)
                        
                        # ENHANCED: Get complete reasoning texts
                        business_reasoning = result.get('business_strategy_reasoning', '')
                        risk_reasoning = result.get('risk_strategy_reasoning', '')
                        
                        # ENHANCED: Try to get reasoning from raw_response if database fields are empty/short
                        if len(business_reasoning) < 50 or len(risk_reasoning) < 50:
                            raw_response = result.get('raw_response')
                            if raw_response:
                                try:
                                    if isinstance(raw_response, str):
                                        raw_data = json.loads(raw_response)
                                    else:
                                        raw_data = raw_response
                                    
                                    # Extract complete reasoning from raw_response
                                    if len(business_reasoning) < 50:
                                        business_reasoning = (raw_data.get('business_strategy', {}).get('dominant_reasoning') or 
                                                            raw_data.get('business_strategy_reasoning') or 
                                                            business_reasoning)
                                    
                                    if len(risk_reasoning) < 50:
                                        risk_reasoning = (raw_data.get('risk_strategy', {}).get('dominant_reasoning') or 
                                                        raw_data.get('risk_strategy_reasoning') or 
                                                        risk_reasoning)
                                    
                                    logger.info(f"✅ ENHANCED: Retrieved reasoning from raw_response - Business: {len(business_reasoning)}, Risk: {len(risk_reasoning)}")
                                except Exception as e:
                                    logger.warning(f"Could not parse raw_response for analysis {result.get('id')}: {e}")
                        
                        logger.info(f"🔍 ENHANCED: Analysis {result.get('id')} text lengths - Business: {len(business_reasoning)}, Risk: {len(risk_reasoning)}")
                        
                        metadata = {
                            'analysis_id': result.get('id'),
                            'company_number': result.get('company_number'),
                            'company_name': result.get('company_name'),
                            'analysis_date': result.get('analysis_date'),
                            'years_analyzed': result.get('years_analyzed', []),
                            'files_processed': result.get('files_processed', 0),
                            'business_strategy': business_strategy,
                            'business_strategy_dominant': business_strategy,
                            'business_strategy_reasoning': business_reasoning,  # ENHANCED: Include complete reasoning
                            'business_strategy_definition': result.get('business_strategy_definition', ''),
                            'risk_strategy': risk_strategy,
                            'risk_strategy_dominant': risk_strategy,
                            'risk_strategy_reasoning': risk_reasoning,  # ENHANCED: Include complete reasoning
                            'risk_strategy_definition': result.get('risk_strategy_definition', ''),
                            'status': result.get('status'),
                            'analysis_type': result.get('analysis_type', 'unknown'),
                            'confidence_level': result.get('confidence_level', 'medium'),
                            'raw_response': result.get('raw_response')  # ENHANCED: Include raw response for frontend processing
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
                        'analyses': analysis_metadata,
                        'enhanced_text_retrieval': True  # NEW: Flag indicating enhanced text retrieval
                    })
            
            # If not a company number or no results, search by company name
            search_results = db.search_companies(company_identifier)
            
            if search_results:
                first_match = search_results[0]
                detailed_analyses = db.get_analysis_by_company(first_match['company_number'])
                
                analysis_metadata = []
                for result in detailed_analyses:
                    # ENHANCED: Same processing as above for name-based searches
                    business_strategy, risk_strategy = extract_archetype_from_analysis(result)
                    
                    business_reasoning = result.get('business_strategy_reasoning', '')
                    risk_reasoning = result.get('risk_strategy_reasoning', '')
                    
                    # Try to get complete reasoning from raw_response if needed
                    if len(business_reasoning) < 50 or len(risk_reasoning) < 50:
                        raw_response = result.get('raw_response')
                        if raw_response:
                            try:
                                if isinstance(raw_response, str):
                                    raw_data = json.loads(raw_response)
                                else:
                                    raw_data = raw_response
                                
                                if len(business_reasoning) < 50:
                                    business_reasoning = (raw_data.get('business_strategy', {}).get('dominant_reasoning') or 
                                                        raw_data.get('business_strategy_reasoning') or 
                                                        business_reasoning)
                                
                                if len(risk_reasoning) < 50:
                                    risk_reasoning = (raw_data.get('risk_strategy', {}).get('dominant_reasoning') or 
                                                    raw_data.get('risk_strategy_reasoning') or 
                                                    risk_reasoning)
                            except Exception as e:
                                logger.warning(f"Could not parse raw_response for analysis {result.get('id')}: {e}")
                    
                    metadata = {
                        'analysis_id': result.get('id'),
                        'company_number': result.get('company_number'),
                        'company_name': result.get('company_name'),
                        'analysis_date': result.get('analysis_date'),
                        'years_analyzed': result.get('years_analyzed', []),
                        'files_processed': result.get('files_processed', 0),
                        'business_strategy': business_strategy,
                        'business_strategy_dominant': business_strategy,
                        'business_strategy_reasoning': business_reasoning,
                        'business_strategy_definition': result.get('business_strategy_definition', ''),
                        'risk_strategy': risk_strategy,
                        'risk_strategy_dominant': risk_strategy,
                        'risk_strategy_reasoning': risk_reasoning,
                        'risk_strategy_definition': result.get('risk_strategy_definition', ''),
                        'status': result.get('status'),
                        'analysis_type': result.get('analysis_type', 'unknown'),
                        'confidence_level': result.get('confidence_level', 'medium'),
                        'raw_response': result.get('raw_response')
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
                    'other_matches': len(search_results) - 1 if len(search_results) > 1 else 0,
                    'enhanced_text_retrieval': True
                })
            
            return jsonify({
                'success': True,
                'found': False,
                'search_term': company_identifier,
                'message': 'No previous analyses found for this company',
                'suggestion': 'Use /api/analyze to create a new analysis'
            })
            
        except Exception as e:
            logger.error(f"❌ ENHANCED LOOKUP ERROR for {company_identifier}: {e}")
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
        """ENHANCED: Main analysis endpoint with complete text preservation and FIXED confidence extraction"""
        
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
        logger.info(f"🆔 ENHANCED Analysis request {request_id} started")
        
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
            
            logger.info(f"🚀 Request {request_id}: Starting ENHANCED board-grade analysis for company {company_number}, years: {years}")
            
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
            
            # ***** ENHANCED DEBUG LOGGING SECTION *****
            logger.info(f"🔍 ENHANCED AI ANALYZER RESULT KEYS: {list(board_analysis.keys())}")
            
            # DETAILED logging of strategy objects
            if 'business_strategy' in board_analysis:
                bs = board_analysis['business_strategy']
                logger.info(f"🔍 BUSINESS STRATEGY TYPE: {type(bs)}")
                if isinstance(bs, dict):
                    logger.info(f"🔍 BUSINESS STRATEGY KEYS: {list(bs.keys())}")
                    dominant_reasoning = bs.get('dominant_reasoning', bs.get('reasoning', ''))
                    logger.info(f"🔍 BUSINESS REASONING LENGTH: {len(dominant_reasoning)} chars")
                    logger.info(f"🔍 BUSINESS REASONING PREVIEW: {dominant_reasoning[:100]}...")
            
            if 'risk_strategy' in board_analysis:
                rs = board_analysis['risk_strategy']
                logger.info(f"🔍 RISK STRATEGY TYPE: {type(rs)}")
                if isinstance(rs, dict):
                    logger.info(f"🔍 RISK STRATEGY KEYS: {list(rs.keys())}")
                    dominant_reasoning = rs.get('dominant_reasoning', rs.get('reasoning', ''))
                    logger.info(f"🔍 RISK REASONING LENGTH: {len(dominant_reasoning)} chars")
                    logger.info(f"🔍 RISK REASONING PREVIEW: {dominant_reasoning[:100]}...")
            # ***** END ENHANCED DEBUG LOGGING *****
            
            # FIXED: Get confidence level from the correct location in board_analysis
            # The AI analyzer puts confidence_level at the top level, not under analysis_metadata
            calculated_confidence = board_analysis.get('confidence_level', 'medium')
            
            logger.info(f"🔍 CONFIDENCE LEVEL EXTRACTION:")
            logger.info(f"   From board_analysis.confidence_level: {calculated_confidence}")
            logger.info(f"   Board analysis keys: {list(board_analysis.keys())}")
            
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
                
                # ENHANCED: Direct structured report fields with COMPLETE text preservation
                'business_strategy': board_analysis.get('business_strategy', {}),
                'risk_strategy': board_analysis.get('risk_strategy', {}),
                'swot_analysis': board_analysis.get('swot_analysis', {}),
                
                # ENHANCED: Extract and preserve archetype names for direct access
                'business_strategy_dominant': board_analysis.get('business_strategy', {}).get('dominant', 'Disciplined Specialist Growth'),
                'business_strategy_secondary': board_analysis.get('business_strategy', {}).get('secondary', 'Service-Driven Differentiator'),
                'risk_strategy_dominant': board_analysis.get('risk_strategy', {}).get('dominant', 'Risk-First Conservative'),
                'risk_strategy_secondary': board_analysis.get('risk_strategy', {}).get('secondary', 'Rules-Led Operator'),
                
                # ENHANCED: Extract and preserve COMPLETE reasoning texts for direct access
                'business_strategy_reasoning': board_analysis.get('business_strategy', {}).get('dominant_reasoning', ''),
                'risk_strategy_reasoning': board_analysis.get('risk_strategy', {}).get('dominant_reasoning', ''),
                'business_strategy_definition': board_analysis.get('business_strategy', {}).get('dominant_definition', ''),
                'risk_strategy_definition': board_analysis.get('risk_strategy', {}).get('dominant_definition', ''),
                
                'analysis_date': datetime.now().isoformat(),
                'analysis_type': board_analysis.get('analysis_type', 'board_grade_executive'),
                
                # FIXED: Use the correctly calculated confidence level from AI analyzer
                'confidence_level': calculated_confidence,  # Use the AI analyzer's calculation, not a fallback
                
                'processing_stats': {
                    'parallel_extraction_used': len(filtered_files) > 1,
                    'total_content_length': sum(len(content.get('content', '')) for content in extracted_content),
                    'extraction_methods': list(set(
                        content.get('metadata', {}).get('extraction_method', 'unknown') 
                        for content in extracted_content
                    )),
                    'board_grade_analysis': True,
                    'text_preservation_enhanced': True,  # NEW: Flag indicating enhanced text preservation
                    'confidence_calculation_source': 'ai_analyzer'  # NEW: Flag showing confidence source
                }
            }
            
            # ENHANCED: Verify the confidence level is correctly set
            logger.info(f"🎯 FINAL CONFIDENCE VERIFICATION:")
            logger.info(f"   Years analyzed: {years} (count: {len(years)})")
            logger.info(f"   Files processed: {len(extracted_content)}")
            logger.info(f"   Expected confidence: HIGH (5 years + 5 files = 100/100 points)")
            logger.info(f"   Actual confidence set: {calculated_confidence}")
            
            if calculated_confidence != 'high' and len(years) >= 5 and len(extracted_content) >= 5:
                logger.warning(f"❌ CONFIDENCE MISMATCH: Should be 'high' but got '{calculated_confidence}'")
                logger.warning(f"   This indicates the AI analyzer confidence calculation may need debugging")
            else:
                logger.info(f"✅ CONFIDENCE CORRECT: '{calculated_confidence}' matches expected level")
            
            # ENHANCED: Log the complete text lengths before storage
            logger.info(f"🔍 ENHANCED: Response data text lengths before storage:")
            logger.info(f"   business_strategy_reasoning: {len(response_data.get('business_strategy_reasoning', ''))} chars")
            logger.info(f"   risk_strategy_reasoning: {len(response_data.get('risk_strategy_reasoning', ''))} chars")
            logger.info(f"   business_strategy object reasoning: {len(response_data.get('business_strategy', {}).get('dominant_reasoning', ''))} chars")
            logger.info(f"   risk_strategy object reasoning: {len(response_data.get('risk_strategy', {}).get('dominant_reasoning', ''))} chars")
            
            # ENHANCED: Store in database with complete text preservation
            if db and components_status.get('AnalysisDatabase', {}).get('status') == 'ok':
                try:
                    logger.info(f"💾 Request {request_id}: Storing board-grade analysis with ENHANCED text preservation...")
                    record_id = store_analysis_with_enhanced_preservation(db, response_data)
                    
                    if record_id:
                        response_data['database_id'] = record_id
                        logger.info(f"✅ Request {request_id}: ENHANCED storage completed with ID: {record_id}")
                    else:
                        logger.error(f"❌ Request {request_id}: Enhanced storage failed")
                        response_data['database_warning'] = 'Analysis completed but enhanced database storage had issues'
                        
                except Exception as db_error:
                    logger.error(f"❌ Request {request_id}: Enhanced database storage failed: {str(db_error)}")
                    response_data['database_warning'] = 'Analysis completed but enhanced database storage had issues'
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
            
            logger.info(f"🎉 Request {request_id}: ENHANCED board-grade analysis completed successfully for {company_number}")
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
                'database_integration': db is not None,
                'enhanced_text_preservation': True,  # NEW: Enhanced text preservation capability
                'confidence_calculation_fixed': True  # NEW: Confidence calculation fix
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
                'version': '3.1.0-CONFIDENCE-FIXED',  # Enhanced version number
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
                },
                'enhancements': {
                    'complete_text_preservation': True,
                    'enhanced_field_mapping': True,
                    'improved_extraction_methods': True,
                    'comprehensive_debugging': True,
                    'confidence_calculation_fixed': True
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
            if not years or not isinstance(years, list):
                errors.append('Years array is required')
            elif len(years) > MAX_FILES_PER_ANALYSIS:
                errors.append(f'Too many years requested. Maximum: {MAX_FILES_PER_ANALYSIS}')
            elif len(years) == 0:
                errors.append('At least one year must be specified')
            
            # Check memory usage
            try:
                memory = psutil.virtual_memory()
                if memory.percent > MEMORY_WARNING_THRESHOLD:
                    warnings.append(f'High memory usage: {memory.percent:.1f}%. Analysis may be slower.')
            except:
                warnings.append('Could not check memory usage')
            
            # Check database availability
            if not db or components_status.get('AnalysisDatabase', {}).get('status') != 'ok':
                warnings.append('Database not available - results will not be stored')
            
            # Validate analysis context (optional)
            analysis_context = data.get('analysis_context', 'Strategic Review')
            if not isinstance(analysis_context, str) or len(analysis_context.strip()) == 0:
                warnings.append('Analysis context should be a non-empty string')
            
            # Calculate estimated processing time
            estimated_time = len(years) * 30  # Rough estimate: 30 seconds per year
            if estimated_time > 300:  # 5 minutes
                warnings.append(f'Estimated processing time: {estimated_time//60} minutes')
            
            return jsonify({
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'estimated_processing_time_seconds': estimated_time,
                'component_status': {name: info['status'] for name, info in components_status.items()},
                'system_ready': len(errors) == 0
            })
            
        except Exception as e:
            logger.error(f"Error validating request: {e}")
            return jsonify({
                'valid': False,
                'errors': [f'Validation error: {str(e)}']
            }), 500

    return app

# Helper functions that need to be defined
def download_company_filings(company_number, max_years):
    """Download company filings using the Companies House client"""
    if not ch_client:
        return None
    
    try:
        return ch_client.download_annual_accounts(company_number, max_years)
    except Exception as e:
        logger.error(f"Error downloading filings for {company_number}: {e}")
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
    """Process and analyze content for board-grade strategic analysis"""
    if not archetype_analyzer or not content_processor:
        logger.error("Required analyzers not available")
        return None
    
    try:
        # Process content using content processor - FIXED to use available methods
        logger.info("Processing content for board analysis...")
        
        # Combine all extracted content into a single text for processing
        combined_content = ""
        processed_files = []
        
        for i, content_data in enumerate(extracted_content):
            logger.info(f"📋 Processing document {i+1}: {content_data.get('filename', 'unknown')}")
            
            # Add content to combined text
            file_content = content_data.get('content', '')
            combined_content += f"\n\n=== FILE: {content_data.get('filename', 'unknown')} ===\n"
            combined_content += file_content
            
            # Track processed file info
            processed_files.append({
                'filename': content_data.get('filename', 'unknown'),
                'content_length': len(file_content),
                'date': content_data.get('date', ''),
                'metadata': content_data.get('metadata', {})
            })
        
        logger.info(f"✅ Combined content preparation completed: {len(combined_content):,} characters from {len(extracted_content)} files")
        
        # Perform board-grade AI analysis using the archetype analyzer
        logger.info("Performing board-grade AI analysis...")
        board_analysis = archetype_analyzer.analyze_for_board_optimized(
            combined_content, 
            company_name, 
            company_number,
            extracted_content,
            analysis_context
        )
        
        if not board_analysis:
            logger.error("Board-grade analysis failed")
            return None
        
        logger.info("✅ Board-grade archetype analysis completed")
        
        return {
            'board_analysis': board_analysis,
            'processed_content': {
                'combined_length': len(combined_content),
                'files_processed': processed_files,
                'total_files': len(extracted_content)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in board content analysis: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

# Create the Flask application instance
app = create_app()

if __name__ == '__main__':
    # Development server
    app.run(host='0.0.0.0', port=5000, debug=False)
else:
    # Production deployment with Gunicorn
    # The app instance will be used by Gunicorn
    pass