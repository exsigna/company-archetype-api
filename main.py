#!/usr/bin/env python3
"""
Flask API for Strategic Analysis Tool with Database Integration
Fixed for Flask 2.3+ compatibility and Gunicorn deployment
Updated for ExecutiveAIAnalyzer (Board-Grade Analysis)
Enhanced Lookup API to extract and return archetype names
DEBUG VERSION - Enhanced logging for database lookup issues
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

def extract_archetype_from_analysis(result):
    """Extract business and risk archetype names from analysis result"""
    business_strategy = 'Not Available'
    risk_strategy = 'Not Available'
    
    try:
        # Try to get from direct fields first (most common case)
        business_strategy = result.get('business_strategy_dominant') or 'Not Available'
        risk_strategy = result.get('risk_strategy_dominant') or 'Not Available'
        
        # If we got valid values that aren't 'Not Available', return them
        if business_strategy != 'Not Available' and risk_strategy != 'Not Available':
            return business_strategy, risk_strategy
        
        # If not available in direct fields, try to extract from raw_response
        raw_response = result.get('raw_response')
        if raw_response:
            # Parse raw_response if it's a string
            if isinstance(raw_response, str):
                try:
                    raw_data = json.loads(raw_response)
                except json.JSONDecodeError:
                    raw_data = {}
            else:
                raw_data = raw_response or {}
            
            # Extract business strategy archetype with multiple fallback paths
            if business_strategy == 'Not Available':
                business_strategy = (
                    raw_data.get('business_strategy_analysis', {}).get('dominant_archetype') or
                    raw_data.get('business_strategy_analysis', {}).get('dominant') or
                    raw_data.get('business_strategy', {}).get('dominant_archetype') or
                    raw_data.get('business_strategy', {}).get('dominant') or
                    raw_data.get('business_strategy_dominant') or
                    raw_data.get('board_analysis', {}).get('business_strategy_analysis', {}).get('dominant_archetype') or
                    'Not Available'
                )
            
            # Extract risk strategy archetype with multiple fallback paths
            if risk_strategy == 'Not Available':
                risk_strategy = (
                    raw_data.get('risk_strategy_analysis', {}).get('dominant_archetype') or
                    raw_data.get('risk_strategy_analysis', {}).get('dominant') or
                    raw_data.get('risk_strategy', {}).get('dominant_archetype') or
                    raw_data.get('risk_strategy', {}).get('dominant') or
                    raw_data.get('risk_strategy_dominant') or
                    raw_data.get('board_analysis', {}).get('risk_strategy_analysis', {}).get('dominant_archetype') or
                    'Not Available'
                )
    
    except Exception as e:
        logger.warning(f"Error extracting archetypes from analysis {result.get('id', 'unknown')}: {e}")
        # Keep the default 'Not Available' values
    
    return business_strategy, risk_strategy

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
            'version': '3.0.0-DEBUG',  # Updated version for board-grade analysis
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
                'Memory usage monitoring and optimization',
                'DEBUG: Enhanced database lookup logging'
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
            'debug_note': 'This is a DEBUG version with enhanced database lookup logging'
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
        """DEBUG VERSION - Look up previous analyses with enhanced logging"""
        logger.info(f"🔍 DEBUG: Lookup request received for: '{company_identifier}'")
        logger.info(f"🔍 DEBUG: Request type: {type(company_identifier)}")
        logger.info(f"🔍 DEBUG: Request length: {len(company_identifier)}")
        
        # Check database availability with detailed logging
        if not db:
            logger.error(f"🔍 DEBUG: Database object is None")
            return jsonify({
                'success': False,
                'error': 'Database object not initialized',
                'debug_info': {
                    'db_object': str(db),
                    'components_status': components_status
                }
            }), 503
        
        db_status = components_status.get('AnalysisDatabase', {}).get('status')
        logger.info(f"🔍 DEBUG: Database status: {db_status}")
        
        if db_status != 'ok':
            logger.error(f"🔍 DEBUG: Database not available - status: {db_status}")
            return jsonify({
                'success': False,
                'error': 'Database not available',
                'debug_info': {
                    'db_status': db_status,
                    'components_status': components_status.get('AnalysisDatabase', {})
                }
            }), 503
        
        try:
            logger.info(f"🔍 DEBUG: Starting lookup process for: {company_identifier}")
            
            # Test database connection first
            try:
                db_test = db.test_connection()
                logger.info(f"🔍 DEBUG: Database connection test: {db_test}")
            except Exception as db_test_error:
                logger.error(f"🔍 DEBUG: Database connection test failed: {db_test_error}")
            
            # Validate company number with detailed logging
            is_valid_company_number = validate_company_number(company_identifier)
            logger.info(f"🔍 DEBUG: Company number validation result: {is_valid_company_number}")
            
            if is_valid_company_number:
                logger.info(f"🔍 DEBUG: Processing as company number: {company_identifier}")
                
                # Try database query with extensive logging
                try:
                    logger.info(f"🔍 DEBUG: About to call db.get_analysis_by_company('{company_identifier}')")
                    logger.info(f"🔍 DEBUG: Database object type: {type(db)}")
                    logger.info(f"🔍 DEBUG: Database object methods: {[method for method in dir(db) if not method.startswith('_')]}")
                    
                    results = db.get_analysis_by_company(company_identifier)
                    
                    logger.info(f"🔍 DEBUG: Database query completed")
                    logger.info(f"🔍 DEBUG: Results type: {type(results)}")
                    logger.info(f"🔍 DEBUG: Results: {results}")
                    logger.info(f"🔍 DEBUG: Number of results: {len(results) if results else 0}")
                    
                    if results:
                        logger.info(f"🔍 DEBUG: First result type: {type(results[0])}")
                        logger.info(f"🔍 DEBUG: First result keys: {list(results[0].keys()) if hasattr(results[0], 'keys') else 'No keys method'}")
                        logger.info(f"🔍 DEBUG: First result sample: {dict(results[0]) if hasattr(results[0], 'keys') else results[0]}")
                        
                        # Process results normally
                        analysis_metadata = []
                        for result in results:
                            try:
                                business_strategy, risk_strategy = extract_archetype_from_analysis(result)
                                logger.info(f"🔍 DEBUG: Extracted archetypes - Business: {business_strategy}, Risk: {risk_strategy}")
                            except Exception as e:
                                logger.error(f"🔍 DEBUG: Error extracting archetypes for analysis {result.get('id')}: {e}")
                                business_strategy = result.get('business_strategy_dominant', 'Not Available')
                                risk_strategy = result.get('risk_strategy_dominant', 'Not Available')
                            
                            metadata = {
                                'analysis_id': result.get('id'),
                                'company_number': result.get('company_number'),
                                'company_name': result.get('company_name'),
                                'analysis_date': result.get('analysis_date'),
                                'years_analyzed': result.get('years_analyzed', []),
                                'files_processed': result.get('files_processed', 0),
                                'business_strategy': business_strategy,
                                'business_strategy_dominant': business_strategy,
                                'risk_strategy': risk_strategy,
                                'risk_strategy_dominant': risk_strategy,
                                'status': result.get('status'),
                                'analysis_type': result.get('analysis_type', 'unknown'),
                                'confidence_level': result.get('confidence_level', 'medium')
                            }
                            analysis_metadata.append(metadata)
                        
                        logger.info(f"🔍 DEBUG: Successfully processed {len(analysis_metadata)} analyses")
                        
                        return jsonify({
                            'success': True,
                            'found': True,
                            'search_term': company_identifier,
                            'search_type': 'company_number',
                            'company_number': results[0].get('company_number'),
                            'company_name': results[0].get('company_name'),
                            'total_analyses': len(results),
                            'analyses': analysis_metadata,
                            'debug_info': {
                                'raw_results_count': len(results),
                                'processed_count': len(analysis_metadata)
                            }
                        })
                    else:
                        logger.info(f"🔍 DEBUG: No results found for company number {company_identifier}")
                        
                except Exception as query_error:
                    logger.error(f"🔍 DEBUG: Database query failed with error: {query_error}")
                    logger.error(f"🔍 DEBUG: Error type: {type(query_error)}")
                    return jsonify({
                        'success': False,
                        'error': f'Database query failed: {str(query_error)}',
                        'debug_info': {
                            'query_error': str(query_error),
                            'error_type': str(type(query_error)),
                            'company_identifier': company_identifier
                        }
                    }), 500
            
            # If not a company number or no results, search by company name
            logger.info(f"🔍 DEBUG: Attempting company name search for: {company_identifier}")
            try:
                search_results = db.search_companies(company_identifier)
                logger.info(f"🔍 DEBUG: Company name search returned: {len(search_results) if search_results else 0} results")
                
                if search_results:
                    first_match = search_results[0]
                    logger.info(f"🔍 DEBUG: First search match: {first_match}")
                    
                    detailed_analyses = db.get_analysis_by_company(first_match['company_number'])
                    logger.info(f"🔍 DEBUG: Detailed analyses for {first_match['company_number']}: {len(detailed_analyses) if detailed_analyses else 0}")
                    
                    # Process detailed analyses...
                    analysis_metadata = []
                    for result in detailed_analyses:
                        try:
                            business_strategy, risk_strategy = extract_archetype_from_analysis(result)
                        except Exception as e:
                            logger.error(f"🔍 DEBUG: Error extracting archetypes for analysis {result.get('id')}: {e}")
                            business_strategy = result.get('business_strategy_dominant', 'Not Available')
                            risk_strategy = result.get('risk_strategy_dominant', 'Not Available')
                        
                        metadata = {
                            'analysis_id': result.get('id'),
                            'company_number': result.get('company_number'),
                            'company_name': result.get('company_name'),
                            'analysis_date': result.get('analysis_date'),
                            'years_analyzed': result.get('years_analyzed', []),
                            'files_processed': result.get('files_processed', 0),
                            'business_strategy': business_strategy,
                            'business_strategy_dominant': business_strategy,
                            'risk_strategy': risk_strategy,
                            'risk_strategy_dominant': risk_strategy,
                            'status': result.get('status'),
                            'analysis_type': result.get('analysis_type', 'unknown'),
                            'confidence_level': result.get('confidence_level', 'medium')
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
            except Exception as search_error:
                logger.error(f"🔍 DEBUG: Company name search failed: {search_error}")
            
            # No results found
            logger.info(f"🔍 DEBUG: No analyses found for {company_identifier}")
            return jsonify({
                'success': True,
                'found': False,
                'search_term': company_identifier,
                'message': 'No previous analyses found for this company',
                'suggestion': 'Use /api/analyze to create a new analysis',
                'debug_info': {
                    'company_identifier': company_identifier,
                    'validation_result': is_valid_company_number,
                    'search_attempted': True
                }
            })
            
        except Exception as e:
            logger.error(f"🔍 DEBUG: Lookup failed with error: {e}")
            logger.error(f"🔍 DEBUG: Error type: {type(e)}")
            return jsonify({
                'success': False,
                'error': str(e),
                'debug_info': {
                    'error_type': str(type(e)),
                    'company_identifier': company_identifier,
                    'db_available': db is not None,
                    'db_status': components_status.get('AnalysisDatabase', {})
                }
            }), 500

    # Include all other endpoints from the original file...
    # (For brevity, I'm including just the essential ones for debugging)

    @app.route('/api/database/test')
    def test_database_endpoint():
        """Test database connection with enhanced debugging"""
        logger.info("🔍 DEBUG: Database test endpoint called")
        
        if not db:
            logger.error("🔍 DEBUG: Database object is None")
            return jsonify({
                'success': False,
                'message': 'Database component not initialized',
                'debug_info': {
                    'db_object': str(db),
                    'components_status': components_status
                }
            }), 503
        
        try:
            logger.info(f"🔍 DEBUG: Testing database connection...")
            success = db.test_connection()
            logger.info(f"🔍 DEBUG: Database test result: {success}")
            
            return jsonify({
                'success': success,
                'message': 'Database connection successful' if success else 'Database connection failed',
                'debug_info': {
                    'db_object_type': str(type(db)),
                    'test_result': success
                }
            })
        except Exception as e:
            logger.error(f"🔍 DEBUG: Database test failed: {e}")
            return jsonify({
                'success': False, 
                'error': str(e),
                'debug_info': {
                    'error_type': str(type(e)),
                    'db_object': str(db)
                }
            }), 500

    @app.route('/api/analysis/recent')
    def get_recent():
        """Get recent analyses with debug logging"""
        logger.info("🔍 DEBUG: Recent analyses endpoint called")
        
        if not db or components_status.get('AnalysisDatabase', {}).get('status') != 'ok':
            return jsonify({
                'success': False,
                'error': 'Database not available',
                'debug_info': {
                    'db_available': db is not None,
                    'db_status': components_status.get('AnalysisDatabase', {})
                }
            }), 503
        
        try:
            logger.info("🔍 DEBUG: Calling db.get_recent_analyses()")
            results = db.get_recent_analyses()
            logger.info(f"🔍 DEBUG: Recent analyses query returned {len(results) if results else 0} results")
            
            return jsonify({
                'success': True,
                'results': results,
                'count': len(results),
                'debug_info': {
                    'results_type': str(type(results)),
                    'first_result_sample': dict(results[0]) if results and hasattr(results[0], 'keys') else None
                }
            })
        except Exception as e:
            logger.error(f"🔍 DEBUG: Error getting recent analyses: {e}")
            return jsonify({
                'success': False, 
                'error': str(e),
                'debug_info': {
                    'error_type': str(type(e))
                }
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

# Create the Flask app
app = create_app()

if __name__ == '__main__':
    # Enhanced startup logging
    logger.info("=" * 70)
    logger.info("🚀 STRATEGIC ANALYSIS API v3.0.0-DEBUG - BOARD-GRADE ANALYSIS")
    logger.info("=" * 70)
    logger.info("🔧 DEBUG Features:")
    logger.info("   ✅ Enhanced database lookup logging")
    logger.info("   ✅ Detailed error reporting")
    logger.info("   ✅ Component status debugging")
    logger.info("   ✅ Database query tracing")
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
    
    logger.info(f"🌐 Starting DEBUG server on port {port} (debug: {debug_mode})")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)