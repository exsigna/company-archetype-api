#!/usr/bin/env python3
"""
Flask API for Strategic Analysis Tool with Database Integration
Enhanced with Full Multi-File Analysis Support
"""

import os
import sys
import json
import logging
import uuid
from datetime import datetime, date
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import your existing modules
try:
    from config import validate_config, validate_company_number
    from companies_house_client import CompaniesHouseClient
    from content_processor import ContentProcessor
    from pdf_extractor import PDFExtractor
    from ai_analyzer import AIArchetypeAnalyzer
    from file_manager import FileManager
    from report_generator import ReportGenerator
    from database import AnalysisDatabase
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

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

# Initialize components (with error handling)
try:
    ch_client = CompaniesHouseClient()
    content_processor = ContentProcessor()
    pdf_extractor = PDFExtractor()
    archetype_analyzer = AIArchetypeAnalyzer()
    file_manager = FileManager()
    report_generator = ReportGenerator()
    db = AnalysisDatabase()
    logger.info("All components initialized successfully")
except Exception as e:
    logger.error(f"Error initializing components: {e}")
    # Don't exit - let the app start so we can see the error

# Flag to track if initialization has been done
_app_initialized = False

@app.before_request
def initialize_app():
    """Initialize app and test database connection"""
    global _app_initialized
    if not _app_initialized:
        _app_initialized = True
        
        logger.info("Initializing Strategic Analysis API...")
        
        try:
            # Validate configuration
            if not validate_config():
                logger.error("Configuration validation failed")
                return
            
            # Test database connection
            success = db.test_connection()
            if success:
                logger.info("✅ Database connected successfully")
            else:
                logger.error("❌ Database connection failed")
            
            logger.info("✅ Strategic Analysis API initialized")
        except Exception as e:
            logger.error(f"Error during initialization: {e}")

@app.route('/')
def home():
    """Basic home page"""
    return jsonify({
        'service': 'Strategic Analysis API',
        'status': 'running',
        'version': '2.0.0',  # Updated version for multi-file support
        'features': [
            'Multi-file analysis with individual file processing',
            'Enhanced AI archetype classification',
            'Intelligent content sampling (15K chars)',
            'File-by-file synthesis and confidence scoring',
            'Comprehensive evidence-based reasoning'
        ],
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
            'database_stats': '/api/database/stats'
        },
        'usage': {
            'lookup_company': 'GET /api/company/lookup/Marine - Check previous analyses',
            'check_company': 'POST /api/company/check {"company_identifier": "Marine"}',
            'preview_cleanup': 'GET /api/database/preview-cleanup/02613335 - Preview what would be deleted',
            'cleanup_analysis': 'DELETE /api/database/cleanup/02613335/1 - Delete specific analysis',
            'cleanup_invalid': 'DELETE /api/database/cleanup/invalid/02613335 - Clean invalid analyses (USE WITH CAUTION)'
        },
        'safety_note': 'Always use preview-cleanup before cleanup-invalid to see what will be deleted'
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/company/lookup/<company_identifier>')
def lookup_company_analysis(company_identifier):
    """
    Look up previous analyses for a company by name or number
    
    Args:
        company_identifier: Company name or company number
    
    Returns:
        JSON with previous analysis metadata or empty if none found
    """
    try:
        logger.info(f"Looking up previous analyses for: {company_identifier}")
        
        # First, try to find by company number (exact match)
        if validate_company_number(company_identifier):
            # It's a valid company number format
            results = db.get_analysis_by_company(company_identifier)
            if results:
                # Format the response
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
            # Get detailed analysis for the first matching company
            first_match = search_results[0]
            detailed_analyses = db.get_analysis_by_company(first_match['company_number'])
            
            # Format the response
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
        
        # No previous analyses found
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
    """
    Check if a company has previous analyses before running new analysis
    Accepts both company name and company number
    
    Request body:
    {
        "company_identifier": "Marine and General" or "00000006"
    }
    """
    try:
        data = request.get_json()
        if not data or not data.get('company_identifier'):
            return jsonify({
                'success': False,
                'error': 'company_identifier is required'
            }), 400
        
        company_identifier = data.get('company_identifier', '').strip()
        
        # Use the lookup function
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
    try:
        # Validate company number
        if not validate_company_number(company_number):
            return jsonify({
                'success': False,
                'error': 'Invalid company number format'
            }), 400
        
        logger.info(f"Getting available years for company {company_number}")
        
        # Get filing history
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
    """Enhanced main analysis endpoint with full multi-file support and database storage"""
    
    # Generate unique request ID to track this analysis
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"🆔 Analysis request {request_id} started")
    
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
        
        logger.info(f"🚀 Request {request_id}: Starting ENHANCED multi-file analysis for company {company_number}, years: {years}")
        
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
        
        # Extract content from PDFs
        logger.info(f"📄 Extracting content from {len(filtered_files)} files...")
        extracted_content = extract_content_from_files(filtered_files)
        
        if not extracted_content:
            return jsonify({
                'success': False,
                'error': 'No readable content could be extracted from the files'
            }), 500
        
        logger.info(f"✅ Successfully extracted content from {len(extracted_content)} files")
        
        # Enhanced content processing and analysis
        logger.info("🧠 Processing and analyzing content with ENHANCED multi-file support...")
        analysis_results = process_and_analyze_content_api_enhanced(
            extracted_content, company_name, company_number
        )
        
        if not analysis_results:
            return jsonify({
                'success': False,
                'error': 'Content analysis failed'
            }), 500
        
        # Enhanced response preparation with additional metadata
        archetype_analysis = analysis_results.get('archetype_analysis', {})
        analysis_metadata = archetype_analysis.get('analysis_metadata', {})
        
        response_data = {
            'success': True,
            'company_number': company_number,
            'company_name': company_name,
            'years_analyzed': years,
            'files_processed': len(extracted_content),
            'business_strategy': make_json_serializable(archetype_analysis.get('business_strategy_archetypes', {})),
            'risk_strategy': make_json_serializable(archetype_analysis.get('risk_strategy_archetypes', {})),
            'analysis_date': datetime.now().isoformat(),
            'analysis_type': archetype_analysis.get('analysis_type', 'unknown'),
            'analysis_metadata': {
                'files_analyzed': analysis_metadata.get('files_analyzed', len(extracted_content)),
                'total_content_chars': analysis_metadata.get('total_content_chars', 0),
                'confidence_level': analysis_metadata.get('confidence_level', 'medium'),
                'content_utilization': analysis_metadata.get('content_utilization', 'multi_file'),
                'model_used': archetype_analysis.get('model_used', 'enhanced_analyzer')
            },
            'file_details': [
                {
                    'filename': content['filename'],
                    'date': make_json_serializable(content['date']),  # Ensure date is serializable
                    'content_length': len(content['content']),
                    'extraction_method': content['metadata'].get('extraction_method', 'unknown')
                }
                for content in extracted_content
            ]
        }
        
        # Store in database with enhanced error handling (prevent retry loops)
        try:
            logger.info(f"💾 Request {request_id}: Attempting to store analysis results in database...")
            
            # Ensure all data is JSON serializable before storing
            serializable_response = make_json_serializable(response_data)
            
            record_id = db.store_analysis_result(serializable_response)
            response_data['database_id'] = record_id
            logger.info(f"✅ Request {request_id}: Analysis stored in database with ID: {record_id}")
            
        except Exception as db_error:
            logger.error(f"❌ Request {request_id}: Database storage failed: {str(db_error)}")
            logger.error(f"🔍 Request {request_id}: Error type: {type(db_error).__name__}")
            
            # Add database warning but DON'T retry to avoid multiple errors
            response_data['database_warning'] = f'Analysis completed but database storage failed: {str(db_error)}'
            logger.warning(f"⚠️ Request {request_id}: Continuing without database storage to prevent retry loops")
        
        # Clean up temporary files
        try:
            ch_client.cleanup_temp_files()
            logger.info("🧹 Temporary files cleaned up")
        except Exception as cleanup_error:
            logger.warning(f"⚠️ Cleanup warning: {cleanup_error}")
        
        logger.info(f"🎉 Request {request_id}: Enhanced multi-file analysis completed successfully for {company_number}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ Request {request_id}: Analysis failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'request_id': request_id
        }), 500

def download_company_filings(company_number, max_years):
    """Download company filings using existing method"""
    try:
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
    """Extract content from downloaded files with enhanced logging"""
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
                if content and len(content.strip()) > 100:  # Minimum content check
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

def process_and_analyze_content_api_enhanced(extracted_content, company_name, company_number):
    """
    ENHANCED process and analyze content for API with full multi-file support
    
    This is the key enhancement that enables individual file analysis and synthesis
    """
    try:
        logger.info(f"🧠 Starting ENHANCED content processing for {len(extracted_content)} files")
        
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
        
        # Prepare combined content (for backward compatibility)
        combined_content = "\n\n".join([content_data['content'] for content_data in extracted_content])
        logger.info(f"📊 Combined content length: {len(combined_content):,} characters")
        
        # *** ENHANCED ARCHETYPE ANALYSIS WITH INDIVIDUAL FILE DATA ***
        logger.info("🚀 Performing ENHANCED archetype analysis with individual file support")
        
        # Ensure extracted_content dates are serializable before passing to analyzer
        serializable_extracted_content = []
        for content_data in extracted_content:
            serializable_content = {
                'filename': content_data['filename'],
                'date': make_json_serializable(content_data['date']),  # Fix date serialization
                'content': content_data['content'],
                'metadata': content_data['metadata']
            }
            serializable_extracted_content.append(serializable_content)
        
        archetype_analysis = archetype_analyzer.analyze_archetypes(
            combined_content,           # Combined content for compatibility
            company_name, 
            company_number,
            extracted_content=serializable_extracted_content  # Pass serializable file data
        )
        
        logger.info("✅ Enhanced archetype analysis completed")
        
        return {
            'processed_content': combined_analysis,
            'archetype_analysis': archetype_analysis,
            'document_count': len(extracted_content),
            'enhancement_status': 'multi_file_analysis_enabled'
        }
        
    except Exception as e:
        logger.error(f"❌ Error in enhanced content processing: {e}")
        return None

# Legacy function for backward compatibility
def process_and_analyze_content_api(extracted_content, company_name, company_number):
    """Legacy process and analyze content for API (calls enhanced version)"""
    logger.info("🔄 Using enhanced processing pipeline for backward compatibility")
    return process_and_analyze_content_api_enhanced(extracted_content, company_name, company_number)

# Database endpoints
@app.route('/api/analysis/history/<company_number>')
def get_company_history(company_number):
    """Get analysis history for a company"""
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
    try:
        success = db.test_connection()
        return jsonify({
            'success': success,
            'message': 'Database connection successful' if success else 'Database connection failed'
        })
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# SAFE DATABASE CLEANUP ENDPOINTS
@app.route('/api/database/preview-cleanup/<company_number>')
def preview_cleanup(company_number):
    """Preview what would be deleted without actually deleting - SAFETY FIRST"""
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
    try:
        logger.info(f"Attempting to delete analysis ID {analysis_id} for company {company_number}")
        
        # Call database method to delete specific analysis
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
    try:
        logger.warning(f"⚠️  DANGEROUS OPERATION: Cleaning up invalid analyses for company {company_number}")
        
        # Call database method to cleanup invalid analyses (now much more conservative)
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
    try:
        stats = db.get_analysis_statistics()
        return jsonify({
            'success': True,
            'statistics': stats
        })
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# New endpoint for analysis summary and insights
@app.route('/api/analysis/summary/<company_number>/<int:analysis_id>')
def get_analysis_summary(company_number, analysis_id):
    """Get detailed summary of a specific analysis"""
    try:
        # Get the specific analysis from database
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
        
        # Generate summary using the AI analyzer
        if hasattr(archetype_analyzer, 'get_analysis_summary'):
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

# New endpoint for multi-file analysis comparison
@app.route('/api/analysis/compare/<company_number>')
def compare_analyses(company_number):
    """Compare multiple analyses for the same company to show evolution over time"""
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

# Enhanced endpoint for checking system status
@app.route('/api/system/status')
def system_status():
    """Get comprehensive system status including multi-file analysis capabilities"""
    try:
        status = {
            'service': 'Strategic Analysis API',
            'version': '2.0.0',
            'status': 'operational',
            'timestamp': datetime.now().isoformat(),
            'capabilities': {
                'multi_file_analysis': True,
                'ai_archetype_classification': archetype_analyzer.client_type == 'openai',
                'enhanced_content_sampling': True,
                'individual_file_processing': True,
                'synthesis_and_confidence_scoring': True,
                'database_integration': True
            },
            'components': {
                'companies_house_client': 'operational',
                'pdf_extractor': 'operational',
                'content_processor': 'operational',
                'ai_analyzer': archetype_analyzer.client_type,
                'database': 'operational' if db.test_connection() else 'error',
                'file_manager': 'operational',
                'report_generator': 'operational'
            },
            'analysis_features': {
                'content_sample_size': '15,000 characters (enhanced)',
                'archetype_categories': {
                    'business_strategy': len(archetype_analyzer.business_archetypes),
                    'risk_strategy': len(archetype_analyzer.risk_archetypes)
                },
                'file_formats_supported': ['PDF (text)', 'PDF (OCR)', 'PDF (hybrid)'],
                'years_supported': 'Configurable (default: last 6 years)',
                'confidence_scoring': 'Multi-file synthesis based'
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

# Utility endpoint for validating requests before processing
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
        elif len(years) > 10:
            warnings.append('More than 10 years selected - this may take a long time')
        elif len(years) == 1:
            warnings.append('Only 1 year selected - multi-file analysis benefits require multiple years')
        
        # Check if company exists (if no critical errors so far)
        company_exists = False
        company_name = 'Unknown'
        if not errors and company_number:
            try:
                company_exists, company_name = ch_client.validate_company_exists(company_number)
                if not company_exists:
                    errors.append(f'Company {company_number} not found')
            except Exception as e:
                warnings.append(f'Could not verify company existence: {str(e)}')
        
        # Check for previous analyses
        previous_analyses = []
        if company_exists:
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
                'estimated_files': len(years),  # Rough estimate
                'previous_analyses_count': len(previous_analyses)
            },
            'system_capabilities': {
                'ai_analysis_available': archetype_analyzer.client_type == 'openai',
                'multi_file_support': True,
                'max_recommended_years': 10
            }
        }
        
        return jsonify(validation_result)
        
    except Exception as e:
        logger.error(f"Error validating request: {e}")
        return jsonify({
            'valid': False,
            'errors': [f'Validation error: {str(e)}']
        }), 500

if __name__ == '__main__':
    # Enhanced startup logging
    logger.info("=" * 60)
    logger.info("🚀 STRATEGIC ANALYSIS API v2.0.0 - ENHANCED MULTI-FILE")
    logger.info("=" * 60)
    logger.info("🔧 Capabilities:")
    logger.info("   ✅ Multi-file individual analysis and synthesis")
    logger.info("   ✅ Enhanced content sampling (15K chars)")
    logger.info("   ✅ AI-powered archetype classification")
    logger.info("   ✅ Confidence scoring and evidence tracking")
    logger.info("   ✅ Database integration with analysis history")
    logger.info("=" * 60)
    
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)