#!/usr/bin/env python3
"""
Flask API for Strategic Analysis Tool with Database Integration
"""

import os
import sys
import json
import logging
from datetime import datetime
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
        'version': '1.0.0',
        'endpoints': {
            'analyze': '/api/analyze',
            'years': '/api/years/<company_number>',
            'history': '/api/analysis/history/<company_number>',
            'recent': '/api/analysis/recent',
            'search': '/api/analysis/search/<term>',
            'test_db': '/api/database/test'
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

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
    """Main analysis endpoint with database storage"""
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
        
        logger.info(f"Starting analysis for company {company_number}, years: {years}")
        
        # Validate company exists
        exists, company_name = ch_client.validate_company_exists(company_number)
        if not exists:
            return jsonify({
                'success': False,
                'error': f'Company {company_number} not found'
            }), 404
        
        logger.info(f"Company validated: {company_name}")
        
        # Download filings
        logger.info("Downloading filings...")
        max_years_needed = max(datetime.now().year - min(years) + 2, 6)
        download_results = download_company_filings(company_number, max_years_needed)
        
        if not download_results or download_results['total_downloaded'] == 0:
            return jsonify({
                'success': False,
                'error': 'No annual accounts could be downloaded'
            }), 404
        
        # Filter files to selected years
        logger.info(f"Filtering files to selected years: {years}")
        filtered_files = filter_files_by_years(download_results['downloaded_files'], years)
        
        if not filtered_files:
            return jsonify({
                'success': False,
                'error': 'No files found for the selected years'
            }), 404
        
        # Extract content from PDFs
        logger.info(f"Extracting content from {len(filtered_files)} files...")
        extracted_content = extract_content_from_files(filtered_files)
        
        if not extracted_content:
            return jsonify({
                'success': False,
                'error': 'No readable content could be extracted from the files'
            }), 500
        
        # Process and analyze content
        logger.info("Processing and analyzing content...")
        analysis_results = process_and_analyze_content_api(
            extracted_content, company_name, company_number
        )
        
        if not analysis_results:
            return jsonify({
                'success': False,
                'error': 'Content analysis failed'
            }), 500
        
        # Prepare response
        response_data = {
            'success': True,
            'company_number': company_number,
            'company_name': company_name,
            'years_analyzed': years,
            'files_processed': len(extracted_content),
            'business_strategy': analysis_results['archetype_analysis'].get('business_strategy_archetypes', {}),
            'risk_strategy': analysis_results['archetype_analysis'].get('risk_strategy_archetypes', {}),
            'analysis_date': datetime.now().isoformat(),
            'analysis_type': analysis_results['archetype_analysis'].get('analysis_type', 'unknown')
        }
        
        # Store in database
        try:
            record_id = db.store_analysis_result(response_data)
            response_data['database_id'] = record_id
            logger.info(f"Analysis stored in database with ID: {record_id}")
        except Exception as db_error:
            logger.error(f"Database storage failed: {db_error}")
            # Continue without failing the whole request
        
        # Clean up temporary files
        try:
            ch_client.cleanup_temp_files()
        except:
            pass
        
        logger.info(f"Analysis completed successfully for {company_number}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def download_company_filings(company_number, max_years):
    """Download company filings using existing method"""
    try:
        results = ch_client.download_annual_accounts(company_number, max_years)
        logger.info(f"Downloaded {results['total_downloaded']} files for {company_number}")
        return results
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
                    logger.info(f"Included file: {file_info['filename']} (Year {file_year})")
                    
            except Exception as e:
                logger.warning(f"Could not determine year for {file_info['filename']}: {e}")
    
    return filtered_files

def extract_content_from_files(downloaded_files):
    """Extract content from downloaded files"""
    extracted_content = []
    
    for file_info in downloaded_files:
        try:
            logger.info(f"Extracting content from: {file_info['filename']}")
            
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
                    logger.info(f"Successfully extracted {len(content)} characters")
                else:
                    logger.warning(f"Insufficient content extracted from {file_info['filename']}")
            else:
                logger.error(f"Extraction failed for {file_info['filename']}")
                
        except Exception as e:
            logger.error(f"Error extracting content from {file_info['filename']}: {e}")
    
    return extracted_content

def process_and_analyze_content_api(extracted_content, company_name, company_number):
    """Process and analyze content for API"""
    try:
        # Process documents
        processed_documents = []
        for content_data in extracted_content:
            processed = content_processor.process_document_content(
                content_data['content'], content_data['metadata']
            )
            processed_documents.append(processed)
        
        # Combine documents
        combined_analysis = content_processor.combine_multiple_documents(processed_documents)
        
        # Perform archetype analysis
        combined_content = "\n\n".join([content_data['content'] for content_data in extracted_content])
        
        archetype_analysis = archetype_analyzer.analyze_archetypes(
            combined_content, company_name, company_number
        )
        
        return {
            'processed_content': combined_analysis,
            'archetype_analysis': archetype_analysis,
            'document_count': len(extracted_content)
        }
        
    except Exception as e:
        logger.error(f"Error processing content: {e}")
        return None

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

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)