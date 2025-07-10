# Update your main.py to use deterministic analysis

# Replace the existing ExecutiveAIAnalyzer import
from ai_analyzer import DeterministicAIAnalyzer

# Update the initialization in main.py
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

# Update component initialization
archetype_analyzer = safe_init_component('DeterministicAIAnalyzer', DeterministicAIAnalyzer)

# Update the analysis function call in main.py
def process_and_analyze_content_for_board(extracted_content, company_name, company_number, analysis_context):
    """Process and analyze content with deterministic results"""
    try:
        if not content_processor or not archetype_analyzer:
            logger.error("Content processor or deterministic analyzer not available")
            return None
        
        logger.info(f"🔒 Starting deterministic board-grade content processing for {len(extracted_content)} files")
        
        # Process documents individually (same as before)
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
        
        # DETERMINISTIC archetype analysis
        logger.info("🔒 Performing deterministic board-grade archetype analysis")
        
        # Ensure extracted_content dates are serializable
        serializable_extracted_content = []
        for content_data in extracted_content:
            serializable_content = {
                'filename': content_data['filename'],
                'date': make_json_serializable(content_data['date']),
                'content': content_data['content'],
                'metadata': content_data['metadata']
            }
            serializable_extracted_content.append(serializable_content)
        
        # Use the deterministic analysis method
        board_analysis = archetype_analyzer.analyze_for_board_deterministic(
            combined_content,
            company_name, 
            company_number,
            extracted_content=serializable_extracted_content,
            analysis_context=analysis_context
        )
        
        logger.info("✅ Deterministic board-grade archetype analysis completed")
        
        return {
            'processed_content': combined_analysis,
            'board_analysis': board_analysis,
            'document_count': len(extracted_content)
        }
        
    except Exception as e:
        logger.error(f"❌ Error in deterministic board-grade content processing: {e}")
        return None

# Environment variables for deterministic results
import os

# Add these to your environment setup
os.environ.setdefault('ENABLE_DETERMINISTIC_ANALYSIS', 'true')
os.environ.setdefault('CACHE_ANALYSIS_RESULTS', 'true')
os.environ.setdefault('AI_TEMPERATURE', '0.0')  # Zero temperature for consistency

# Update your startup logging
def initialize_app():
    """Initialize app with deterministic settings"""
    logger.info("🔒 Initializing Strategic Analysis API with DETERMINISTIC RESULTS...")
    
    try:
        # Validate configuration if available
        if 'config' not in [m.split(':')[0] for m in missing_modules]:
            if not validate_config():
                logger.error("❌ Configuration validation failed")
                return
            else:
                logger.info("✅ Configuration validated")
        
        # Log deterministic settings
        logger.info("🔒 Deterministic Analysis Settings:")
        logger.info(f"   - Temperature: {os.environ.get('AI_TEMPERATURE', '0.0')}")
        logger.info(f"   - Caching: {os.environ.get('CACHE_ANALYSIS_RESULTS', 'true')}")
        logger.info(f"   - Content Hashing: Enabled")
        logger.info(f"   - Result Validation: Enabled")
        
        # Test database connection if available
        if db and components_status.get('AnalysisDatabase', {}).get('status') == 'ok':
            success = db.test_connection()
            if success:
                logger.info("✅ Database connected successfully")
            else:
                logger.error("❌ Database connection failed")
        
        logger.info("✅ Strategic Analysis API initialized with deterministic results")
    except Exception as e:
        logger.error(f"❌ Error during deterministic initialization: {e}")

# Update your home endpoint to reflect deterministic features
@app.route('/')
def home():
    """Enhanced home page with deterministic analysis features"""
    component_statuses = {}
    for name, info in components_status.items():
        component_statuses[name] = info['status']
    
    return jsonify({
        'service': 'Strategic Analysis API',
        'status': 'running',
        'version': '6.0.0',  # Updated version for deterministic analysis
        'analysis_mode': 'deterministic',
        'component_status': component_statuses,
        'features': [
            '🔒 Deterministic analysis with identical results for identical inputs',
            '💾 Content hashing for consistency tracking',
            '🎯 Zero temperature AI settings for reproducible results',
            '✅ Result validation and normalization',
            '📊 Structured archetype classification with priority ordering',
            'Board-grade strategic analysis with executive insights',
            'Multi-file analysis with individual file processing',
            'Strategic risk assessment and governance evaluation',
            'Comprehensive evidence-based reasoning',
            'Parallel PDF processing with optimization'
        ],
        'deterministic_features': {
            'content_hashing': True,
            'result_caching': True,
            'zero_temperature': True,
            'structured_prompting': True,
            'archetype_validation': True,
            'word_count_enforcement': True
        },
        'consistency_guarantee': 'Identical inputs will produce identical archetype classifications and analysis results',
        'endpoints': {
            'analyze': '/api/analyze',
            'years': '/api/years/<company_number>',
            'lookup_company': '/api/company/lookup/<company_name_or_number>',
            'check_company': '/api/company/check',
            'history': '/api/analysis/history/<company_number>',
            'recent': '/api/analysis/recent',
            'validate_request': '/api/validate/request'
        }
    })