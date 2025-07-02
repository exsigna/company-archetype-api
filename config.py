#!/usr/bin/env python3
"""
Configuration settings for the Company Archetype Analysis Tool
"""

import os
import re
import logging
from pathlib import Path
from typing import List

# Get logger
logger = logging.getLogger(__name__)

# ===== ENVIRONMENT VARIABLES =====
# Companies House API Configuration
COMPANIES_HOUSE_API_KEY = os.getenv('CH_API_KEY', '')
CH_BASE_URL = "https://api.company-information.service.gov.uk"

# AI API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')

# Default AI Models
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-3-haiku-20240307"

# AI Parameters
AI_MAX_TOKENS = 2000
AI_TEMPERATURE = 0.3

# ===== FILE AND DIRECTORY SETTINGS =====
# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TEMP_FOLDER = BASE_DIR / "temp"
OUTPUT_DIR = BASE_DIR / "output"

# Ensure directories exist
for directory in [DATA_DIR, TEMP_FOLDER, OUTPUT_DIR]:
    directory.mkdir(exist_ok=True)

# ===== API SETTINGS =====
REQUEST_TIMEOUT = 30  # seconds
DEFAULT_MAX_YEARS = 5

# ===== CONTENT PROCESSING SETTINGS =====
# Minimum content length for processing
MIN_CONTENT_LENGTH = 50
MIN_EXTRACTION_LENGTH = 200

# Token limits for AI processing
MAX_TOKENS_PER_CHUNK = 4000
TOTAL_TOKEN_LIMIT = 16000
MAX_CONTENT_SECTIONS_PER_CATEGORY = 10

# File validation
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_FILE_TYPES = ['.pdf', '.txt', '.docx']

# ===== PDF EXTRACTION SETTINGS =====
# OCR settings
MAX_PAGES_OCR = 20
OCR_DPI = 300
OCR_CONFIG = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()£$%+-=:; '

# ===== KEYWORD CATEGORIES =====
# Strategy keywords
STRATEGY_KEYWORDS = [
    "strategy", "strategic", "vision", "mission", "objectives", "goals", 
    "growth", "expansion", "market", "competitive", "advantage", "innovation",
    "transformation", "digital", "technology", "future", "planning", "roadmap",
    "investment", "capital", "allocation", "portfolio", "diversification",
    "acquisition", "merger", "partnership", "collaboration", "joint venture",
    "sustainability", "ESG", "environmental", "social", "governance"
]

# Governance keywords  
GOVERNANCE_KEYWORDS = [
    "governance", "board", "directors", "chairman", "CEO", "executive",
    "committee", "oversight", "supervision", "accountability", "transparency",
    "ethics", "compliance", "regulatory", "policy", "framework", "structure",
    "independence", "nomination", "remuneration", "audit committee",
    "stakeholder", "shareholder", "investor", "disclosure", "reporting"
]

# Risk keywords
RISK_KEYWORDS = [
    "risk", "risks", "uncertainty", "threat", "exposure", "vulnerability",
    "mitigation", "management", "assessment", "monitoring", "control",
    "operational", "credit", "market", "liquidity", "regulatory", "compliance",
    "cyber", "cybersecurity", "information security", "data protection",
    "business continuity", "disaster recovery", "stress testing", "scenario",
    "capital adequacy", "Basel", "prudential", "ICAAP", "ILAAP"
]

# Audit keywords
AUDIT_KEYWORDS = [
    "audit", "auditor", "auditing", "assurance", "review", "examination",
    "testing", "verification", "validation", "internal control", "control",
    "procedures", "processes", "effectiveness", "efficiency", "accuracy",
    "completeness", "existence", "valuation", "presentation", "disclosure",
    "material", "materiality", "significant", "deficiency", "weakness",
    "recommendation", "management letter", "findings", "observations"
]

# ===== VALIDATION FUNCTIONS =====
def validate_config() -> bool:
    """
    Validate configuration settings
    
    Returns:
        True if configuration is valid, False otherwise
    """
    valid = True
    
    # Check Companies House API key
    if not COMPANIES_HOUSE_API_KEY or COMPANIES_HOUSE_API_KEY.startswith('your_'):
        logger.warning("Companies House API key not configured")
        valid = False
    
    # Check if at least one AI API key is configured (optional)
    if not OPENAI_API_KEY and not ANTHROPIC_API_KEY:
        logger.info("No AI API keys configured - will use pattern-based analysis")
    elif OPENAI_API_KEY and OPENAI_API_KEY.startswith('your_'):
        logger.warning("OpenAI API key appears to be placeholder")
    elif ANTHROPIC_API_KEY and ANTHROPIC_API_KEY.startswith('your_'):
        logger.warning("Anthropic API key appears to be placeholder")
    
    # Check directories
    for directory in [TEMP_FOLDER, OUTPUT_DIR]:
        if not directory.exists():
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                logger.error(f"Could not create directory {directory}: {e}")
                valid = False
    
    return valid

def validate_company_number(company_number: str) -> bool:
    """
    Validate UK company number format
    
    Args:
        company_number: Company number to validate
        
    Returns:
        True if valid format, False otherwise
    """
    if not company_number:
        return False
    
    # Remove any spaces
    company_number = company_number.replace(' ', '')
    
    # UK company numbers are typically 8 characters
    # Can be 8 digits, or 2 letters + 6 digits (Scotland/NI)
    patterns = [
        r'^\d{8}$',  # 8 digits
        r'^[A-Z]{2}\d{6}$',  # 2 letters + 6 digits (Scotland: SC, Northern Ireland: NI)
        r'^[A-Z]\d{7}$'  # 1 letter + 7 digits (rare but valid)
    ]
    
    return any(re.match(pattern, company_number.upper()) for pattern in patterns)

def validate_file_size(file_size: int) -> bool:
    """
    Validate file size
    
    Args:
        file_size: File size in bytes
        
    Returns:
        True if size is acceptable, False otherwise
    """
    return 0 < file_size <= MAX_FILE_SIZE

def validate_file_type(filename: str) -> bool:
    """
    Validate file type
    
    Args:
        filename: Name of the file
        
    Returns:
        True if file type is allowed, False otherwise
    """
    if not filename:
        return False
    
    file_extension = Path(filename).suffix.lower()
    return file_extension in ALLOWED_FILE_TYPES

# ===== MESSAGE TEMPLATES =====
ERROR_MESSAGES = {
    'company_not_found': "Company not found in Companies House records",
    'no_filings': "No annual accounts found for this company",
    'extraction_failed': "Could not extract content from documents",
    'analysis_failed': "Analysis could not be completed",
    'api_error': "API request failed",
    'invalid_company_number': "Invalid UK company number format",
    'file_too_large': f"File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit",
    'unsupported_file_type': f"File type not supported. Allowed: {', '.join(ALLOWED_FILE_TYPES)}"
}

SUCCESS_MESSAGES = {
    'company_validated': "Company successfully validated",
    'filings_downloaded': "Annual accounts downloaded successfully",
    'content_extracted': "Content extracted from documents",
    'analysis_complete': "Archetype analysis completed",
    'results_saved': "Results saved successfully"
}

# ===== ARCHETYPE DEFINITIONS =====
BUSINESS_STRATEGY_ARCHETYPES = {
    'Scale-through-Distribution': 'Gains share primarily by adding new channels or partners faster than control maturity develops.',
    'Land-Grab Platform': 'Uses aggressive below-market pricing or incentives to build a large multi-sided platform quickly.',
    'Asset-Velocity Maximiser': 'Chases rapid originations/turnover even at higher funding costs.',
    'Yield-Hunting': 'Prioritises high-margin segments and prices for risk premium.',
    'Fee-Extraction Engine': 'Relies on ancillary fees and cross-sales for majority of profit.',
    'Disciplined Specialist Growth': 'Niche focus with strong underwriting edge; grows opportunistically.',
    'Expert Niche Leader': 'Deep expertise in micro-segment with modest but steady growth.',
    'Service-Driven Differentiator': 'Wins by superior client experience rather than price.',
    'Cost-Leadership Operator': 'Drives ROE via lean cost base and operational efficiency.',
    'Tech-Productivity Accelerator': 'Heavy automation/AI to compress unit costs.',
    'Product-Innovation Flywheel': 'Constantly launches novel product variants/features.',
    'Data-Monetisation Pioneer': 'Converts proprietary data into revenue streams.',
    'Balance-Sheet Steward': 'Low-risk appetite, prioritises capital strength.',
    'Regulatory Shelter Occupant': 'Leverages regulatory protections to defend position.',
    'Regulator-Mandated Remediation': 'Operating under regulatory constraints.',
    'Wind-down / Run-off': 'Managing existing book to maturity.',
    'Strategic Withdrawal': 'Actively divesting to refocus core franchise.',
    'Distressed-Asset Harvester': 'Buys under-priced portfolios during downturns.',
    'Counter-Cyclical Capitaliser': 'Expands when competitors retrench.'
}

RISK_STRATEGY_ARCHETYPES = {
    'Risk-First Conservative': 'Prioritises capital preservation and regulatory compliance.',
    'Rules-Led Operator': 'Strict adherence to rules and checklists.',
    'Resilience-Focused Architect': 'Designs for operational continuity and crisis endurance.',
    'Strategic Risk-Taker': 'Accepts elevated risk to unlock growth or margin.',
    'Control-Lag Follower': 'Expands ahead of control maturity development.',
    'Reactive Remediator': 'Risk strategy is event-driven and compliance-focused.',
    'Reputation-First Shield': 'Actively avoids reputational or political risk.',
    'Embedded Risk Partner': 'Risk teams embedded in frontline decisions.',
    'Quant-Control Enthusiast': 'Leverages data and analytics as core risk tools.',
    'Tick-Box Minimalist': 'Superficial controls for compliance optics.',
    'Mission-Driven Prudence': 'Risk appetite anchored in stakeholder protection.'
}

# ===== LOGGING CONFIGURATION =====
def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    log_level = getattr(logging, level.upper())
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set specific loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

# ===== RUNTIME CONFIGURATION =====
# Setup logging if this module is imported
if not logging.getLogger().handlers:
    setup_logging()

# Validate configuration on import
if __name__ == "__main__":
    print("Configuration Validation:")
    print(f"Base directory: {BASE_DIR}")
    print(f"Companies House API key configured: {bool(COMPANIES_HOUSE_API_KEY and not COMPANIES_HOUSE_API_KEY.startswith('your_'))}")
    print(f"OpenAI API key configured: {bool(OPENAI_API_KEY and not OPENAI_API_KEY.startswith('your_'))}")
    print(f"Anthropic API key configured: {bool(ANTHROPIC_API_KEY and not ANTHROPIC_API_KEY.startswith('your_'))}")
    print(f"Validation result: {validate_config()}")
else:
    # Silent validation when imported
    validate_config()