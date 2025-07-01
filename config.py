import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Main variables
COMPANIES_HOUSE_API_KEY = os.getenv('CH_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
DEFAULT_MAX_YEARS = int(os.getenv('DEFAULT_MAX_YEARS', 3))

# Companies House API settings
CH_BASE_URL = "https://api.company-information.service.gov.uk"
REQUEST_TIMEOUT = int(os.getenv('DOWNLOAD_TIMEOUT', 300))

# Messages
ERROR_MESSAGES = {
    "api_key_missing": "Companies House API key is required",
    "company_not_found": "Company not found",
    "download_failed": "Download failed",
    "timeout": "Request timed out"
}

SUCCESS_MESSAGES = {
    "company_found": "Company found successfully",
    "download_complete": "Download completed",
    "validation_passed": "Validation successful"
}

# Content processing keywords
STRATEGY_KEYWORDS = [
    "strategy", "strategic", "vision", "mission", "objectives",
    "growth", "expansion", "transformation", "digital", "technology",
    "modernisation", "customer outcomes", "sustainability"
]

GOVERNANCE_KEYWORDS = [
    "governance", "board", "directors", "committee", "audit",
    "remuneration", "nomination", "risk committee", "compliance"
]

RISK_KEYWORDS = [
    "credit risk", "operational risk", "liquidity risk", "market risk",
    "compliance risk", "strategic risk", "conduct risk", "funding risk",
    "interest rate risk", "capital risk"
]

AUDIT_KEYWORDS = [
    "audit", "auditor", "internal audit", "external audit", "assurance",
    "control", "compliance", "governance", "risk management"
]

# Sentiment analysis keywords (moved from content_processor.py)
POSITIVE_WORDS = [
    "growth", "increase", "improve", "strong", "robust", "positive", 
    "successful", "expand", "opportunity", "confident", "optimistic",
    "profitable", "efficient", "innovative", "competitive", "resilient"
]

NEGATIVE_WORDS = [
    "decline", "decrease", "reduce", "weak", "poor", "negative",
    "challenge", "risk", "concern", "difficult", "loss", "falling",
    "uncertainty", "volatile", "pressure", "constraint", "adverse"
]

RISK_WORDS = [
    "risk", "threat", "vulnerability", "exposure", "uncertainty",
    "volatility", "compliance", "regulatory", "operational", "credit",
    "market", "liquidity", "cyber", "reputation", "strategic"
]

# Content processing limits
MAX_TOKENS_PER_CHUNK = 8000
TOTAL_TOKEN_LIMIT = 50000
MIN_CONTENT_LENGTH = 100
MAX_CONTENT_SECTIONS_PER_CATEGORY = 10

# AI Model configuration (moved from ai_analyzer.py)
DEFAULT_OPENAI_MODEL = "gpt-4"
DEFAULT_ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
AI_MAX_TOKENS = 4000
AI_TEMPERATURE = 0.3

# PDF processing settings
MAX_PAGES_OCR = 50
OCR_DPI = 150
OCR_CONFIG = "--psm 3"

# Text extraction settings (moved from main.py)
MIN_EXTRACTION_LENGTH = 200  # Minimum characters for successful extraction
MAX_FILE_SIZE_MB = 50
SUPPORTED_FILE_TYPES = [".pdf"]

# Directories
BASE_DIR = Path(__file__).parent
DOWNLOADS_FOLDER = BASE_DIR / "downloads"
RESULTS_FOLDER = BASE_DIR / "results"
TEMP_FOLDER = BASE_DIR / "temp"

for folder in [DOWNLOADS_FOLDER, RESULTS_FOLDER, TEMP_FOLDER]:
    folder.mkdir(exist_ok=True)

# Additional variables for consistency
DOWNLOAD_TIMEOUT = REQUEST_TIMEOUT

def validate_config():
    """Validate that required configuration is present"""
    if not COMPANIES_HOUSE_API_KEY:
        return False
    
    # Check if at least one AI API key is available
    ai_keys_available = bool(OPENAI_API_KEY) or bool(ANTHROPIC_API_KEY)
    if not ai_keys_available:
        print("Warning: No AI API keys configured. Will use fallback analysis.")
    
    return True

def validate_company_number(company_number):
    """Validate UK company number format"""
    import re
    if not company_number:
        return False
    patterns = [r"^\d{8}$", r"^[A-Z]{2}\d{6}$"]
    return any(re.match(p, company_number.strip().upper()) for p in patterns)

def validate_file_size(file_size_bytes):
    """Validate file size against maximum allowed"""
    max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    return file_size_bytes <= max_size_bytes

def validate_file_type(filename):
    """Validate file type against supported types"""
    from pathlib import Path
    file_ext = Path(filename).suffix.lower()
    return file_ext in SUPPORTED_FILE_TYPES

def get_download_path(filename):
    """Get path for downloaded file"""
    return DOWNLOADS_FOLDER / filename

def get_results_path(filename):
    """Get path for results file"""
    return RESULTS_FOLDER / filename

def get_temp_path(filename):
    """Get path for temporary file"""
    return TEMP_FOLDER / filename