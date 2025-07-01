# Strategy, Governance, Risk & Audit Analysis Tool

A modular Python tool for analyzing company accounts and classifying strategic evolution using AI-powered analysis.

## Features

- **Automated Account Download**: Fetches company accounts from Companies House API
- **Multi-Method PDF Extraction**: Uses pdfplumber, PyPDF2, and OCR for robust text extraction
- **AI-Powered Strategic Analysis**: Classifies business and risk strategies using OpenAI GPT-4
- **Automatic Content Chunking**: Handles large documents that exceed token limits
- **Comprehensive Reporting**: Generates detailed reports in multiple formats
- **Batch Processing**: Analyze multiple companies in a single run

## Architecture

The tool is built with a modular architecture for maintainability and testability:

```
├── config.py                 # Configuration and constants
├── companies_house_client.py  # Companies House API integration
├── pdf_extractor.py          # PDF text extraction
├── content_processor.py      # Content analysis and categorization
├── ai_analyzer.py           # AI-powered strategic analysis
├── file_manager.py          # File operations and storage
├── report_generator.py      # Report generation
└── main.py                  # Main orchestration
```

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install OCR dependencies (optional but recommended):**
   ```bash
   # macOS
   brew install tesseract
   
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # Windows - Download from:
   # https://github.com/UB-Mannheim/tesseract/wiki
   ```

4. **Set up API keys:**
   
   Create environment variables or update the keys in `config.py`:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export CH_API_KEY="your-companies-house-api-key"
   ```

## Usage

### Command Line Interface

Run the main script for interactive analysis:

```bash
python main.py
```

This provides options to:
- Analyze single company
- Batch analyze multiple companies  
- Show system status
- Exit

### Programmatic Usage

```python
from main import AnalysisOrchestrator

# Initialize the orchestrator
orchestrator = AnalysisOrchestrator()

# Analyze a single company
result = orchestrator.analyze_company("12345678", max_years=5)

# Batch analyze multiple companies
companies = ["12345678", "87654321", "11223344"]
batch_result = orchestrator.batch_analyze_companies(companies)
```

### Individual Module Usage

```python
# Use individual components
from companies_house_client import CompaniesHouseClient
from pdf_extractor import PDFExtractor
from ai_analyzer import AIAnalyzer

# Download accounts
client = CompaniesHouseClient()
download_results = client.download_company_accounts("12345678")

# Extract text from PDF
extractor = PDFExtractor()
with open("accounts.pdf", "rb") as f:
    extraction_result = extractor.extract_text_from_pdf(f.read(), "accounts.pdf")

# Run AI analysis
analyzer = AIAnalyzer()
evolution_result = analyzer.analyze_evolution(text_content, "12345678")
```

## Configuration

Key configuration options in `config.py`:

- **API Keys**: OpenAI and Companies House API credentials
- **File Paths**: Results and temporary file locations
- **Analysis Parameters**: Token limits, content thresholds
- **Strategy Archetypes**: Business and risk strategy classifications

## Strategy Archetypes

The tool classifies companies into strategic archetypes:

### Business Strategy Archetypes
- Scale-through-Distribution
- Land-Grab Platform  
- Asset-Velocity Maximiser
- Yield-Hunting
- Fee-Extraction Engine
- And 14 more...

### Risk Strategy Archetypes
- Risk-First Conservative
- Rules-Led Operator
- Resilience-Focused Architect
- Strategic Risk-Taker
- Control-Lag Follower
- And 6 more...

## Output Files

The tool generates several output files:

- **Portfolio Analysis**: Detailed JSON with all analysis data
- **Executive Report**: Human-readable analysis report
- **Executive Summary**: Concise summary of findings
- **JSON Summary**: Structured data for programmatic use
- **Debug Files**: Combined content and chunk analysis (when needed)

## System Requirements

- Python 3.8+
- Internet connection for API calls
- ~500MB disk space for temporary files
- OpenAI API access
- Companies House API key (free)

## Error Handling

The tool includes robust error handling:

- **PDF Extraction**: Falls back through multiple extraction methods
- **API Limits**: Automatic chunking for large content
- **File Operations**: Comprehensive error checking and cleanup
- **Network Issues**: Retry logic and graceful degradation

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black *.py
flake8 *.py
```

### Adding New Features

The modular architecture makes it easy to extend:

1. **New PDF Extraction Method**: Add to `PDFExtractor` class
2. **Additional Content Analysis**: Extend `ContentProcessor`
3. **New Report Formats**: Add to `ReportGenerator`
4. **Different AI Models**: Modify `AIAnalyzer`

## Troubleshooting

### Common Issues

1. **"OpenAI client initialisation failed"**
   - Check your OpenAI API key
   - Verify internet connection

2. **"PDF extraction failed"**
   - Install missing dependencies (pdfplumber, PyPDF2, tesseract)
   - Check PDF file isn't corrupted

3. **"No accounts found"**
   - Verify company number is correct
   - Check Companies House API key
   - Company may not have filed recent accounts

4. **"Token limit exceeded"**
   - The tool should handle this automatically with chunking
   - If issues persist, check content size in debug files

### Getting Help

- Check the debug files generated in `strategic_analysis_results/`
- Review system status with option 3 in the CLI
- Ensure all dependencies are installed correctly

## License

This tool is for educational and research purposes. Please respect API rate limits and terms of service for OpenAI and Companies House APIs.