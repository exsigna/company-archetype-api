#!/usr/bin/env python3
"""
Strategy, Governance, Risk & Audit Analysis Tool
Fixed version with chunking to handle token limits
"""

import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from openai import OpenAI
import re
import time
import os
import tempfile
import io

# --- Configuration ---
OPENAI_API_KEY = "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"
CH_API_KEY = "7cd3b2c5-ba63-41e9-a408-f9716e40edb3"
CH_BASE_URL = "https://api.company-information.service.gov.uk"
RESULTS_FOLDER = Path("strategic_analysis_results")

# Web Search API Configuration
BRAVE_SEARCH_API_KEY = "BSAlM4Cgleh6VPGbNUyBv-lr5YhezcM"
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Create results folder
RESULTS_FOLDER.mkdir(exist_ok=True)

# Token counting for chunking
def count_tokens_rough(text: str) -> int:
    """Rough token count estimation (4 chars ≈ 1 token)"""
    return len(text) // 4

def smart_chunk_text(text: str, max_tokens: int = 45000) -> List[str]:
    """Split text into chunks that respect sentence boundaries"""
    estimated_tokens = count_tokens_rough(text)
    
    if estimated_tokens <= max_tokens:
        return [text]
    
    # Split into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens_rough(sentence)
        
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            # Start new chunk
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            current_tokens = sentence_tokens
        else:
            current_chunk += sentence + " "
            current_tokens += sentence_tokens
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

# Initialise OpenAI client with error handling
client = None
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("OpenAI client initialised successfully")
except Exception as e:
    print(f"OpenAI client initialisation failed: {e}")
    print("Web search analysis will continue without AI classification")

# Check for required dependencies for PDF analysis
try:
    import PyPDF2
    print("PyPDF2 available for PDF analysis")
except ImportError:
    print("PyPDF2 not installed. Install with: pip install PyPDF2")
    print("   PDF account analysis will be disabled.")
    PyPDF2 = None

try:
    import pdfplumber
    print("pdfplumber available for enhanced PDF extraction")
except ImportError:
    print("pdfplumber not installed. Install with: pip install pdfplumber")
    print("   Enhanced PDF extraction will be disabled.")
    pdfplumber = None

try:
    import pytesseract
    from pdf2image import convert_from_bytes
    print("OCR libraries available (pytesseract + pdf2image)")
except ImportError:
    print("OCR libraries not installed. Install with: pip install pytesseract pdf2image")
    print("   MacOS also needs: brew install tesseract")
    print("   Image-based PDF extraction will be disabled.")
    pytesseract = None
    convert_from_bytes = None

# --- Hardcoded Strategy Archetypes ---
BUSINESS_STRATEGY_ARCHETYPES = [
    ("Scale-through-Distribution", "Gains share primarily by adding new channels or partners faster than control maturity develops."),
    ("Land-Grab Platform", "Uses aggressive below-market pricing or incentives to build a large multi-sided platform quickly (BNPL, FX apps, etc.)."),
    ("Asset-Velocity Maximiser", "Chases rapid originations / turnover (e.g. bridging, invoice finance) even at higher funding costs."),
    ("Yield-Hunting", "Prioritises high-margin segments (credit-impaired, niche commercial) and prices for risk premium."),
    ("Fee-Extraction Engine", "Relies on ancillary fees, add-ons or cross-sales for majority of profit (packaged accounts, paid add-ons)."),
    ("Disciplined Specialist Growth", "Niche focus with strong underwriting edge; grows opportunistically while recycling balance-sheet."),
    ("Expert Niche Leader", "Deep expertise in a micro-segment (e.g. HNW Islamic mortgages) with modest but steady growth."),
    ("Service-Driven Differentiator", "Wins by superior client experience / advice rather than price or scale."),
    ("Cost-Leadership Operator", "Drives ROE via lean cost base, digital self-service, zero-based budgeting."),
    ("Tech-Productivity Accelerator", "Heavy automation/AI to compress unit costs and redeploy staff."),
    ("Product-Innovation Flywheel", "Constantly launches novel product variants/features to capture share."),
    ("Data-Monetisation Pioneer", "Converts proprietary data into fees."),
    ("Balance-Sheet Steward", "Low-risk appetite, prioritises capital strength and membership value."),
    ("Regulatory Shelter Occupant", "Leverages regulatory or franchise protections to defend share."),
    ("Regulator-Mandated Remediation", "Operating under s.166, VREQ or RMAR constraints; resources diverted to fix historical failings."),
    ("Wind-down / Run-off", "Managing existing book to maturity or sale; minimal new origination."),
    ("Strategic Withdrawal", "Actively divesting lines/geographies to refocus core franchise."),
    ("Distressed-Asset Harvester", "Buys NPLs or under-priced portfolios during downturns for future upside."),
    ("Counter-Cyclical Capitaliser", "Expands lending precisely when competitors retrench, using strong liquidity.")
]

RISK_STRATEGY_ARCHETYPES = [
    ("Risk-First Conservative", "Prioritises capital preservation and regulatory compliance; growth is secondary to resilience."),
    ("Rules-Led Operator", "Strict adherence to rules and checklists; prioritises control consistency over judgment or speed."),
    ("Resilience-Focused Architect", "Designs for operational continuity and crisis endurance; invests in stress testing and scenario planning."),
    ("Strategic Risk-Taker", "Accepts elevated risk to unlock growth or margin; uses pricing, underwriting, or innovation to offset exposure."),
    ("Control-Lag Follower", "Expands products or markets ahead of control maturity; plays regulatory catch-up after scaling."),
    ("Reactive Remediator", "Risk strategy is event-driven, typically shaped by enforcement, audit findings, or external reviews."),
    ("Reputation-First Shield", "Actively avoids reputational or political risk, sometimes at the expense of commercial logic."),
    ("Embedded Risk Partner", "Risk teams are embedded in frontline decisions; risk appetite is shaped collaboratively across the business."),
    ("Quant-Control Enthusiast", "Leverages data, automation, and predictive analytics as core risk management tools."),
    ("Tick-Box Minimalist", "Superficial control structures exist for compliance optics, not genuine governance intent."),
    ("Mission-Driven Prudence", "Risk appetite is anchored in stakeholder protection, community outcomes, or long-term social licence.")
]

def analyze_chunk(content: str, chunk_num: int, total_chunks: int) -> dict:
    """Analyze a single chunk of content for strategic indicators"""
    business = "\n".join([f"- {name}: {desc}" for name, desc in BUSINESS_STRATEGY_ARCHETYPES])
    risk = "\n".join([f"- {name}: {desc}" for name, desc in RISK_STRATEGY_ARCHETYPES])
    
    prompt = f"""You are analyzing chunk {chunk_num}/{total_chunks} of company financial documents.

Extract strategic indicators from this content chunk and map them to the provided archetypes.

BUSINESS ARCHETYPES:
{business}

RISK ARCHETYPES:
{risk}

CONTENT CHUNK:
{content}

Identify business and risk strategy signals in this chunk. Return JSON:
{{
  "business_signals": ["list of business strategy indicators found"],
  "risk_signals": ["list of risk strategy indicators found"], 
  "governance_signals": ["governance elements found"],
  "key_strategic_content": ["important strategic quotes/statements"],
  "year_indicators": ["any time period indicators"],
  "confidence": "high/medium/low"
}}"""

    try:
        if client is None:
            return {"error": "OpenAI client not initialized"}
            
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Extract strategic indicators from financial documents. Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Clean JSON response
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = result_text.strip()
        
        return json.loads(result_text)
        
    except json.JSONDecodeError as e:
        return {"error": f"JSON parsing error: {str(e)}", "raw_response": result_text if 'result_text' in locals() else "No response"}
    except Exception as e:
        return {"error": f"API error: {str(e)}"}

def synthesize_evolution(chunk_analyses: List[dict], company_number: str) -> dict:
    """Synthesize chunk analyses into final strategic classification"""
    business = "\n".join([f"- {name}: {desc}" for name, desc in BUSINESS_STRATEGY_ARCHETYPES])
    risk = "\n".join([f"- {name}: {desc}" for name, desc in RISK_STRATEGY_ARCHETYPES])
    
    # Combine successful analyses
    all_signals = {
        "business_signals": [],
        "risk_signals": [],
        "governance_signals": [],
        "key_strategic_content": []
    }
    
    for analysis in chunk_analyses:
        if "error" not in analysis:
            for key in all_signals.keys():
                if key in analysis:
                    all_signals[key].extend(analysis[key])
    
    synthesis_data = json.dumps(all_signals, indent=2)
    
    prompt = f"""Based on strategic signals extracted from multiple document chunks, provide final classification:

BUSINESS ARCHETYPES:
{business}

RISK ARCHETYPES:
{risk}

EXTRACTED STRATEGIC SIGNALS:
{synthesis_data}

Classify the firm's strategy and evolution. Return JSON:
{{
  "business_primary": "primary business strategy archetype",
  "business_secondary": "secondary business strategy archetype", 
  "risk_primary": "primary risk strategy archetype",
  "risk_secondary": "secondary risk strategy archetype",
  "business_rationale": "detailed explanation for business classification",
  "risk_rationale": "detailed explanation for risk classification", 
  "evolution_summary": "how strategy evolved over time based on evidence",
  "confidence_level": "high/medium/low"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Provide final strategic classification based on extracted signals. Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=3000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Clean JSON response
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = result_text.strip()
        
        return json.loads(result_text)
        
    except json.JSONDecodeError as e:
        return {"error": f"JSON parsing error: {str(e)}", "raw_response": result_text if 'result_text' in locals() else "No response"}
    except Exception as e:
        return {"error": f"Synthesis API error: {str(e)}"}

def analyze_evolution(content: str, company_number: str) -> dict:
    """Analyze strategic evolution with automatic chunking for large content"""
    print(f"Starting evolution analysis for company {company_number}")
    
    # Estimate content size
    estimated_tokens = count_tokens_rough(content)
    print(f"Estimated content tokens: {estimated_tokens:,}")
    
    # If content is small enough for single call
    if estimated_tokens <= 80000:
        print("Content fits in single call - using direct analysis")
        return analyze_evolution_single_call(content, company_number)
    
    # Use chunked approach for large content
    print("Content too large - using chunked analysis")
    chunks = smart_chunk_text(content, max_tokens=45000)
    print(f"Split content into {len(chunks)} chunks")
    
    # Analyze each chunk
    chunk_analyses = []
    for i, chunk in enumerate(chunks):
        print(f"Analyzing chunk {i + 1}/{len(chunks)}...")
        chunk_tokens = count_tokens_rough(chunk)
        print(f"  Chunk {i + 1}: ~{chunk_tokens:,} tokens")
        
        analysis = analyze_chunk(chunk, i + 1, len(chunks))
        chunk_analyses.append(analysis)
        
        if "error" in analysis:
            print(f"  Error in chunk {i + 1}: {analysis['error']}")
        else:
            print(f"  Chunk {i + 1} analyzed successfully")
        
        # Small delay to avoid rate limits
        time.sleep(0.5)
    
    # Synthesize final result
    print("Synthesizing chunk analyses into final classification...")
    final_result = synthesize_evolution(chunk_analyses, company_number)
    
    # Add metadata
    final_result["analysis_metadata"] = {
        "total_estimated_tokens": estimated_tokens,
        "chunks_processed": len(chunks),
        "chunks_successful": len([a for a in chunk_analyses if "error" not in a]),
        "analysis_method": "chunked"
    }
    
    # Save debug info
    debug_path = os.path.join("strategic_analysis_results", f"{company_number}_chunk_debug.json")
    try:
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump({
                "chunk_analyses": chunk_analyses,
                "final_synthesis": final_result,
                "metadata": final_result["analysis_metadata"]
            }, f, indent=2)
        print(f"Debug analysis saved to: {debug_path}")
    except Exception as e:
        print(f"Could not save debug info: {e}")
    
    return final_result

def analyze_evolution_single_call(content: str, company_number: str) -> dict:
    """Original single-call analysis for smaller content"""
    business = "\n".join([f"- {name}: {desc}" for name, desc in BUSINESS_STRATEGY_ARCHETYPES])
    risk = "\n".join([f"- {name}: {desc}" for name, desc in RISK_STRATEGY_ARCHETYPES])
    
    prompt = (
        "You are a regulatory strategy analyst. Based on the firm's account content from multiple years, complete the following:\n\n"
        "1. Classify the dominant and secondary BUSINESS strategy archetypes\n"
        "2. Classify the dominant and secondary RISK strategy archetypes\n"
        "3. Explain how the firm's strategic posture evolved over time\n"
        "4. Use tone, structure, and strategic disclosures to infer progression\n"
        "5. Do NOT say 'insufficient data' — classify using best approximation\n\n"
        "BUSINESS ARCHETYPES:\n" + business + "\n\n"
        "RISK ARCHETYPES:\n" + risk + "\n\n"
        "CONTENT:\n" + content + "\n\n"
        "Return JSON:\n"
        "{\n"
        "  \"business_primary\": \"...\",\n"
        "  \"business_secondary\": \"...\",\n"
        "  \"risk_primary\": \"...\",\n"
        "  \"risk_secondary\": \"...\",\n"
        "  \"business_rationale\": \"...\",\n"
        "  \"risk_rationale\": \"...\",\n"
        "  \"evolution_summary\": \"...\"\n"
        "}\n"
    )
    
    print("Using model: gpt-4o (single call)")
    estimated_tokens = count_tokens_rough(prompt)
    print(f"Estimated prompt tokens: {estimated_tokens:,}")
    
    # Create results directory if it doesn't exist
    os.makedirs("strategic_analysis_results", exist_ok=True)
    
    # Save prompt to file
    prompt_path = os.path.join("strategic_analysis_results", f"{company_number}_openai_prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as pf:
        pf.write(prompt)
    print(f"Prompt saved to: {prompt_path}")
    
    try:
        if client is None:
            return {"error": "OpenAI client not initialized"}
            
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a regulatory strategy analyst. You must respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4000
        )
        
        result_text = response.choices[0].message.content.strip()
        print(f"OpenAI response length: {len(result_text)} chars")
        print(f"OpenAI response preview: {result_text[:200]}...")
        
        # Clean the response - remove any markdown formatting
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = result_text.strip()
        
        # Parse JSON
        parsed_result = json.loads(result_text)
        parsed_result["analysis_metadata"] = {
            "total_estimated_tokens": estimated_tokens,
            "analysis_method": "single_call"
        }
        return parsed_result
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw response: {result_text if 'result_text' in locals() else 'No response'}")
        return {"error": f"Failed to parse JSON response: {str(e)}", "raw_response": result_text if 'result_text' in locals() else "No response"}
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return {"error": f"OpenAI API call failed: {str(e)}"}

def get_trading_names(company_number: str) -> List[str]:
    """Fetch confirmed trading names from Companies House API"""
    confirmed_names = []

    try:
        url = f"{CH_BASE_URL}/company/{company_number}"
        response = requests.get(url, auth=(CH_API_KEY, ""))
        if response.status_code == 200:
            data = response.json()

            current_name = data.get("company_name", "")
            if current_name:
                confirmed_names.append(current_name)
                print(f"Confirmed current name: {current_name}")

            previous_names = data.get("previous_company_names", [])
            print(f"Found {len(previous_names)} previous names in Companies House")
            for prev_name in previous_names:
                if isinstance(prev_name, dict):
                    name = prev_name.get("name", "")
                    if name:
                        confirmed_names.append(name)
                        print(f"Confirmed previous name: {name}")

        # Remove duplicates whilst preserving order
        unique_names = []
        seen = set()
        for name in confirmed_names:
            if name.lower() not in seen:
                unique_names.append(name)
                seen.add(name.lower())

        print(f"Total confirmed trading names: {len(unique_names)}")
        return unique_names

    except Exception as e:
        print(f"Error fetching trading names: {e}")
        return ["Unknown Company"]

def get_company_profile(company_number: str) -> Dict:
    """Fetch company profile from Companies House API"""
    url = f"{CH_BASE_URL}/company/{company_number}"
    try:
        response = requests.get(url, auth=(CH_API_KEY, ""))
        return response.json() if response.status_code == 200 else {}
    except Exception as e:
        print(f"Error fetching company profile: {e}")
        return {}

def download_company_accounts(company_number: str, max_years: int = 5, save_dir: str = None) -> Dict:
    """Download recent company accounts PDFs from Companies House API"""

    if save_dir is None:
        save_dir = f"{company_number}_accounts"

    print(f"Downloading accounts for company {company_number} (last {max_years} years)")

    # Create directory
    os.makedirs(save_dir, exist_ok=True)

    download_results = {
        "company_number": company_number,
        "save_directory": save_dir,
        "downloaded_files": [],
        "download_count": 0,
        "total_accounts_found": 0,
        "error_count": 0,
        "earliest_date": None,
        "latest_date": None
    }

    # Calculate cutoff date for filtering
    cutoff_date = datetime.now() - timedelta(days=max_years * 365)
    cutoff_date_str = cutoff_date.strftime("%Y-%m-%d")

    start_index = 0

    try:
        while True:
            # Get filing history
            url = f"{CH_BASE_URL}/company/{company_number}/filing-history?start_index={start_index}&items_per_page=100"

            response = requests.get(url, auth=(CH_API_KEY, ''))
            response.raise_for_status()
            data = response.json()

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                item_date = item.get("date", "")

                # Skip if filing is too old
                if item_date < cutoff_date_str:
                    continue

                description = item.get("description", "").lower()
                category = item.get("category", "")
                subcategory = item.get("subcategory", "")

                # Check if this is an accounts filing
                is_accounts = (
                    "accounts" in description or
                    "full accounts" in description or
                    category == "accounts" or
                    subcategory == "full"
                )

                if is_accounts:
                    download_results["total_accounts_found"] += 1

                    # Update date range
                    if not download_results["earliest_date"] or item_date < download_results["earliest_date"]:
                        download_results["earliest_date"] = item_date
                    if not download_results["latest_date"] or item_date > download_results["latest_date"]:
                        download_results["latest_date"] = item_date

                    print(f"Found accounts: {item.get('description', 'N/A')} ({item_date})")

                    # Check if document is downloadable
                    if 'links' not in item or 'document_metadata' not in item['links']:
                        print("   Document not downloadable")
                        continue

                    try:
                        # Get document metadata
                        doc_meta_url = item['links']['document_metadata']
                        doc_res = requests.get(doc_meta_url, auth=(CH_API_KEY, ''))
                        doc_res.raise_for_status()
                        doc_data = doc_res.json()

                        # Get PDF download URL
                        if 'links' not in doc_data or 'document' not in doc_data['links']:
                            print("   PDF download link not available")
                            continue

                        pdf_url = doc_data['links']['document']

                        # Generate filename safely
                        transaction_id = item.get("transaction_id", "no_id")
                        file_name = f"{item_date}_{transaction_id}.pdf"
                        file_path = os.path.join(save_dir, file_name)

                        # Check if file already exists
                        if os.path.exists(file_path):
                            print(f"   Already downloaded: {file_name}")
                            download_results["downloaded_files"].append({
                                "filename": file_name,
                                "filepath": file_path,
                                "filing_date": item_date,
                                "description": item.get('description', ''),
                                "status": "already_exists"
                            })
                            continue

                        # Download PDF
                        print(f"   Downloading: {file_name}")
                        pdf_response = requests.get(pdf_url, auth=(CH_API_KEY, ''))
                        pdf_response.raise_for_status()

                        with open(file_path, "wb") as f:
                            f.write(pdf_response.content)

                        download_results["download_count"] += 1
                        download_results["downloaded_files"].append({
                            "filename": file_name,
                            "filepath": file_path,
                            "filing_date": item_date,
                            "description": item.get('description', ''),
                            "status": "downloaded",
                            "file_size": len(pdf_response.content),
                        })

                        print(f"   Downloaded: {file_name} ({len(pdf_response.content):,} bytes)")

                    except Exception as e:
                        print(f"   Download error: {e}")
                        download_results["error_count"] += 1

            # Check if we've processed all items or reached our date limit
            if len(items) < 100:
                break
            start_index += 100

    except Exception as e:
        print(f"Error fetching filing history: {e}")
        download_results["error"] = str(e)

    print(f"Download complete: {download_results['download_count']} new files, {download_results['total_accounts_found']} accounts found")
    return download_results

def process_financial_table(table) -> str:
    """Convert table data to searchable text format"""
    if not table or not any(table):
        return ""

    table_text = ""

    for row_num, row in enumerate(table):
        if not row or not any(row):
            continue

        # Clean and join cells
        clean_cells = []
        for cell in row:
            if cell:
                # Clean cell content
                cell_str = str(cell).strip()
                if cell_str and cell_str not in ['', 'None']:
                    clean_cells.append(cell_str)

        if clean_cells:
            # Join with pipes for easy parsing
            row_text = " | ".join(clean_cells)
            table_text += f"ROW{row_num}: {row_text}\n"

    return table_text

def extract_strategy_governance_risk_audit(text: str) -> Dict:
    """Extract strategy, governance, risk and audit content from PDF text"""

    findings = {
        "strategy_content": [],
        "governance_content": [],
        "risk_content": [],
        "audit_content": [],
        "summary": {
            "strategy_found": False,
            "governance_found": False,
            "risk_found": False,
            "audit_found": False
        }
    }

    text_lower = text.lower()

    # Enhanced Strategy keywords and patterns - capture larger context
    strategy_patterns = [
        r'.{0,300}strategic.{0,500}',
        r'.{0,300}strategy.{0,500}',
        r'.{0,300}business model.{0,500}',
        r'.{0,300}transformation.{0,500}',
        r'.{0,300}future plans?.{0,500}',
        r'.{0,300}objectives?.{0,500}',
        r'.{0,300}digital.{0,500}',
        r'.{0,300}innovation.{0,500}',
        r'.{0,300}growth.{0,500}',
        r'.{0,300}expansion.{0,500}',
        r'.{0,300}market position.{0,500}',
        r'.{0,300}competitive.{0,500}',
        r'.{0,300}lending.{0,500}',
        r'.{0,300}origination.{0,500}',
        r'.{0,300}loan book.{0,500}',
        r'.{0,300}customer.{0,500}',
        r'.{0,300}product.{0,500}',
        r'.{0,300}service.{0,500}'
    ]

    # Enhanced Governance keywords and patterns
    governance_patterns = [
        r'.{0,300}governance.{0,500}',
        r'.{0,300}board.{0,500}',
        r'.{0,300}directors?.{0,500}',
        r'.{0,300}chairman.{0,500}',
        r'.{0,300}chief executive.{0,500}',
        r'.{0,300}executive committee.{0,500}',
        r'.{0,300}audit committee.{0,500}',
        r'.{0,300}risk committee.{0,500}',
        r'.{0,300}remuneration.{0,500}',
        r'.{0,300}nomination.{0,500}',
        r'.{0,300}independent directors?.{0,500}',
        r'.{0,300}board effectiveness.{0,500}',
        r'.{0,300}oversight.{0,500}',
        r'.{0,300}leadership.{0,500}'
    ]

    # Enhanced Risk keywords and patterns
    risk_patterns = [
        r'.{0,300}risk management.{0,500}',
        r'.{0,300}risk appetite.{0,500}',
        r'.{0,300}risk framework.{0,500}',
        r'.{0,300}operational risk.{0,500}',
        r'.{0,300}credit risk.{0,500}',
        r'.{0,300}market risk.{0,500}',
        r'.{0,300}liquidity risk.{0,500}',
        r'.{0,300}conduct risk.{0,500}',
        r'.{0,300}regulatory risk.{0,500}',
        r'.{0,300}reputational risk.{0,500}',
        r'.{0,300}cyber risk.{0,500}',
        r'.{0,300}risk tolerance.{0,500}',
        r'.{0,300}stress test.{0,500}',
        r'.{0,300}capital adequacy.{0,500}',
        r'.{0,300}three lines.{0,500}',
        r'.{0,300}control.{0,500}',
        r'.{0,300}principal risks.{0,500}'
    ]

    # Enhanced Audit keywords and patterns
    audit_patterns = [
        r'.{0,300}audit.{0,500}',
        r'.{0,300}internal audit.{0,500}',
        r'.{0,300}external audit.{0,500}',
        r'.{0,300}auditor.{0,500}',
        r'.{0,300}assurance.{0,500}',
        r'.{0,300}compliance.{0,500}',
        r'.{0,300}control framework.{0,500}',
        r'.{0,300}internal controls?.{0,500}',
        r'.{0,300}audit opinion.{0,500}',
        r'.{0,300}going concern.{0,500}',
        r'.{0,300}regulatory.{0,500}',
        r'.{0,300}fca.{0,500}',
        r'.{0,300}prudential.{0,500}'
    ]

    # Extract strategy content
    for pattern in strategy_patterns:
        matches = re.finditer(pattern, text_lower, re.MULTILINE | re.DOTALL)
        for match in matches:
            content = match.group(0).strip()
            if len(content) > 50:  # Increased minimum content length
                findings["strategy_content"].append({
                    "context": "Strategic content found",
                    "content": content
                })
                findings["summary"]["strategy_found"] = True

    # Extract governance content
    for pattern in governance_patterns:
        matches = re.finditer(pattern, text_lower, re.MULTILINE | re.DOTALL)
        for match in matches:
            content = match.group(0).strip()
            if len(content) > 50:
                findings["governance_content"].append({
                    "context": "Governance content found",
                    "content": content
                })
                findings["summary"]["governance_found"] = True

    # Extract risk content
    for pattern in risk_patterns:
        matches = re.finditer(pattern, text_lower, re.MULTILINE | re.DOTALL)
        for match in matches:
            content = match.group(0).strip()
            if len(content) > 50:
                findings["risk_content"].append({
                    "context": "Risk management content found",
                    "content": content
                })
                findings["summary"]["risk_found"] = True

    # Extract audit content
    for pattern in audit_patterns:
        matches = re.finditer(pattern, text_lower, re.MULTILINE | re.DOTALL)
        for match in matches:
            content = match.group(0).strip()
            if len(content) > 50:
                findings["audit_content"].append({
                    "context": "Audit/compliance content found",
                    "content": content
                })
                findings["summary"]["audit_found"] = True

    # Deduplicate and limit results to top findings
    for category in ["strategy_content", "governance_content", "risk_content", "audit_content"]:
        # Remove duplicates and limit to top findings
        unique_content = []
        seen_content = set()
        for item in findings[category]:
            content_key = item["content"][:200]  # Use first 200 chars as key for better deduplication
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_content.append(item)
                if len(unique_content) >= 10:  # Increased to 10 per category
                    break
        findings[category] = unique_content

    return findings

def extract_accounts_content_from_pdf(pdf_content: bytes, pdf_filename: str) -> Dict:
    """Extract strategy, governance, risk and audit content from PDF"""

    accounts_data = {
        "filename": pdf_filename,
        "extraction_status": "failed",
        "extraction_method": "none",
        "raw_text": "",
        "strategy_governance_risk_audit": {},
        "content_summary": {
            "strategy_found": False,
            "governance_found": False,
            "risk_found": False,
            "audit_found": False,
            "total_content_sections": 0
        },
        "debug_info": {
            "total_pages": 0,
            "text_length": 0,
            "tables_found": 0,
            "sample_text": "",
            "ocr_attempted": False
        }
    }

    # Method 1: Try pdfplumber first (best for structured documents)
    try:
        if pdfplumber is not None:
            print(f"     Attempting pdfplumber extraction...")

            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                full_text = ""
                tables_found = 0

                accounts_data["debug_info"]["total_pages"] = len(pdf.pages)
                print(f"     PDF has {len(pdf.pages)} pages")

                # Process each page with enhanced extraction
                for page_num, page in enumerate(pdf.pages):
                    page_header = f"\n{'='*50}\nPAGE {page_num + 1}\n{'='*50}\n"
                    
                    # Extract regular text with better formatting
                    page_text = page.extract_text()
                    if page_text:
                        # Clean and format the text better
                        cleaned_text = page_text.replace('\n\n\n', '\n\n').replace('\t', ' ')
                        full_text += page_header + cleaned_text + "\n"

                    # Extract tables with enhanced processing
                    tables = page.extract_tables()
                    if tables:
                        tables_found += len(tables)
                        print(f"     Found {len(tables)} tables on page {page_num + 1}")

                        for table_num, table in enumerate(tables):
                            table_text = process_financial_table(table)
                            if table_text:
                                table_header = f"\n--- TABLE {page_num + 1}.{table_num + 1} ---\n"
                                full_text += table_header + table_text + "\n"

                accounts_data["debug_info"]["text_length"] = len(full_text)
                accounts_data["debug_info"]["tables_found"] = tables_found

                print(f"     Extracted {len(full_text)} characters of text")
                print(f"     Found {tables_found} tables total")

                if len(full_text) > 200:  # Reasonable amount of text extracted
                    accounts_data["raw_text"] = full_text
                    accounts_data["extraction_status"] = "success"
                    accounts_data["extraction_method"] = "pdfplumber"
                    accounts_data["debug_info"]["sample_text"] = full_text[:500]

                    # Extract strategy, governance, risk, audit content
                    content_findings = extract_strategy_governance_risk_audit(full_text)
                    accounts_data["strategy_governance_risk_audit"] = content_findings
                    accounts_data["content_summary"] = content_findings["summary"]

                    # Count total content sections found
                    total_sections = (
                        len(content_findings["strategy_content"]) +
                        len(content_findings["governance_content"]) +
                        len(content_findings["risk_content"]) +
                        len(content_findings["audit_content"])
                    )
                    accounts_data["content_summary"]["total_content_sections"] = total_sections

                    print(f"     pdfplumber extraction successful - {total_sections} content sections found")
                    return accounts_data
                else:
                    print(f"     pdfplumber extracted minimal text ({len(full_text)} chars)")

    except Exception as e:
        print(f"     pdfplumber extraction failed: {e}")

    # Method 2: Try PyPDF2 as fallback with enhanced extraction
    try:
        if PyPDF2 is not None:
            print(f"     Attempting PyPDF2 extraction...")

            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            full_text = ""

            accounts_data["debug_info"]["total_pages"] = len(pdf_reader.pages)
            print(f"     PDF has {len(pdf_reader.pages)} pages")

            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_header = f"\n{'='*50}\nPAGE {page_num + 1}\n{'='*50}\n"
                    page_text = page.extract_text()
                    if page_text:
                        # Clean and format the text better
                        cleaned_text = page_text.replace('\n\n\n', '\n\n').replace('\t', ' ')
                        full_text += page_header + cleaned_text + "\n"
                except Exception as e:
                    print(f"     Error extracting page {page_num + 1}: {e}")

            accounts_data["debug_info"]["text_length"] = len(full_text)
            print(f"     Extracted {len(full_text)} characters of text")

            if len(full_text) > 200:  # Reasonable amount of text extracted
                accounts_data["raw_text"] = full_text
                accounts_data["extraction_status"] = "success"
                accounts_data["extraction_method"] = "PyPDF2"
                accounts_data["debug_info"]["sample_text"] = full_text[:500]

                # Extract strategy, governance, risk, audit content
                content_findings = extract_strategy_governance_risk_audit(full_text)
                accounts_data["strategy_governance_risk_audit"] = content_findings
                accounts_data["content_summary"] = content_findings["summary"]

                # Count total content sections found
                total_sections = (
                    len(content_findings["strategy_content"]) +
                    len(content_findings["governance_content"]) +
                    len(content_findings["risk_content"]) +
                    len(content_findings["audit_content"])
                )
                accounts_data["content_summary"]["total_content_sections"] = total_sections

                print(f"     PyPDF2 extraction successful - {total_sections} content sections found")
                return accounts_data
            else:
                print(f"     PyPDF2 extracted minimal text ({len(full_text)} chars)")

    except Exception as e:
        print(f"     PyPDF2 extraction failed: {e}")

    # Method 3: Try OCR as last resort with enhanced processing
    try:
        if pytesseract is not None and convert_from_bytes is not None:
            print(f"     Attempting OCR extraction...")
            accounts_data["debug_info"]["ocr_attempted"] = True

            # Convert PDF to images with better quality
            images = convert_from_bytes(pdf_content, dpi=300, first_page=1, last_page=min(50, 50))  # Increased to 50 pages

            full_text = ""
            for page_num, image in enumerate(images):
                try:
                    page_header = f"\n{'='*50}\nPAGE {page_num + 1} (OCR)\n{'='*50}\n"
                    # Use OCR to extract text with better configuration
                    page_text = pytesseract.image_to_string(
                        image, 
                        lang='eng',
                        config='--psm 6 --oem 3'  # Better OCR configuration
                    )
                    if page_text and page_text.strip():
                        cleaned_text = page_text.replace('\n\n\n', '\n\n').replace('\t', ' ')
                        full_text += page_header + cleaned_text + "\n"
                except Exception as e:
                    print(f"     OCR error on page {page_num + 1}: {e}")

            accounts_data["debug_info"]["text_length"] = len(full_text)
            print(f"     OCR extracted {len(full_text)} characters of text")

            if len(full_text) > 200:  # Reasonable amount of text extracted
                accounts_data["raw_text"] = full_text
                accounts_data["extraction_status"] = "success"
                accounts_data["extraction_method"] = "OCR"
                accounts_data["debug_info"]["sample_text"] = full_text[:500]

                # Extract strategy, governance, risk, audit content
                content_findings = extract_strategy_governance_risk_audit(full_text)
                accounts_data["strategy_governance_risk_audit"] = content_findings
                accounts_data["content_summary"] = content_findings["summary"]

                # Count total content sections found
                total_sections = (
                    len(content_findings["strategy_content"]) +
                    len(content_findings["governance_content"]) +
                    len(content_findings["risk_content"]) +
                    len(content_findings["audit_content"])
                )
                accounts_data["content_summary"]["total_content_sections"] = total_sections

                print(f"     OCR extraction successful - {total_sections} content sections found")
                return accounts_data
            else:
                print(f"     OCR extracted minimal text ({len(full_text)} chars)")

    except Exception as e:
        print(f"     OCR extraction failed: {e}")

    # If all methods failed
    print(f"     All extraction methods failed for {pdf_filename}")
    accounts_data["extraction_status"] = "failed"
    accounts_data["extraction_method"] = "none"
    return accounts_data

def analyze_accounts_portfolio(download_results: Dict) -> Dict:
    """Analyze downloaded accounts files for strategy, governance, risk and audit content"""

    print(f"\nAnalyzing {len(download_results['downloaded_files'])} downloaded account files...")

    portfolio_analysis = {
        "company_number": download_results["company_number"],
        "analysis_timestamp": datetime.now().isoformat(),
        "files_analyzed": 0,
        "files_successful": 0,
        "files_failed": 0,
        "total_content_sections": 0,
        "file_analyses": [],
        "consolidated_findings": {
            "strategy_content": [],
            "governance_content": [],
            "risk_content": [],
            "audit_content": []
        }
    }

    extracted_accounts = []

    for file_info in download_results["downloaded_files"]:
        file_path = file_info["filepath"]
        filename = file_info["filename"]
        filing_date = file_info["filing_date"]

        print(f"\nAnalyzing: {filename}")
        portfolio_analysis["files_analyzed"] += 1

        try:
            # Read PDF content
            with open(file_path, "rb") as f:
                pdf_content = f.read()

            # Extract content from PDF
            extraction_result = extract_accounts_content_from_pdf(pdf_content, filename)

            if extraction_result["extraction_status"] == "success":
                portfolio_analysis["files_successful"] += 1
                portfolio_analysis["total_content_sections"] += extraction_result["content_summary"]["total_content_sections"]

                # Add filing date to the analysis
                extraction_result["filing_date"] = filing_date

                # Collect text for consolidated analysis
                combined_content = ""
                for category in ["strategy_content", "governance_content", "risk_content", "audit_content"]:
                    for item in extraction_result["strategy_governance_risk_audit"].get(category, []):
                        combined_content += item["content"] + "\n"
                
                # Also include the full raw text for better analysis
                if extraction_result.get("raw_text"):
                    combined_content += "\n\nFULL DOCUMENT TEXT:\n" + extraction_result["raw_text"]
                
                if combined_content.strip():
                    extracted_accounts.append({
                        "filing_date": filing_date,
                        "full_text": combined_content
                    })

                # Add to consolidated findings
                for category in ["strategy_content", "governance_content", "risk_content", "audit_content"]:
                    for item in extraction_result["strategy_governance_risk_audit"].get(category, []):
                        portfolio_analysis["consolidated_findings"][category].append(item)

                portfolio_analysis["file_analyses"].append(extraction_result)
                print(f"     Analysis complete - {extraction_result['content_summary']['total_content_sections']} sections found")
            else:
                portfolio_analysis["files_failed"] += 1
                portfolio_analysis["file_analyses"].append(extraction_result)
                print(f"     Analysis failed: {extraction_result.get('error', 'Unknown error')}")

        except Exception as e:
            portfolio_analysis["files_failed"] += 1
            error_result = {
                "filename": filename,
                "extraction_status": "error",
                "error": str(e)
            }
            portfolio_analysis["file_analyses"].append(error_result)
            print(f"     File processing error: {e}")

    # CONSOLIDATED ANALYSIS WITH CHUNKING SUPPORT
    if extracted_accounts:
        print(f"\nRunning consolidated evolution analysis on all {len(extracted_accounts)} documents...")
        
        # Sort by filing date
        extracted_accounts.sort(key=lambda x: x["filing_date"])
        
        # Combine all text with filing dates
        combined_text = ""
        for entry in extracted_accounts:
            filing_date = entry["filing_date"]
            full_text = entry["full_text"]
            combined_text += f"\n--- FILING DATE: {filing_date} ---\n{full_text}\n"
        
        if combined_text.strip():
            print("Making evolution analysis call...")
            
            # Save the combined text for debugging
            debug_path = os.path.join("strategic_analysis_results", f"{download_results['company_number']}_combined_content.txt")
            try:
                with open(debug_path, "w", encoding="utf-8") as f:
                    f.write(combined_text)
                print(f"Combined content saved for debugging: {debug_path}")
            except Exception as e:
                print(f"Could not save debug content: {e}")
            
            # This will now automatically handle chunking if content is too large
            evolution_result = analyze_evolution(combined_text, download_results["company_number"])
            portfolio_analysis["evolution_analysis"] = evolution_result
        else:
            print("No content available for evolution analysis")

    print(f"\nPortfolio analysis complete:")
    print(f"   Files analyzed: {portfolio_analysis['files_analyzed']}")
    print(f"   Successful extractions: {portfolio_analysis['files_successful']}")
    print(f"   Failed extractions: {portfolio_analysis['files_failed']}")
    print(f"   Total content sections: {portfolio_analysis['total_content_sections']}")

    return portfolio_analysis

def save_analysis_results(firm_name: str, company_number: str, analysis_data: Dict, filename_suffix: str = "") -> str:
    """Save analysis results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if filename_suffix:
        filename = f"{company_number}_{filename_suffix}_{timestamp}.json"
    else:
        filename = f"{company_number}_analysis_{timestamp}.json"
        
    filepath = RESULTS_FOLDER / filename
    
    # Add firm name and company number to the data
    analysis_data["firm_name"] = firm_name
    analysis_data["company_number"] = company_number
    analysis_data["analysis_timestamp"] = timestamp
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"Analysis results saved to: {filepath}")
        return str(filepath)
    except Exception as e:
        print(f"Error saving analysis results: {e}")
        return ""

def generate_analysis_report(portfolio_analysis: Dict) -> str:
    """Generate a comprehensive analysis report in plain text format"""
    company_number = portfolio_analysis["company_number"]
    files_successful = portfolio_analysis["files_successful"]
    total_sections = portfolio_analysis["total_content_sections"]
    
    # Get company name from Companies House
    company_profile = get_company_profile(company_number)
    company_name = company_profile.get("company_name", "Unknown Company")
    
    # Format analysis date as dd/mm/yyyy
    analysis_timestamp = portfolio_analysis.get('analysis_timestamp')
    if analysis_timestamp:
        try:
            # Handle the case where timestamp might be a datetime object or string
            if isinstance(analysis_timestamp, str):
                # Try to parse ISO format first
                try:
                    # Remove timezone info if present
                    if 'T' in analysis_timestamp:
                        analysis_date = datetime.fromisoformat(analysis_timestamp.replace('Z', '+00:00').split('+')[0])
                    else:
                        # Try the timestamp format we're using (YYYYMMDD_HHMMSS)
                        analysis_date = datetime.strptime(analysis_timestamp, "%Y%m%d_%H%M%S")
                except ValueError:
                    # If parsing fails, use current time
                    analysis_date = datetime.now()
            else:
                analysis_date = analysis_timestamp
        except Exception:
            analysis_date = datetime.now()
    else:
        analysis_date = datetime.now()
    
    formatted_date = analysis_date.strftime("%d/%m/%Y")
    
    report = f"""EXSIGNA STRATEGY & GOVERNANCE REPORT
====================================
Company Name: {company_name}
Company Number: {company_number}
Analysis Date: {formatted_date}

EXECUTIVE SUMMARY
================
Files Analyzed: {portfolio_analysis['files_analyzed']}
Successful Extractions: {files_successful}
Total Content Sections: {total_sections}

"""

    # Add evolution analysis results if available
    if portfolio_analysis.get("evolution_analysis"):
        evolution_data = portfolio_analysis["evolution_analysis"]
        if "error" not in evolution_data:
            report += f"""STRATEGIC EVOLUTION ANALYSIS
============================
Primary Business Strategy: {evolution_data.get('business_primary', 'Unknown')}
Secondary Business Strategy: {evolution_data.get('business_secondary', 'Unknown')}
Primary Risk Strategy: {evolution_data.get('risk_primary', 'Unknown')}
Secondary Risk Strategy: {evolution_data.get('risk_secondary', 'Unknown')}

Business Strategy Rationale:
{evolution_data.get('business_rationale', 'No rationale provided')}

Risk Strategy Rationale:
{evolution_data.get('risk_rationale', 'No rationale provided')}

Evolution Summary:
{evolution_data.get('evolution_summary', 'No evolution summary provided')}

"""
            # Add metadata about analysis method
            if evolution_data.get("analysis_metadata"):
                metadata = evolution_data["analysis_metadata"]
                report += f"""ANALYSIS METADATA
================
Analysis Method: {metadata.get('analysis_method', 'unknown')}
Content Size: {metadata.get('total_estimated_tokens', 'unknown'):,} estimated tokens
"""
                if metadata.get('analysis_method') == 'chunked':
                    report += f"Chunks Processed: {metadata.get('chunks_processed', 0)}\n"
                    report += f"Chunks Successful: {metadata.get('chunks_successful', 0)}\n"
                report += "\n"
        else:
            report += f"""STRATEGIC EVOLUTION ANALYSIS
============================
Error in analysis: {evolution_data.get('error', 'Unknown error')}
Raw response (if available): {evolution_data.get('raw_response', 'No response')}

"""
    else:
        report += """STRATEGIC EVOLUTION ANALYSIS
============================
No evolution analysis available - insufficient content for classification.

"""

    report += """TECHNICAL DETAILS
================
Analysis generated using: Strategy, Governance, Risk & Audit Analysis Tool (Chunked Version)
PDF extraction methods: pdfplumber, PyPDF2, OCR (as needed)
AI archetype classification: OpenAI GPT-4o (Automatic Chunking for Large Content)
"""
    
    return report

def main_analysis_workflow(company_number: str, max_years: int = 5) -> Dict:
    """Main workflow to download and analyze company accounts"""
    print(f"\nStarting comprehensive analysis for company {company_number}")
    print(f"Analyzing last {max_years} years of filings")
    
    # Get company name for saving results
    company_profile = get_company_profile(company_number)
    firm_name = company_profile.get("company_name", "Unknown Company")
    
    # Step 1: Download accounts
    print("\n" + "="*60)
    print("STEP 1: Downloading Company Accounts")
    print("="*60)
    
    download_results = download_company_accounts(company_number, max_years)
    
    if download_results["download_count"] == 0 and not download_results["downloaded_files"]:
        print("No accounts found or downloaded. Cannot proceed with analysis.")
        return {"error": "No accounts available for analysis"}
    
    # Step 2: Analyze accounts content
    print("\n" + "="*60)
    print("STEP 2: Analyzing Account Content")
    print("="*60)
    
    portfolio_analysis = analyze_accounts_portfolio(download_results)
    
    # Clean up downloaded files immediately after processing
    download_dir = download_results.get("save_directory")
    if download_dir and os.path.exists(download_dir):
        import shutil
        try:
            shutil.rmtree(download_dir)
            print(f"Cleaned up temporary directory: {download_dir}")
        except Exception as e:
            print(f"Could not delete temporary directory: {e}")
    
    # Step 3: Generate and save results
    print("\n" + "="*60)
    print("STEP 3: Generating Results")
    print("="*60)
    
    # Save detailed analysis
    analysis_filepath = save_analysis_results(firm_name, company_number, portfolio_analysis, "portfolio_analysis")
    
    # Generate readable report
    analysis_report = generate_analysis_report(portfolio_analysis)
    
    # Save report as plain text
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{company_number}_report_{timestamp}.txt"
    report_filepath = RESULTS_FOLDER / report_filename
    
    try:
        with open(report_filepath, "w", encoding="utf-8") as f:
            f.write(analysis_report)
        print(f"Analysis report saved to: {report_filepath}")
    except Exception as e:
        print(f"Error saving report: {e}")
    
    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Successfully analyzed {portfolio_analysis['files_successful']} files")
    print(f"Found {portfolio_analysis['total_content_sections']} content sections")
    print(f"Results saved to: {analysis_filepath}")
    print(f"Report saved to: {report_filepath}")
    
    return {
        "portfolio_analysis": portfolio_analysis,
        "analysis_filepath": analysis_filepath,
        "report_filepath": str(report_filepath),
        "summary": {
            "files_successful": portfolio_analysis['files_successful'],
            "total_content_sections": portfolio_analysis['total_content_sections'],
            "analysis_timestamp": portfolio_analysis['analysis_timestamp']
        }
    }

# Example usage and testing
if __name__ == "__main__":
    print("Strategy, Governance, Risk & Audit Analysis Tool (Fixed Version)")
    print("=" * 65)
    print("✅ Includes automatic chunking to handle large documents")
    print("✅ Prevents OpenAI API token limit errors")
    print("✅ Maintains strategic analysis quality")
    print()
    
    # Interactive mode
    try:
        company_number = input("Enter company number to analyze: ").strip()
        if company_number:
            max_years = input("Enter max years to analyze (default 5): ").strip()
            max_years = int(max_years) if max_years.isdigit() else 5
            
            result = main_analysis_workflow(company_number, max_years)
            
            if "error" not in result:
                print(f"Analysis complete for company {company_number}")
                # Print evolution results if available
                evolution = result.get("portfolio_analysis", {}).get("evolution_analysis", {})
                if evolution and "error" not in evolution:
                    print(f"✅ Strategic classification successful:")
                    print(f"   Business: {evolution.get('business_primary', 'Unknown')}")
                    print(f"   Risk: {evolution.get('risk_primary', 'Unknown')}")
                    if evolution.get("analysis_metadata"):
                        method = evolution["analysis_metadata"].get("analysis_method", "unknown")
                        print(f"   Analysis method: {method}")
            else:
                print(f"Analysis failed: {result['error']}")
        else:
            print("No company number provided. Exiting.")
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Error in main workflow: {e}")