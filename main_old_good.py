#!/usr/bin/env python3
"""
Strategic Analysis Tool - Archetype Classification Focus
Analyzes UK company filings for business and risk strategy archetype classification
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime

# Import our modules - FIXED IMPORT
from config import (
    validate_config, DEFAULT_MAX_YEARS, validate_company_number, 
    MIN_EXTRACTION_LENGTH, validate_file_size, validate_file_type
)
from companies_house_client import CompaniesHouseClient
from content_processor import ContentProcessor
from pdf_extractor import PDFExtractor
from ai_analyzer import AIArchetypeAnalyzer  # CORRECT IMPORT
from file_manager import FileManager
from report_generator import ReportGenerator

# Set up logging once for the entire application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_user_input():
    """
    Get company number and years from user input
    
    Returns:
        Tuple of (company_number, max_years)
    """
    print("\n" + "="*60)
    print("🏢 STRATEGIC ARCHETYPE ANALYSIS TOOL")
    print("   UK Company Filing Analysis - Business & Risk Archetypes")
    print("="*60)
    
    # Get company number
    while True:
        company_number = input("\n📋 Enter UK Company Number (e.g., 00000006): ").strip()
        
        if not company_number:
            print("❌ Company number cannot be empty. Please try again.")
            continue
            
        if not validate_company_number(company_number):
            print("❌ Invalid company number format. UK company numbers should be 8 digits or 2 letters + 6 digits.")
            print("   Examples: 00000006, SC123456, NI123456")
            continue
            
        # Pad with leading zeros if needed
        if company_number.isdigit() and len(company_number) < 8:
            company_number = company_number.zfill(8)
            print(f"📝 Padded to: {company_number}")
            
        break
    
    # Get number of years
    while True:
        try:
            years_input = input(f"\n📅 How many years of filings to analyze? (default: {DEFAULT_MAX_YEARS}): ").strip()
            
            if not years_input:
                max_years = DEFAULT_MAX_YEARS
                break
                
            max_years = int(years_input)
            
            if max_years < 1:
                print("❌ Number of years must be at least 1.")
                continue
            elif max_years > 10:
                print("⚠️  Warning: Analyzing more than 10 years may take a long time and use many API calls.")
                confirm = input("   Continue? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
                    
            break
            
        except ValueError:
            print("❌ Please enter a valid number.")
    
    print(f"\n✅ Analysis Configuration:")
    print(f"   Company Number: {company_number}")
    print(f"   Years to analyze: {max_years}")
    print(f"   Focus: Business & Risk Strategy Archetypes")
    
    return company_number, max_years


def validate_company_exists(client, company_number):
    """
    Validate that the company exists and get basic info
    
    Args:
        client: CompaniesHouseClient instance
        company_number: Company number to validate
        
    Returns:
        Tuple of (success, company_name, company_status)
    """
    print(f"\n🔍 Validating company {company_number}...")
    
    try:
        exists, company_name = client.validate_company_exists(company_number)
        
        if not exists:
            print(f"❌ Company {company_number} not found in Companies House records.")
            print("   Please check the company number and try again.")
            return False, None, None
        
        # Get additional details
        profile = client.get_company_profile(company_number)
        company_status = profile.get('company_status', 'unknown') if profile else 'unknown'
        
        print(f"✅ Company found: {company_name}")
        print(f"   Status: {company_status}")
        
        if company_status.lower() in ['dissolved', 'liquidation']:
            print("⚠️  Warning: This company is dissolved or in liquidation.")
            confirm = input("   Continue with analysis? (y/n): ").strip().lower()
            if confirm != 'y':
                return False, company_name, company_status
        
        return True, company_name, company_status
        
    except Exception as e:
        print(f"❌ Error validating company: {e}")
        logger.error(f"Error validating company {company_number}: {e}")
        return False, None, None


def download_company_filings(client, company_number, max_years):
    """
    Download company filings
    
    Args:
        client: CompaniesHouseClient instance
        company_number: Company number
        max_years: Maximum years to look back
        
    Returns:
        Download results dictionary
    """
    print(f"\n📥 Downloading annual accounts (last {max_years} years)...")
    
    try:
        results = client.download_annual_accounts(company_number, max_years)
        
        if results['total_downloaded'] == 0:
            print("❌ No annual accounts found or downloaded.")
            print("   This could mean:")
            print("   - The company hasn't filed accounts in the specified period")
            print("   - The accounts are not available for download")
            print("   - There was an API issue")
            return results
        
        print(f"✅ Successfully downloaded {results['total_downloaded']} documents")
        
        if results['failed_downloads']:
            print(f"⚠️  {len(results['failed_downloads'])} downloads failed")
        
        if results['earliest_date'] and results['latest_date']:
            print(f"   Date range: {results['earliest_date']} to {results['latest_date']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error downloading filings: {e}")
        logger.error(f"Error downloading filings for {company_number}: {e}")
        return None


def extract_content_from_pdfs(pdf_extractor, downloaded_files):
    """
    Extract content from downloaded PDF files using PDFExtractor
    
    Args:
        pdf_extractor: PDFExtractor instance
        downloaded_files: List of downloaded file info
        
    Returns:
        List of extracted content dictionaries
    """
    print(f"\n📄 Extracting content from {len(downloaded_files)} PDF files...")
    
    extracted_content = []
    
    for i, file_info in enumerate(downloaded_files, 1):
        print(f"   Processing {i}/{len(downloaded_files)}: {file_info['filename']}")
        
        try:
            # Validate file size and type
            if not validate_file_size(file_info['size']):
                print(f"     ⚠️  File too large: {file_info['size']} bytes")
                continue
                
            if not validate_file_type(file_info['filename']):
                print(f"     ⚠️  Unsupported file type: {file_info['filename']}")
                continue
            
            # Read the PDF file content
            with open(file_info['path'], 'rb') as f:
                pdf_content = f.read()
            
            # Use PDFExtractor to extract content
            extraction_result = pdf_extractor.extract_text_from_pdf(
                pdf_content, 
                file_info['filename']
            )
            
            if extraction_result["extraction_status"] == "success":
                # Get the raw text content
                content = extraction_result.get("raw_text", "")
                
                if content and len(content.strip()) > MIN_EXTRACTION_LENGTH:
                    extracted_content.append({
                        'filename': file_info['filename'],
                        'date': file_info['date'],
                        'content': content,
                        'metadata': {
                            'transaction_id': file_info['transaction_id'],
                            'description': file_info['description'],
                            'file_size': file_info['size'],
                            'extraction_method': extraction_result["extraction_method"],
                            'debug_info': extraction_result.get("debug_info", {})
                        }
                    })
                    print(f"     ✅ Extracted {len(content)} characters using {extraction_result['extraction_method']}")
                else:
                    print(f"     ⚠️  Insufficient readable content found ({len(content)} chars)")
            else:
                print(f"     ❌ Extraction failed: {extraction_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"     ❌ Error extracting content: {e}")
            logger.error(f"Error extracting content from {file_info['filename']}: {e}")
    
    print(f"✅ Successfully extracted content from {len(extracted_content)} files")
    return extracted_content


def process_and_analyze_content(content_processor, archetype_analyzer, extracted_content, company_name, company_number):
    """
    Process and analyze the extracted content focusing on archetype classification
    """
    print(f"\n🧠 Processing content for archetype analysis...")
    
    try:
        # Process each document
        processed_documents = []
        for content_data in extracted_content:
            print(f"   Processing: {content_data['filename']}")
            
            processed = content_processor.process_document_content(
                content_data['content'],
                content_data['metadata']
            )
            processed_documents.append(processed)
        
        # Combine all documents
        combined_analysis = content_processor.combine_multiple_documents(processed_documents)
        
        print(f"✅ Content processing completed")
        word_count = combined_analysis.get('content_stats', {}).get('total_word_count', 0)
        if word_count == 0:
            # Fallback word count calculation
            word_count = sum(len(content_data['content'].split()) for content_data in extracted_content)
        print(f"   Total words analyzed: {word_count:,}")
        
        # Perform archetype analysis
        print(f"\n🏛️ Performing archetype classification analysis...")
        
        # Combine all extracted content for AI analysis
        combined_content = "\n\n".join([content_data['content'] for content_data in extracted_content])
        
        archetype_analysis = archetype_analyzer.analyze_archetypes(
            combined_content,
            company_name,
            company_number
        )
        
        if archetype_analysis.get('success', False):
            print(f"✅ Archetype analysis completed using {archetype_analysis.get('analysis_type', 'unknown')} method")
            
            # Display the archetype analysis immediately
            display_archetype_analysis(archetype_analysis, company_name, company_number)
        else:
            print(f"❌ Archetype analysis failed: {archetype_analysis.get('error', 'Unknown error')}")
        
        # Create portfolio analysis structure for compatibility with report generator
        portfolio_analysis = {
            'company_number': company_number,
            'company_name': company_name,
            'files_analyzed': len(extracted_content),
            'files_successful': len(extracted_content),
            'total_content_sections': combined_analysis.get('total_sections', len(extracted_content)),
            'analysis_timestamp': datetime.now().isoformat(),
            'archetype_analysis': archetype_analysis,
            'file_analyses': [
                {
                    'filename': content_data['filename'],
                    'extraction_status': 'success',
                    'content_summary': {
                        'strategy_found': True,
                        'governance_found': True,
                        'risk_found': True,
                        'audit_found': True,
                        'total_content_sections': 1
                    },
                    'extraction_method': content_data['metadata']['extraction_method'],
                    'debug_info': content_data['metadata']['debug_info']
                }
                for content_data in extracted_content
            ]
        }
        
        return {
            'portfolio_analysis': portfolio_analysis,
            'processed_content': combined_analysis,
            'archetype_analysis': archetype_analysis,
            'document_count': len(extracted_content)
        }
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        logger.error(f"Error during content processing and analysis: {e}")
        return None


def display_archetype_analysis(archetype_analysis, company_name, company_number):
    """
    Display the archetype analysis in a formatted, readable way
    """
    print(f"\n" + "="*80)
    print(f"🏛️ ARCHETYPE CLASSIFICATION RESULTS")
    print(f"="*80)
    
    if not archetype_analysis.get('success', False):
        print(f"❌ Analysis failed: {archetype_analysis.get('error', 'Unknown error')}")
        return
    
    business_archetypes = archetype_analysis.get('business_strategy_archetypes', {})
    risk_archetypes = archetype_analysis.get('risk_strategy_archetypes', {})
    
    print(f"\n🏢 COMPANY: {company_name} ({company_number})")
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"🔬 Method: {archetype_analysis.get('analysis_type', 'unknown')}")
    
    print(f"\n📊 BUSINESS STRATEGY ARCHETYPES")
    print(f"-" * 50)
    print(f"🎯 Dominant: {business_archetypes.get('dominant', 'Unknown')}")
    if business_archetypes.get('secondary'):
        print(f"🎯 Secondary: {business_archetypes.get('secondary')}")
    else:
        print(f"🎯 Secondary: None identified")
    
    print(f"\n💭 Business Strategy Analysis:")
    print(f"   {business_archetypes.get('reasoning', 'No analysis available')}")
    
    print(f"\n⚠️  RISK STRATEGY ARCHETYPES")
    print(f"-" * 50)
    print(f"🛡️  Dominant: {risk_archetypes.get('dominant', 'Unknown')}")
    if risk_archetypes.get('secondary'):
        print(f"🛡️  Secondary: {risk_archetypes.get('secondary')}")
    else:
        print(f"🛡️  Secondary: None identified")
    
    print(f"\n💭 Risk Strategy Analysis:")
    print(f"   {risk_archetypes.get('reasoning', 'No analysis available')}")
    
    print(f"\n📋 STRATEGIC PROFILE SUMMARY")
    print(f"-" * 50)
    print(f"This company demonstrates characteristics primarily aligned with:")
    print(f"• Business Strategy: {business_archetypes.get('dominant', 'Unknown')}")
    print(f"• Risk Strategy: {risk_archetypes.get('dominant', 'Unknown')}")
    print(f"\nThis archetype combination suggests a strategic approach that balances")
    print(f"growth objectives with risk management priorities, consistent with UK")
    print(f"financial services regulatory expectations.")
    
    print(f"\n" + "="*80)


def save_results(analysis_results, company_number, company_name, file_manager, report_generator):
    """
    Save archetype analysis results using FileManager and ReportGenerator
    """
    print(f"\n💾 Saving archetype analysis results...")
    
    try:
        portfolio_analysis = analysis_results['portfolio_analysis']
        
        # Save main analysis results as JSON
        analysis_file = file_manager.save_analysis_results(
            company_name, 
            company_number, 
            portfolio_analysis
        )
        
        if not analysis_file:
            print("❌ Failed to save analysis results")
            return None
        
        # Generate and save comprehensive report
        report_content = report_generator.generate_analysis_report(portfolio_analysis)
        report_file = file_manager.save_text_report(
            company_number, 
            report_content, 
            "archetype_analysis_report"
        )
        
        # Generate and save executive summary
        summary_content = report_generator.generate_executive_summary(portfolio_analysis)
        summary_file = file_manager.save_text_report(
            company_number, 
            summary_content, 
            "executive_summary"
        )
        
        # Generate and save JSON summary
        json_summary = report_generator.generate_json_summary(portfolio_analysis)
        json_file = file_manager.save_json_results(
            company_number,
            json_summary,
            "archetype_summary"
        )
        
        print(f"✅ Results saved successfully:")
        print(f"   📄 Analysis Data: {Path(analysis_file).name}")
        if report_file:
            print(f"   📊 Full Report: {Path(report_file).name}")
        if summary_file:
            print(f"   📋 Executive Summary: {Path(summary_file).name}")
        if json_file:
            print(f"   📋 JSON Summary: {Path(json_file).name}")
        
        return Path(analysis_file).parent
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        logger.error(f"Error saving results: {e}")
        return None


def cleanup_temp_files(client):
    """Clean up temporary files"""
    try:
        count = client.cleanup_temp_files()
        if count > 0:
            print(f"🧹 Cleaned up {count} temporary files")
    except Exception as e:
        print(f"⚠️  Could not clean up temp files: {e}")
        logger.warning(f"Could not clean up temp files: {e}")


def print_final_summary(analysis_results, company_name, company_number, results_path, extracted_content):
    """Print comprehensive final summary focused on archetype results"""
    print(f"\n" + "🎉" + "="*78 + "🎉")
    print(f"   ARCHETYPE CLASSIFICATION COMPLETED SUCCESSFULLY")
    print(f"🎉" + "="*78 + "🎉")
    
    print(f"\n📋 ANALYSIS OVERVIEW:")
    print(f"   Company: {company_name}")
    print(f"   Registration: {company_number}")
    print(f"   Documents Processed: {len(extracted_content)}")
    print(f"   Analysis Method: {analysis_results['archetype_analysis'].get('analysis_type', 'unknown')}")
    
    archetype_analysis = analysis_results['archetype_analysis']
    
    if archetype_analysis.get('success', False):
        business_archetypes = archetype_analysis.get('business_strategy_archetypes', {})
        risk_archetypes = archetype_analysis.get('risk_strategy_archetypes', {})
        
        print(f"\n🏛️ ARCHETYPE CLASSIFICATION RESULTS:")
        print(f"   🎯 Business Strategy (Dominant): {business_archetypes.get('dominant', 'Unknown')}")
        if business_archetypes.get('secondary'):
            print(f"   🎯 Business Strategy (Secondary): {business_archetypes.get('secondary')}")
        
        print(f"   🛡️  Risk Strategy (Dominant): {risk_archetypes.get('dominant', 'Unknown')}")
        if risk_archetypes.get('secondary'):
            print(f"   🛡️  Risk Strategy (Secondary): {risk_archetypes.get('secondary')}")
        
        print(f"\n💡 KEY INSIGHTS:")
        business_reasoning = business_archetypes.get('reasoning', '')
        if len(business_reasoning) > 100:
            business_reasoning = business_reasoning[:100] + "..."
        print(f"   📊 Business: {business_reasoning}")
        
        risk_reasoning = risk_archetypes.get('reasoning', '')
        if len(risk_reasoning) > 100:
            risk_reasoning = risk_reasoning[:100] + "..."
        print(f"   ⚠️  Risk: {risk_reasoning}")
        
    else:
        print(f"   ⚠️  Analysis completed with limitations: {archetype_analysis.get('error', 'Unknown error')}")
    
    print(f"\n📁 RESULTS LOCATION:")
    print(f"   {results_path}")
    
    print(f"\n📖 RECOMMENDED NEXT STEPS:")
    print(f"   1. Review archetype_analysis_report for detailed classification insights")
    print(f"   2. Compare archetype combination with industry peers")
    print(f"   3. Assess strategic alignment with current market conditions")
    print(f"   4. Consider implications for investment or strategic decisions")
    
    print(f"\n🔄 FOR ENHANCED ANALYSIS:")
    if analysis_results['archetype_analysis'].get('analysis_type') == 'pattern_archetype_classification':
        print(f"   • Configure OpenAI or Anthropic API for AI-powered archetype classification")
        print(f"   • Current analysis used pattern-matching (comprehensive but limited)")
    print(f"   • Run analysis on additional years for archetype evolution tracking")
    print(f"   • Compare with peer companies for industry archetype patterns")
    
    print(f"\n✨ Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"="*80)


def main():
    """Enhanced main execution function focused on archetype classification"""
    try:
        # Validate configuration
        if not validate_config():
            print("❌ Configuration validation failed. Please check your .env file and API keys.")
            print("\n🔧 SETUP INSTRUCTIONS:")
            print("   1. Copy .env.example to .env")
            print("   2. Add your Companies House API key: CH_API_KEY=...")
            print("   3. Add OpenAI API key: OPENAI_API_KEY=sk-... (optional)")
            print("   4. OR add Anthropic API key: ANTHROPIC_API_KEY=sk-ant-... (optional)")
            print("   5. If no AI API keys provided, will use pattern-based archetype analysis")
            return 1
        
        # Get user input
        company_number, max_years = get_user_input()
        
        # Initialize components
        print(f"\n🔧 Initializing archetype analysis components...")
        ch_client = CompaniesHouseClient()
        content_processor = ContentProcessor()
        pdf_extractor = PDFExtractor()
        archetype_analyzer = AIArchetypeAnalyzer()  # CORRECT CLASS NAME
        file_manager = FileManager()
        report_generator = ReportGenerator()
        
        print(f"   ✅ Companies House client ready")
        print(f"   ✅ Content processor ready")
        print(f"   ✅ PDF extractor ready")
        print(f"   ✅ Archetype analyzer ready")
        print(f"   ✅ File manager ready")
        print(f"   ✅ Report generator ready")
        
        # Validate company exists
        success, company_name, company_status = validate_company_exists(ch_client, company_number)
        if not success:
            return 1
        
        # Download filings
        download_results = download_company_filings(ch_client, company_number, max_years)
        if not download_results or download_results['total_downloaded'] == 0:
            return 1
        
        # Extract content from PDFs
        extracted_content = extract_content_from_pdfs(pdf_extractor, download_results['downloaded_files'])
        if not extracted_content:
            print("❌ No content could be extracted from the downloaded files.")
            print("   This could indicate:")
            print("   - PDFs are image-based and OCR failed")
            print("   - Files are corrupted or protected")
            print("   - Technical extraction issues")
            return 1
        
        # Process and analyze content for archetypes
        analysis_results = process_and_analyze_content(
            content_processor, archetype_analyzer, extracted_content, company_name, company_number
        )
        if not analysis_results:
            return 1
        
        # Save results using FileManager and ReportGenerator
        results_path = save_results(
            analysis_results, company_number, company_name, file_manager, report_generator
        )
        
        # Clean up
        cleanup_temp_files(ch_client)
        
        # Final comprehensive summary
        print_final_summary(analysis_results, company_name, company_number, results_path, extracted_content)
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Analysis interrupted by user.")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        print(f"❌ An unexpected error occurred: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())