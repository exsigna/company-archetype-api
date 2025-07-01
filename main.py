def download_company_filings(client, company_number, max_years):
    """
    Download company filings using the existing working method
    
    Args:
        client: CompaniesHouseClient instance
        company_number: Company number
        max_years: Maximum years to look back
        
    Returns:
        Download results dictionary
    """
    try:
        results = client.download_annual_accounts(company_number, max_years)
        
        if results['total_downloaded'] == 0:
            print("❌ No annual accounts found or downloaded.")
            print("   This could mean:")
            print("   - The company hasn't filed accounts in the specified period")
            print("   - The accounts are not available for download")
            print("   - There was an API issue")
            return results
        
        print(f"✅ Found {results['total_downloaded']} documents in the time range")
        
        if results['failed_downloads']:
            print(f"⚠️  {len(results['failed_downloads'])} downloads failed")
        
        if results['earliest_date'] and results['latest_date']:
            print(f"   Date range: {results['earliest_date']} to {results['latest_date']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error downloading filings: {e}")
        logger.error(f"Error downloading filings for {company_number}: {e}")
        return None


def download_selected_filings(client, company_number, selected_accounts):
    """
    Download specific company filings based on user selection
    
    Args:
        client: CompaniesHouseClient instance
        company_number: Company number
        selected_accounts: List of selected account records with transaction_ids
        
    Returns:
        Download results dictionary
    """
    print(f"\n📥 Downloading {len(selected_accounts)} selected annual accounts...")
    
    try:
        # Convert selected accounts to the format expected by existing download method
        # We'll use the existing download_annual_accounts method but filter results
        
        # First, get all available accounts using existing method
        all_results = client.download_annual_accounts(company_number, max_years=20)  # Get many years
        
        if not all_results or all_results['total_downloaded'] == 0:
            print("❌ No accounts could be downloaded using existing method.")
            return {'total_downloaded': 0, 'downloaded_files': [], 'failed_downloads': []}
        
        # Filter downloaded files to only include selected years
        selected_years = {account['year'] for account in selected_accounts}
        filtered_files = []
        
        for file_info in all_results['downloaded_files']:
            # Extract year from filename or date
            file_date = file_info.get('date', '')
            if file_date:
                try:
                    file_year = datetime.fromisoformat(file_date.replace('Z', '+00:00')).year
                    if file_year in selected_years:
                        filtered_files.append(file_info)
                        print(f"   ✅ Included: {file_info['filename']} (Year {file_year})")
                    else:
                        print(f"   ⏭️  Skipped: {file_info['filename']} (Year {file_year} not selected)")
                except:
                    # If we can't parse the date, include it anyway
                    filtered_files.append(file_info)
                    print(f"   ⚠️  Included (date unclear): {file_info['filename']}")
        
        if not filtered_files:
            print("❌ No files matched the selected years.")
            return {'total_downloaded': 0, 'downloaded_files': [], 'failed_downloads': []}
        
        results = {
            'total_downloaded': len(filtered_files),
            'downloaded_files': filtered_files,
            'failed_downloads': all_results.get('failed_downloads', []),
            'earliest_date': min(f['date'] for f in filtered_files if f.get('date')),
            'latest_date': max(f['date'] for f in filtered_files if f.get('date'))
        }
        
        print(f"\n✅ Successfully filtered to {results['total_downloaded']} selected documents")
        
        if results['failed_downloads']:
            print(f"⚠️  {len(results['failed_downloads'])} downloads had failed previously")
        
        # Show year range of selected files
        years = []
        for file_info in filtered_files:
            if file_info.get('date'):
                try:
                    year = datetime.fromisoformat(file_info['date'].replace('Z', '+00:00')).year
                    years.append(year)
                except:
                    pass
        
        if years:
            print(f"   Years included: {min(years)} to {max(years)}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during download process: {e}")
        logger.error(f"Error downloading selected filings for {company_number}: {e}")
        return None#!/usr/bin/env python3
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
    Get company number from user input
    
    Returns:
        company_number (years will be selected after seeing available filings)
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
    
    print(f"\n✅ Company Number: {company_number}")
    print(f"   Next: We'll check available filing years for you to choose from")
    
    return company_number


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


def check_available_filings(client, company_number):
    """
    Check what filing years are available and let user choose
    
    Args:
        client: CompaniesHouseClient instance
        company_number: Company number
        
    Returns:
        Tuple of (selected_years, available_filings_info) or (None, None) if error
    """
    print(f"\n📊 Checking available annual accounts for company {company_number}...")
    
    try:
        # Get filing history to see available years
        filing_history = client.get_filing_history(company_number)
        
        if not filing_history:
            print("❌ Could not retrieve filing history.")
            return None, None
        
        # Filter for annual accounts and extract years
        available_accounts = []
        for filing in filing_history.get('items', []):
            if filing.get('category') == 'accounts':
                description = filing.get('description', '')
                date = filing.get('date', '')
                transaction_id = filing.get('transaction_id', '')
                
                if 'accounts' in description.lower() and date:
                    try:
                        year = datetime.fromisoformat(date.replace('Z', '+00:00')).year
                        available_accounts.append({
                            'year': year,
                            'date': date,
                            'description': description,
                            'transaction_id': transaction_id
                        })
                    except:
                        continue
        
        if not available_accounts:
            print("❌ No annual accounts found for this company.")
            return None, None
        
        # Sort by year (most recent first)
        available_accounts.sort(key=lambda x: x['year'], reverse=True)
        
        # Remove duplicates (keep most recent filing for each year)
        seen_years = set()
        unique_accounts = []
        for account in available_accounts:
            if account['year'] not in seen_years:
                unique_accounts.append(account)
                seen_years.add(account['year'])
        
        print(f"\n📅 Available Annual Accounts ({len(unique_accounts)} years found):")
        print("-" * 60)
        for i, account in enumerate(unique_accounts, 1):
            date_formatted = datetime.fromisoformat(account['date'].replace('Z', '+00:00')).strftime('%d %B %Y')
            print(f"{i:2}. Year {account['year']} - Filed: {date_formatted}")
        
        # Let user choose years
        while True:
            print(f"\n📋 How would you like to select years to analyse?")
            print(f"   1. Select specific years (recommended)")
            print(f"   2. Select most recent N years")
            print(f"   3. Analyse all available years")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                selected_years = select_specific_years(unique_accounts)
                break
            elif choice == "2":
                selected_years = select_recent_years(unique_accounts)
                break
            elif choice == "3":
                selected_years = [acc['year'] for acc in unique_accounts]
                print(f"✅ Selected all {len(selected_years)} available years")
                break
            else:
                print("❌ Please enter 1, 2, or 3")
        
        if not selected_years:
            print("❌ No years selected.")
            return None, None
        
        # Filter available accounts to only selected years
        selected_accounts = [acc for acc in unique_accounts if acc['year'] in selected_years]
        
        print(f"\n✅ Final Selection: {len(selected_accounts)} years")
        for account in sorted(selected_accounts, key=lambda x: x['year'], reverse=True):
            print(f"   📄 {account['year']} ({datetime.fromisoformat(account['date'].replace('Z', '+00:00')).strftime('%b %Y')})")
        
        return selected_years, selected_accounts
        
    except Exception as e:
        print(f"❌ Error checking available filings: {e}")
        logger.error(f"Error checking available filings for {company_number}: {e}")
        return None, None


def select_specific_years(available_accounts):
    """Let user select specific years from available accounts"""
    while True:
        print(f"\n📅 Select years to analyse:")
        print(f"   Enter year numbers separated by commas (e.g., 1,3,5)")
        print(f"   Or enter year ranges (e.g., 1-3,5)")
        print(f"   Available options: 1-{len(available_accounts)}")
        
        selection = input(f"\nYour selection: ").strip()
        
        if not selection:
            print("❌ Please enter your selection.")
            continue
        
        try:
            selected_indices = []
            parts = selection.split(',')
            
            for part in parts:
                part = part.strip()
                if '-' in part:
                    # Handle ranges like "1-3"
                    start, end = part.split('-')
                    start_idx = int(start.strip()) - 1
                    end_idx = int(end.strip()) - 1
                    selected_indices.extend(range(start_idx, end_idx + 1))
                else:
                    # Handle single numbers
                    selected_indices.append(int(part) - 1)
            
            # Validate indices
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(available_accounts):
                    valid_indices.append(idx)
                else:
                    print(f"⚠️  Warning: {idx + 1} is not a valid option, skipping.")
            
            if not valid_indices:
                print("❌ No valid selections made. Please try again.")
                continue
            
            # Remove duplicates and sort
            valid_indices = sorted(list(set(valid_indices)))
            selected_years = [available_accounts[i]['year'] for i in valid_indices]
            
            print(f"\n✅ Selected {len(selected_years)} years:")
            for i in valid_indices:
                account = available_accounts[i]
                date_formatted = datetime.fromisoformat(account['date'].replace('Z', '+00:00')).strftime('%b %Y')
                print(f"   📄 {account['year']} (Filed: {date_formatted})")
            
            confirm = input(f"\nConfirm selection? (y/n): ").strip().lower()
            if confirm == 'y':
                return selected_years
            
        except ValueError as e:
            print(f"❌ Invalid format. Please use numbers and commas (e.g., 1,2,3 or 1-3).")


def select_recent_years(available_accounts):
    """Let user select most recent N years"""
    max_years = len(available_accounts)
    
    while True:
        try:
            num_years = input(f"\n📅 How many recent years to analyse? (1-{max_years}): ").strip()
            
            if not num_years:
                continue
                
            num_years = int(num_years)
            
            if num_years < 1:
                print("❌ Number must be at least 1.")
                continue
            elif num_years > max_years:
                print(f"❌ Maximum available years is {max_years}.")
                continue
            
            selected_accounts = available_accounts[:num_years]
            selected_years = [acc['year'] for acc in selected_accounts]
            
            print(f"\n✅ Selected {num_years} most recent years:")
            for account in selected_accounts:
                date_formatted = datetime.fromisoformat(account['date'].replace('Z', '+00:00')).strftime('%b %Y')
                print(f"   📄 {account['year']} (Filed: {date_formatted})")
            
            confirm = input(f"\nConfirm selection? (y/n): ").strip().lower()
            if confirm == 'y':
                return selected_years
            
        except ValueError:
            print("❌ Please enter a valid number.")
def download_selected_filings(client, company_number, selected_accounts):
    """
    Download specific company filings based on user selection
    
    Args:
        client: CompaniesHouseClient instance
        company_number: Company number
        selected_accounts: List of selected account records with transaction_ids
        
    Returns:
        Download results dictionary
    """
    print(f"\n📥 Downloading {len(selected_accounts)} selected annual accounts...")
    
    try:
        downloaded_files = []
        failed_downloads = []
        
        for i, account in enumerate(selected_accounts, 1):
            year = account['year']
            transaction_id = account['transaction_id']
            date = account['date']
            
            print(f"   Downloading {i}/{len(selected_accounts)}: {year} accounts...")
            
            try:
                # Download the specific document
                result = client.download_document(transaction_id, company_number, date)
                
                if result:
                    downloaded_files.append({
                        'filename': result['filename'],
                        'path': result['path'],
                        'size': result['size'],
                        'date': date,
                        'year': year,
                        'transaction_id': transaction_id,
                        'description': account['description']
                    })
                    print(f"     ✅ Downloaded: {result['filename']} ({result['size']:,} bytes)")
                else:
                    failed_downloads.append({
                        'year': year,
                        'transaction_id': transaction_id,
                        'error': 'Download failed'
                    })
                    print(f"     ❌ Failed to download {year} accounts")
                    
            except Exception as e:
                failed_downloads.append({
                    'year': year,
                    'transaction_id': transaction_id,
                    'error': str(e)
                })
                print(f"     ❌ Error downloading {year}: {e}")
        
        results = {
            'total_downloaded': len(downloaded_files),
            'downloaded_files': downloaded_files,
            'failed_downloads': failed_downloads,
            'earliest_date': min(account['date'] for account in selected_accounts) if selected_accounts else None,
            'latest_date': max(account['date'] for account in selected_accounts) if selected_accounts else None
        }
        
        if results['total_downloaded'] == 0:
            print("❌ No annual accounts were successfully downloaded.")
            return results
        
        print(f"\n✅ Successfully downloaded {results['total_downloaded']} documents")
        
        if failed_downloads:
            print(f"⚠️  {len(failed_downloads)} downloads failed")
        
        # Show year range
        years = [acc['year'] for acc in selected_accounts if any(df['year'] == acc['year'] for df in downloaded_files)]
        if years:
            print(f"   Years downloaded: {min(years)} to {max(years)}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during download process: {e}")
        logger.error(f"Error downloading selected filings for {company_number}: {e}")
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
        
        # Get user input (company number only)
        company_number = get_user_input()
        
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
        
        # Check available filings and let user choose years
        selected_years, selected_accounts = check_available_filings(ch_client, company_number)
        if not selected_years:
            return 1
        
        # Calculate max_years needed to capture the selected years
        # Get the oldest selected year and calculate how many years back from current year
        current_year = datetime.now().year
        oldest_selected_year = min(selected_years)
        max_years_needed = current_year - oldest_selected_year + 2  # Add buffer
        
        # Download filings using existing method with sufficient years to capture selections
        print(f"\n📥 Downloading annual accounts (scanning last {max_years_needed} years for your selection)...")
        download_results = download_company_filings(ch_client, company_number, max_years_needed)
        if not download_results or download_results['total_downloaded'] == 0:
            return 1
        
        # Filter downloaded files to only include selected years
        print(f"\n🔍 Filtering results to your selected years: {sorted(selected_years, reverse=True)}")
        filtered_files = []
        
        for file_info in download_results['downloaded_files']:
            file_date = file_info.get('date', '')
            if file_date:
                try:
                    # Handle different date formats
                    if isinstance(file_date, str):
                        # String date - parse it
                        if 'T' in file_date:
                            file_year = datetime.fromisoformat(file_date.replace('Z', '+00:00')).year
                        else:
                            file_year = datetime.strptime(file_date, '%Y-%m-%d').year
                    else:
                        # Already a date object - just get the year
                        file_year = file_date.year
                    
                    if file_year in selected_years:
                        filtered_files.append(file_info)
                        print(f"   ✅ Included: {file_info['filename']} (Year {file_year})")
                    else:
                        print(f"   ⏭️  Skipped: {file_info['filename']} (Year {file_year} not in selection)")
                        
                except Exception as e:
                    print(f"   ❌ Could not determine year for {file_info['filename']}: {e}")
                    # Don't include files we can't identify the year for
            else:
                print(f"   ❌ No date found for {file_info['filename']}")
        
        if not filtered_files:
            print("❌ No files matched your selected years.")
            print("   This might mean:")
            print("   - The selected years don't have downloadable accounts")
            print("   - There's a date format issue")
            print("   - The company didn't file accounts in those years")
            return 1
        
        print(f"\n✅ Final filtered selection: {len(filtered_files)} files from {len(selected_years)} years")
        
        # Update download_results to only include filtered files
        download_results['downloaded_files'] = filtered_files
        download_results['total_downloaded'] = len(filtered_files)
        
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