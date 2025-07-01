#!/usr/bin/env python3
"""
Companies House API Client
Handles API interactions with Companies House to fetch company data and filing documents
"""

import requests
import json
import base64
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urljoin

from config import (
    COMPANIES_HOUSE_API_KEY, CH_BASE_URL, REQUEST_TIMEOUT, TEMP_FOLDER, 
    DEFAULT_MAX_YEARS, ERROR_MESSAGES, SUCCESS_MESSAGES
)

# Get logger (don't configure here)
logger = logging.getLogger(__name__)


class CompaniesHouseClient:
    """
    Client for interacting with the Companies House API
    """
    
    def __init__(self):
        """Initialize the Companies House client"""
        self.api_key = COMPANIES_HOUSE_API_KEY
        self.base_url = CH_BASE_URL
        self.session = requests.Session()
        
        # Set up authentication - Companies House uses basic auth with API key as username
        auth_string = base64.b64encode(f"{self.api_key}:".encode()).decode()
        self.session.headers.update({
            'Authorization': f'Basic {auth_string}',
            'Content-Type': 'application/json',
            'User-Agent': 'Strategic Analysis Tool/1.0'
        })
        
        logger.info("Companies House client initialized")
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make a request to the Companies House API
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response or None if failed
        """
        try:
            url = urljoin(self.base_url, endpoint)
            logger.debug(f"Making request to: {url}")
            
            response = self.session.get(
                url, 
                params=params, 
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.warning(f"Resource not found: {url}")
                return None
            elif response.status_code == 429:
                logger.warning("Rate limit exceeded, waiting...")
                time.sleep(1)
                return self._make_request(endpoint, params)
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout: {endpoint}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None
    
    def get_company_profile(self, company_number: str) -> Optional[Dict]:
        """
        Get company profile information
        
        Args:
            company_number: UK company number
            
        Returns:
            Company profile data or None if not found
        """
        endpoint = f"/company/{company_number}"
        return self._make_request(endpoint)
    
    def get_company_officers(self, company_number: str) -> Optional[Dict]:
        """
        Get company officers information
        
        Args:
            company_number: UK company number
            
        Returns:
            Officers data or None if not found
        """
        endpoint = f"/company/{company_number}/officers"
        return self._make_request(endpoint)
    
    def get_filing_history(self, company_number: str, category: Optional[str] = None, 
                          items_per_page: int = 100) -> Optional[Dict]:
        """
        Get company filing history
        
        Args:
            company_number: UK company number
            category: Filing category filter
            items_per_page: Number of items per page (max 100)
            
        Returns:
            Filing history data or None if not found
        """
        endpoint = f"/company/{company_number}/filing-history"
        params = {"items_per_page": min(items_per_page, 100)}
        
        if category:
            params["category"] = category
            
        return self._make_request(endpoint, params)
    
    def get_filing_document(self, company_number: str, transaction_id: str) -> Optional[bytes]:
        """
        Download a filing document
        
        Args:
            company_number: UK company number
            transaction_id: Filing transaction ID
            
        Returns:
            Document content as bytes or None if failed
        """
        try:
            endpoint = f"/company/{company_number}/filing-history/{transaction_id}/document"
            url = urljoin(self.base_url, endpoint)
            
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                return response.content
            else:
                logger.error(f"Failed to download document: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading document: {e}")
            return None
    
    def search_companies(self, query: str, items_per_page: int = 20) -> Optional[Dict]:
        """
        Search for companies
        
        Args:
            query: Search query
            items_per_page: Number of results per page
            
        Returns:
            Search results or None if failed
        """
        endpoint = "/search/companies"
        params = {
            "q": query,
            "items_per_page": min(items_per_page, 100)
        }
        
        return self._make_request(endpoint, params)
    
    def download_annual_accounts(self, company_number: str, max_years: int = DEFAULT_MAX_YEARS) -> Dict[str, Any]:
        """
        Download annual accounts for a company
        
        Args:
            company_number: UK company number
            max_years: Maximum years to look back
            
        Returns:
            Dictionary with download results and metadata
        """
        download_results = {
            "company_number": company_number,
            "downloaded_files": [],
            "failed_downloads": [],
            "total_found": 0,
            "total_downloaded": 0,
            "earliest_date": None,
            "latest_date": None,
            "errors": []
        }
        
        try:
            logger.info(f"Fetching filing history for company {company_number}")
            
            # Get filing history
            filing_data = self.get_filing_history(company_number, category="accounts")
            
            if not filing_data:
                error_msg = f"No filing history found for company {company_number}"
                logger.error(error_msg)
                download_results["errors"].append(error_msg)
                return download_results
            
            items = filing_data.get("items", [])
            download_results["total_found"] = len(items)
            
            if not items:
                logger.warning("No filing items found")
                return download_results
            
            # Calculate date cutoff
            cutoff_date = datetime.now() - timedelta(days=max_years * 365)
            
            # Process each filing
            for item in items:
                try:
                    # Check if it's an annual accounts filing
                    description = item.get("description", "").lower()
                    category = item.get("category", "").lower()
                    
                    if not ("accounts" in description or "annual" in description or category == "accounts"):
                        continue
                    
                    # Check date
                    date_str = item.get("date")
                    if not date_str:
                        continue
                        
                    try:
                        item_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    except ValueError:
                        logger.warning(f"Invalid date format: {date_str}")
                        continue
                    
                    # Skip if too old
                    if item_date < cutoff_date.date():
                        logger.debug(f"Skipping old filing: {item_date}")
                        continue
                    
                    # Update date range
                    if not download_results["earliest_date"] or item_date < download_results["earliest_date"]:
                        download_results["earliest_date"] = item_date
                    if not download_results["latest_date"] or item_date > download_results["latest_date"]:
                        download_results["latest_date"] = item_date
                    
                    logger.info(f"Found accounts: {item.get('description', 'N/A')} ({item_date})")
                    
                    # Process downloadable document
                    transaction_id = item.get("transaction_id")
                    if not transaction_id:
                        logger.warning("No transaction ID found for filing")
                        continue
                    
                    # Check if document metadata is available (two-step download process)
                    links = item.get("links", {})
                    if "document_metadata" not in links:
                        logger.debug(f"No document metadata link for {transaction_id}")
                        continue
                    
                    try:
                        # Step 1: Get document metadata
                        logger.debug(f"Getting document metadata for {transaction_id}")
                        metadata_url = links["document_metadata"]
                        
                        # Make request to document metadata endpoint
                        metadata_response = self.session.get(metadata_url, timeout=REQUEST_TIMEOUT)
                        if metadata_response.status_code != 200:
                            logger.warning(f"Failed to get document metadata: {metadata_response.status_code}")
                            continue
                        
                        metadata = metadata_response.json()
                        
                        # Step 2: Get actual document download URL from metadata
                        metadata_links = metadata.get("links", {})
                        if "document" not in metadata_links:
                            logger.debug(f"No document download link in metadata for {transaction_id}")
                            continue
                        
                        document_url = metadata_links["document"]
                        
                        # Download the document
                        logger.info(f"Downloading document {transaction_id}")
                        document_response = self.session.get(document_url, timeout=REQUEST_TIMEOUT)
                        
                        if document_response.status_code == 200:
                            document_content = document_response.content
                            
                            # Save the document
                            filename = f"{company_number}_{transaction_id}_{item_date}.pdf"
                            file_path = TEMP_FOLDER / filename
                            
                            try:
                                with open(file_path, 'wb') as f:
                                    f.write(document_content)
                                
                                download_results["downloaded_files"].append({
                                    "filename": filename,
                                    "path": str(file_path),
                                    "transaction_id": transaction_id,
                                    "date": item_date,
                                    "description": item.get("description", ""),
                                    "size": len(document_content)
                                })
                                
                                download_results["total_downloaded"] += 1
                                logger.info(f"Successfully downloaded: {filename} ({len(document_content):,} bytes)")
                                
                            except Exception as e:
                                error_msg = f"Failed to save file {filename}: {e}"
                                logger.error(error_msg)
                                download_results["errors"].append(error_msg)
                                download_results["failed_downloads"].append({
                                    "transaction_id": transaction_id,
                                    "error": str(e)
                                })
                        else:
                            error_msg = f"Failed to download document {transaction_id}: HTTP {document_response.status_code}"
                            logger.error(error_msg)
                            download_results["failed_downloads"].append({
                                "transaction_id": transaction_id,
                                "error": f"HTTP {document_response.status_code}"
                            })
                            
                    except Exception as e:
                        error_msg = f"Error downloading document {transaction_id}: {e}"
                        logger.error(error_msg)
                        download_results["failed_downloads"].append({
                            "transaction_id": transaction_id,
                            "error": str(e)
                        })
                        
                except Exception as e:
                    error_msg = f"Error processing filing item: {e}"
                    logger.error(error_msg)
                    download_results["errors"].append(error_msg)
                    continue
            
            # Summary
            logger.info(f"Download complete: {download_results['total_downloaded']}/{download_results['total_found']} files")
            
            if download_results["earliest_date"] and download_results["latest_date"]:
                logger.info(f"Date range: {download_results['earliest_date']} to {download_results['latest_date']}")
                
        except Exception as e:
            error_msg = f"Error in download process: {e}"
            logger.error(error_msg)
            download_results["errors"].append(error_msg)
        
        return download_results
    
    def get_company_summary(self, company_number: str) -> Dict[str, Any]:
        """
        Get a comprehensive summary of company information
        
        Args:
            company_number: UK company number
            
        Returns:
            Dictionary with company summary data
        """
        summary = {
            "company_number": company_number,
            "profile": None,
            "officers": None,
            "recent_filings": None,
            "status": "unknown",
            "errors": []
        }
        
        try:
            # Get company profile
            logger.info(f"Fetching company profile for {company_number}")
            profile = self.get_company_profile(company_number)
            
            if profile:
                summary["profile"] = profile
                summary["status"] = profile.get("company_status", "unknown")
                logger.info(f"Company: {profile.get('company_name', 'Unknown')} - Status: {summary['status']}")
            else:
                summary["errors"].append("Failed to fetch company profile")
            
            # Get officers
            logger.info("Fetching company officers")
            officers = self.get_company_officers(company_number)
            if officers:
                summary["officers"] = officers
                officer_count = officers.get("total_results", 0)
                logger.info(f"Found {officer_count} officers")
            else:
                summary["errors"].append("Failed to fetch company officers")
            
            # Get recent filings
            logger.info("Fetching recent filings")
            filings = self.get_filing_history(company_number, items_per_page=10)
            if filings:
                summary["recent_filings"] = filings
                filing_count = len(filings.get("items", []))
                logger.info(f"Found {filing_count} recent filings")
            else:
                summary["errors"].append("Failed to fetch recent filings")
                
        except Exception as e:
            error_msg = f"Error creating company summary: {e}"
            logger.error(error_msg)
            summary["errors"].append(error_msg)
        
        return summary
    
    def validate_company_exists(self, company_number: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that a company exists and return basic info
        
        Args:
            company_number: UK company number
            
        Returns:
            Tuple of (exists, company_name)
        """
        try:
            profile = self.get_company_profile(company_number)
            
            if profile:
                company_name = profile.get("company_name", "Unknown")
                logger.info(f"Company validated: {company_name}")
                return True, company_name
            else:
                logger.warning(f"Company {company_number} not found")
                return False, None
                
        except Exception as e:
            logger.error(f"Error validating company: {e}")
            return False, None
    
    def cleanup_temp_files(self) -> int:
        """
        Clean up temporary downloaded files
        
        Returns:
            Number of files cleaned up
        """
        try:
            count = 0
            for file_path in TEMP_FOLDER.glob("*.pdf"):
                try:
                    file_path.unlink()
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")
            
            logger.info(f"Cleaned up {count} temporary files")
            return count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0


# Convenience function for quick company lookup
def quick_company_lookup(company_number: str) -> Optional[Dict]:
    """
    Quick lookup of company basic information
    
    Args:
        company_number: UK company number
        
    Returns:
        Company profile or None if not found
    """
    client = CompaniesHouseClient()
    return client.get_company_profile(company_number)


# Convenience function for downloading accounts
def download_company_accounts(company_number: str, max_years: int = DEFAULT_MAX_YEARS) -> Dict[str, Any]:
    """
    Download company accounts with default settings
    
    Args:
        company_number: UK company number
        max_years: Maximum years to look back
        
    Returns:
        Download results dictionary
    """
    client = CompaniesHouseClient()
    return client.download_annual_accounts(company_number, max_years)


if __name__ == "__main__":
    # Test the client
    import sys
    
    if len(sys.argv) > 1:
        company_number = sys.argv[1]
        print(f"Testing Companies House client with company {company_number}")
        
        client = CompaniesHouseClient()
        
        # Test company validation
        exists, name = client.validate_company_exists(company_number)
        if exists:
            print(f"✓ Company found: {name}")
            
            # Test summary
            summary = client.get_company_summary(company_number)
            print(f"✓ Summary generated with {len(summary.get('errors', []))} errors")
            
            # Test download
            results = client.download_annual_accounts(company_number, max_years=2)
            print(f"✓ Found {results['total_found']} filings, downloaded {results['total_downloaded']}")
            
        else:
            print(f"✗ Company {company_number} not found")
    else:
        print("Usage: python companies_house_client.py <company_number>")
        print("Example: python companies_house_client.py 00000006")