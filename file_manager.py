#!/usr/bin/env python3
"""
File Manager for Strategic Analysis Tool
Handles file operations, saving results, and managing temporary files
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from config import OUTPUT_DIR, TEMP_FOLDER, DATA_DIR

logger = logging.getLogger(__name__)


class FileManager:
    """Manages file operations for the analysis tool"""
    
    def __init__(self):
        """Initialize the file manager"""
        self.output_dir = OUTPUT_DIR
        self.temp_dir = TEMP_FOLDER
        self.data_dir = DATA_DIR
        
        # Ensure directories exist
        for directory in [self.output_dir, self.temp_dir, self.data_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("File manager initialized")
    
    def create_company_folder(self, company_number: str, company_name: str = None) -> Path:
        """
        Create a folder for company results
        
        Args:
            company_number: Company registration number
            company_name: Optional company name for folder naming
            
        Returns:
            Path to the created folder
        """
        # Clean company name for folder name
        if company_name:
            # Remove special characters and limit length
            clean_name = "".join(c for c in company_name if c.isalnum() or c in (' ', '-', '_'))
            clean_name = clean_name.replace(' ', '_').strip('_')[:50]
            folder_name = f"{company_number}_{clean_name}"
        else:
            folder_name = company_number
        
        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{folder_name}_{timestamp}"
        
        folder_path = self.output_dir / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created company folder: {folder_path}")
        return folder_path
    
    def save_analysis_results(self, company_name: str, company_number: str, 
                             analysis_data: Dict[str, Any]) -> Optional[str]:
        """
        Save analysis results to JSON file
        
        Args:
            company_name: Company name
            company_number: Company registration number
            analysis_data: Analysis results to save
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            company_folder = self.create_company_folder(company_number, company_name)
            
            # Add metadata
            analysis_data_with_meta = {
                "metadata": {
                    "company_name": company_name,
                    "company_number": company_number,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "file_type": "analysis_results"
                },
                "analysis": analysis_data
            }
            
            # Save main results
            results_file = company_folder / "analysis_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_data_with_meta, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Analysis results saved: {results_file}")
            return str(results_file)
            
        except Exception as e:
            logger.error(f"Failed to save analysis results: {e}")
            return None
    
    def save_text_report(self, company_number: str, report_content: str, 
                        report_type: str = "report") -> Optional[str]:
        """
        Save text report to file
        
        Args:
            company_number: Company registration number
            report_content: Text content of the report
            report_type: Type of report (for filename)
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            # Find the most recent company folder
            company_folders = [d for d in self.output_dir.iterdir() 
                             if d.is_dir() and d.name.startswith(company_number)]
            
            if not company_folders:
                # Create new folder if none exists
                company_folder = self.create_company_folder(company_number)
            else:
                # Use most recent folder
                company_folder = sorted(company_folders, key=lambda x: x.stat().st_mtime)[-1]
            
            # Save report
            report_file = company_folder / f"{report_type}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Text report saved: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Failed to save text report: {e}")
            return None
    
    def save_json_results(self, company_number: str, data: Dict[str, Any], 
                         filename: str = "summary") -> Optional[str]:
        """
        Save JSON data to file
        
        Args:
            company_number: Company registration number
            data: Data to save
            filename: Filename (without extension)
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            # Find the most recent company folder
            company_folders = [d for d in self.output_dir.iterdir() 
                             if d.is_dir() and d.name.startswith(company_number)]
            
            if not company_folders:
                company_folder = self.create_company_folder(company_number)
            else:
                company_folder = sorted(company_folders, key=lambda x: x.stat().st_mtime)[-1]
            
            # Save JSON data
            json_file = company_folder / f"{filename}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"JSON results saved: {json_file}")
            return str(json_file)
            
        except Exception as e:
            logger.error(f"Failed to save JSON results: {e}")
            return None
    
    def save_raw_content(self, company_number: str, content: str, 
                        source_filename: str) -> Optional[str]:
        """
        Save raw extracted content to file
        
        Args:
            company_number: Company registration number
            content: Raw content text
            source_filename: Original filename
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            # Find the most recent company folder
            company_folders = [d for d in self.output_dir.iterdir() 
                             if d.is_dir() and d.name.startswith(company_number)]
            
            if not company_folders:
                company_folder = self.create_company_folder(company_number)
            else:
                company_folder = sorted(company_folders, key=lambda x: x.stat().st_mtime)[-1]
            
            # Create raw content subfolder
            raw_folder = company_folder / "raw_content"
            raw_folder.mkdir(exist_ok=True)
            
            # Save content
            content_file = raw_folder / f"{Path(source_filename).stem}_content.txt"
            with open(content_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Raw content saved: {content_file}")
            return str(content_file)
            
        except Exception as e:
            logger.error(f"Failed to save raw content: {e}")
            return None
    
    def load_analysis_results(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load analysis results from JSON file
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Analysis data or None if failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Analysis results loaded: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load analysis results: {e}")
            return None
    
    def list_company_analyses(self, company_number: str = None) -> List[Dict[str, Any]]:
        """
        List all available analyses
        
        Args:
            company_number: Optional filter by company number
            
        Returns:
            List of analysis metadata
        """
        analyses = []
        
        try:
            for folder in self.output_dir.iterdir():
                if not folder.is_dir():
                    continue
                
                # Check if this is a company folder
                if company_number and not folder.name.startswith(company_number):
                    continue
                
                # Look for analysis results
                results_file = folder / "analysis_results.json"
                if results_file.exists():
                    try:
                        with open(results_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        metadata = data.get('metadata', {})
                        analyses.append({
                            "folder": str(folder),
                            "company_name": metadata.get('company_name', 'Unknown'),
                            "company_number": metadata.get('company_number', 'Unknown'),
                            "analysis_date": metadata.get('analysis_timestamp', 'Unknown'),
                            "files": [f.name for f in folder.iterdir() if f.is_file()]
                        })
                        
                    except Exception as e:
                        logger.warning(f"Could not read analysis from {folder}: {e}")
                        continue
            
            # Sort by analysis date (most recent first)
            analyses.sort(key=lambda x: x['analysis_date'], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list analyses: {e}")
        
        return analyses
    
    def cleanup_temp_files(self, older_than_hours: int = 24) -> int:
        """
        Clean up temporary files older than specified hours
        
        Args:
            older_than_hours: Remove files older than this many hours
            
        Returns:
            Number of files cleaned up
        """
        count = 0
        cutoff_time = datetime.now().timestamp() - (older_than_hours * 3600)
        
        try:
            for file_path in self.temp_dir.iterdir():
                if file_path.is_file():
                    try:
                        if file_path.stat().st_mtime < cutoff_time:
                            file_path.unlink()
                            count += 1
                    except Exception as e:
                        logger.warning(f"Could not delete {file_path}: {e}")
            
            logger.info(f"Cleaned up {count} temporary files")
            
        except Exception as e:
            logger.error(f"Error during temp file cleanup: {e}")
        
        return count
    
    def get_temp_file_path(self, filename: str) -> Path:
        """
        Get path for a temporary file
        
        Args:
            filename: Name of the file
            
        Returns:
            Path object for the temporary file
        """
        return self.temp_dir / filename
    
    def save_temp_file(self, content: bytes, filename: str) -> Optional[str]:
        """
        Save content to a temporary file
        
        Args:
            content: File content as bytes
            filename: Name of the file
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            file_path = self.get_temp_file_path(filename)
            
            with open(file_path, 'wb') as f:
                f.write(content)
            
            logger.debug(f"Temporary file saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save temporary file: {e}")
            return None
    
    def create_analysis_summary(self, company_number: str) -> Optional[Dict[str, Any]]:
        """
        Create a summary of all analyses for a company
        
        Args:
            company_number: Company registration number
            
        Returns:
            Summary data or None if failed
        """
        try:
            analyses = self.list_company_analyses(company_number)
            
            if not analyses:
                return None
            
            summary = {
                "company_number": company_number,
                "total_analyses": len(analyses),
                "latest_analysis": analyses[0] if analyses else None,
                "analysis_history": analyses,
                "summary_created": datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to create analysis summary: {e}")
            return None
    
    def export_analysis(self, company_number: str, export_format: str = "json") -> Optional[str]:
        """
        Export analysis results in specified format
        
        Args:
            company_number: Company registration number
            export_format: Export format (json, txt, csv)
            
        Returns:
            Path to exported file or None if failed
        """
        try:
            analyses = self.list_company_analyses(company_number)
            
            if not analyses:
                logger.warning(f"No analyses found for company {company_number}")
                return None
            
            latest_analysis = analyses[0]
            results_file = Path(latest_analysis['folder']) / "analysis_results.json"
            
            if not results_file.exists():
                logger.error(f"Analysis results file not found: {results_file}")
                return None
            
            # Load the analysis data
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Export based on format
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if export_format.lower() == "json":
                export_file = self.output_dir / f"{company_number}_export_{timestamp}.json"
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            elif export_format.lower() == "txt":
                export_file = self.output_dir / f"{company_number}_export_{timestamp}.txt"
                with open(export_file, 'w', encoding='utf-8') as f:
                    f.write(f"Analysis Export for Company {company_number}\n")
                    f.write("="*50 + "\n\n")
                    f.write(json.dumps(data, indent=2, default=str))
            
            else:
                logger.error(f"Unsupported export format: {export_format}")
                return None
            
            logger.info(f"Analysis exported: {export_file}")
            return str(export_file)
            
        except Exception as e:
            logger.error(f"Failed to export analysis: {e}")
            return None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics
        
        Returns:
            Dictionary with storage information
        """
        try:
            stats = {
                "output_dir": str(self.output_dir),
                "temp_dir": str(self.temp_dir),
                "data_dir": str(self.data_dir),
                "total_companies": 0,
                "total_analyses": 0,
                "total_files": 0,
                "total_size_mb": 0
            }
            
            # Count companies and analyses
            for folder in self.output_dir.iterdir():
                if folder.is_dir():
                    stats["total_companies"] += 1
                    
                    if (folder / "analysis_results.json").exists():
                        stats["total_analyses"] += 1
                    
                    for file_path in folder.rglob("*"):
                        if file_path.is_file():
                            stats["total_files"] += 1
                            stats["total_size_mb"] += file_path.stat().st_size / (1024 * 1024)
            
            # Round size
            stats["total_size_mb"] = round(stats["total_size_mb"], 2)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {"error": str(e)}


# Convenience functions
def save_analysis(company_name: str, company_number: str, analysis_data: Dict[str, Any]) -> Optional[str]:
    """
    Quick function to save analysis results
    
    Args:
        company_name: Company name
        company_number: Company registration number
        analysis_data: Analysis results
        
    Returns:
        Path to saved file or None if failed
    """
    file_manager = FileManager()
    return file_manager.save_analysis_results(company_name, company_number, analysis_data)


def load_analysis(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Quick function to load analysis results
    
    Args:
        file_path: Path to the analysis file
        
    Returns:
        Analysis data or None if failed
    """
    file_manager = FileManager()
    return file_manager.load_analysis_results(file_path)


if __name__ == "__main__":
    # Test the file manager
    file_manager = FileManager()
    
    # Test creating company folder
    test_folder = file_manager.create_company_folder("12345678", "Test Company Ltd")
    print(f"Created test folder: {test_folder}")
    
    # Test saving analysis
    test_data = {
        "business_strategy": "Test Strategy",
        "risk_strategy": "Test Risk",
        "confidence": 0.85
    }
    
    result = file_manager.save_analysis_results("Test Company Ltd", "12345678", test_data)
    print(f"Analysis saved: {result}")
    
    # Get storage stats
    stats = file_manager.get_storage_stats()
    print(f"Storage stats: {stats}")
    
    # Cleanup test files
    if test_folder.exists():
        import shutil
        shutil.rmtree(test_folder)
        print("Test folder cleaned up")