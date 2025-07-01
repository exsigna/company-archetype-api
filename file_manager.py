#!/usr/bin/env python3
"""
File operations manager for the analysis tool
Handles all file I/O operations, directory management, and cleanup
"""

import json
import shutil
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

from config import RESULTS_FOLDER, TEMP_FOLDER

# Set up logging
logger = logging.getLogger(__name__)


class FileManager:
    """Manages all file operations for the analysis tool"""
    
    def __init__(self):
        self.results_folder = RESULTS_FOLDER
        self.temp_folder = TEMP_FOLDER
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        self.results_folder.mkdir(exist_ok=True)
        self.temp_folder.mkdir(exist_ok=True)
    
    def _get_timestamp(self) -> str:
        """Generate timestamp string for filenames"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_analysis_results(self, firm_name: str, company_number: str, 
                            analysis_data: Dict, filename_suffix: str = "") -> str:
        """
        Save analysis results to JSON file
        
        Args:
            firm_name: Name of the firm
            company_number: Company registration number
            analysis_data: Data to save
            filename_suffix: Optional suffix for filename
            
        Returns:
            Path to saved file, empty string if failed
        """
        timestamp = self._get_timestamp()
        
        if filename_suffix:
            filename = f"{company_number}_{filename_suffix}_{timestamp}.json"
        else:
            filename = f"{company_number}_analysis_{timestamp}.json"
        
        filepath = self.results_folder / filename
        
        # Add metadata to the data
        enhanced_data = analysis_data.copy()
        enhanced_data.update({
            "firm_name": firm_name,
            "company_number": company_number,
            "analysis_timestamp": timestamp,
            "file_metadata": {
                "created_at": datetime.now().isoformat(),
                "filename": filename,
                "file_size": None  # Will be set after saving
            }
        })
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False, default=str)
            
            # Update file size in metadata
            file_size = filepath.stat().st_size
            enhanced_data["file_metadata"]["file_size"] = file_size
            
            # Save again with file size
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Analysis results saved to: {filepath}")
            logger.info(f"File size: {file_size:,} bytes")
            print(f"Analysis results saved to: {filepath}")
            print(f"File size: {file_size:,} bytes")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
            print(f"Error saving analysis results: {e}")
            return ""
    
    def save_text_report(self, company_number: str, report_content: str, 
                        report_type: str = "report") -> str:
        """
        Save text report to file
        
        Args:
            company_number: Company registration number
            report_content: Text content of the report
            report_type: Type of report (for filename)
            
        Returns:
            Path to saved file, empty string if failed
        """
        timestamp = self._get_timestamp()
        filename = f"{company_number}_{report_type}_{timestamp}.txt"
        filepath = self.results_folder / filename
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report_content)
            
            file_size = filepath.stat().st_size
            logger.info(f"Report saved to: {filepath}")
            logger.info(f"File size: {file_size:,} bytes")
            print(f"Report saved to: {filepath}")
            print(f"File size: {file_size:,} bytes")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            print(f"Error saving report: {e}")
            return ""
    
    def save_debug_data(self, company_number: str, debug_data: Dict, 
                       debug_type: str = "debug") -> str:
        """
        Save debug data for troubleshooting
        
        Args:
            company_number: Company registration number
            debug_data: Debug data to save
            debug_type: Type of debug data (for filename)
            
        Returns:
            Path to saved file, empty string if failed
        """
        timestamp = self._get_timestamp()
        filename = f"{company_number}_{debug_type}_{timestamp}.json"
        filepath = self.results_folder / filename
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Debug data saved to: {filepath}")
            print(f"Debug data saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving debug data: {e}")
            print(f"Error saving debug data: {e}")
            return ""
    
    def save_combined_content(self, company_number: str, content: str) -> str:
        """
        Save combined content for debugging AI analysis
        
        Args:
            company_number: Company registration number
            content: Combined content text
            
        Returns:
            Path to saved file, empty string if failed
        """
        filename = f"{company_number}_combined_content.txt"
        filepath = self.results_folder / filename
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            
            file_size = filepath.stat().st_size
            logger.info(f"Combined content saved for debugging: {filepath}")
            logger.info(f"Content size: {file_size:,} bytes")
            print(f"Combined content saved for debugging: {filepath}")
            print(f"Content size: {file_size:,} bytes")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Could not save debug content: {e}")
            print(f"Could not save debug content: {e}")
            return ""
    
    def create_temp_directory(self, company_number: str) -> Path:
        """
        Create temporary directory for a company's downloads
        
        Args:
            company_number: Company registration number
            
        Returns:
            Path to created directory
        """
        # Validate company number
        if not self._validate_company_number(company_number):
            logger.warning(f"Invalid company number format: {company_number}")
        
        temp_dir = self.temp_folder / f"{company_number}_accounts"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
    
    def _validate_company_number(self, company_number: str) -> bool:
        """Validate UK company number format"""
        import re
        # UK company numbers are typically 8 digits, sometimes with leading zeros
        # or 2 letters followed by 6 digits
        pattern = r'^\d{8}$|^[A-Z]{2}\d{6}$'
        return bool(re.match(pattern, company_number.upper().zfill(8)))
    
    def cleanup_temp_directory(self, temp_path: str) -> bool:
        """
        Clean up temporary directory after processing
        
        Args:
            temp_path: Path to temporary directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
                logger.info(f"Cleaned up temporary directory: {temp_path}")
                print(f"Cleaned up temporary directory: {temp_path}")
                return True
            return True
        except Exception as e:
            logger.error(f"Could not delete temporary directory: {e}")
            print(f"Could not delete temporary directory: {e}")
            return False
    
    def load_analysis_results(self, filepath: str) -> Optional[Dict]:
        """
        Load previously saved analysis results
        
        Args:
            filepath: Path to analysis results file
            
        Returns:
            Loaded data or None if failed
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading analysis results: {e}")
            print(f"Error loading analysis results: {e}")
            return None
    
    def list_analysis_files(self, company_number: Optional[str] = None) -> List[str]:
        """
        List analysis files in results directory
        
        Args:
            company_number: Optional filter by company number
            
        Returns:
            List of file paths
        """
        try:
            pattern = f"{company_number}_*" if company_number else "*"
            files = list(self.results_folder.glob(f"{pattern}.json"))
            return [str(f) for f in sorted(files, reverse=True)]  # Most recent first
        except Exception as e:
            logger.error(f"Error listing analysis files: {e}")
            print(f"Error listing analysis files: {e}")
            return []
    
    def get_file_info(self, filepath: str) -> Dict:
        """
        Get information about a file
        
        Args:
            filepath: Path to file
            
        Returns:
            Dict with file information
        """
        try:
            path = Path(filepath)
            stat = path.stat()
            return {
                "filename": path.name,
                "filepath": str(path),
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "exists": True
            }
        except Exception as e:
            return {
                "filename": Path(filepath).name,
                "filepath": filepath,
                "error": str(e),
                "exists": False
            }
    
    def archive_old_files(self, days_old: int = 30) -> int:
        """
        Archive files older than specified days
        
        Args:
            days_old: Archive files older than this many days
            
        Returns:
            Number of files archived
        """
        archive_dir = self.results_folder / "archive"
        archive_dir.mkdir(exist_ok=True)
        
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        archived_count = 0
        
        try:
            for file_path in self.results_folder.glob("*.json"):
                if file_path.stat().st_mtime < cutoff_time:
                    archive_path = archive_dir / file_path.name
                    shutil.move(str(file_path), str(archive_path))
                    archived_count += 1
            
            if archived_count > 0:
                logger.info(f"Archived {archived_count} files older than {days_old} days")
                print(f"Archived {archived_count} files older than {days_old} days")
            
            return archived_count
            
        except Exception as e:
            logger.error(f"Error archiving files: {e}")
            print(f"Error archiving files: {e}")
            return 0
    
    def get_storage_summary(self) -> Dict:
        """
        Get summary of storage usage
        
        Returns:
            Dict with storage information
        """
        summary = {
            "results_folder": str(self.results_folder),
            "temp_folder": str(self.temp_folder),
            "total_files": 0,
            "total_size_bytes": 0,
            "total_size_mb": 0,
            "file_types": {}
        }
        
        try:
            for file_path in self.results_folder.rglob("*"):
                if file_path.is_file():
                    summary["total_files"] += 1
                    file_size = file_path.stat().st_size
                    summary["total_size_bytes"] += file_size
                    
                    # Count by file extension
                    ext = file_path.suffix.lower()
                    if ext not in summary["file_types"]:
                        summary["file_types"][ext] = {"count": 0, "size_bytes": 0}
                    summary["file_types"][ext]["count"] += 1
                    summary["file_types"][ext]["size_bytes"] += file_size
            
            summary["total_size_mb"] = round(summary["total_size_bytes"] / (1024 * 1024), 2)
            
            # Convert file type sizes to MB
            for ext_info in summary["file_types"].values():
                ext_info["size_mb"] = round(ext_info["size_bytes"] / (1024 * 1024), 2)
            
        except Exception as e:
            summary["error"] = str(e)
        
        return summary