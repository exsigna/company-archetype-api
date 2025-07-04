#!/usr/bin/env python3
"""
PDF content extraction module with memory management
Handles text extraction from PDF files using multiple methods
"""

import io
import gc
import logging
from typing import Dict, List, Optional
from pathlib import Path

# Add these imports for memory management
try:
    import psutil
    import os
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from config import MAX_PAGES_OCR, OCR_DPI, OCR_CONFIG, MIN_EXTRACTION_LENGTH

# Get logger (don't configure here)
logger = logging.getLogger(__name__)


class PDFExtractor:
    """Handles PDF text extraction using multiple methods with memory management"""
    
    def __init__(self):
        """Initialize PDFExtractor and check available dependencies"""
        self.available_methods = []
        
        # Check for PyPDF2
        try:
            import PyPDF2
            self.PyPDF2 = PyPDF2
            self.available_methods.append("pypdf2")
            logger.info("PyPDF2 available for PDF analysis")
        except ImportError:
            self.PyPDF2 = None
            logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")
        
        # Check for pdfplumber
        try:
            import pdfplumber
            self.pdfplumber = pdfplumber
            self.available_methods.append("pdfplumber")
            logger.info("pdfplumber available for enhanced PDF extraction")
        except ImportError:
            self.pdfplumber = None
            logger.warning("pdfplumber not installed. Install with: pip install pdfplumber")
        
        # Check for OCR libraries
        try:
            import pytesseract
            from pdf2image import convert_from_bytes
            self.pytesseract = pytesseract
            self.convert_from_bytes = convert_from_bytes
            self.available_methods.append("ocr")
            logger.info("OCR libraries available (pytesseract + pdf2image)")
        except ImportError:
            self.pytesseract = None
            self.convert_from_bytes = None
            logger.warning("OCR libraries not installed. Install with: pip install pytesseract pdf2image")
        
        if not self.available_methods:
            logger.error("No PDF extraction methods available! Please install required dependencies.")
    
    def log_memory_usage(self, context: str):
        """Log current memory usage if psutil is available"""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
                logger.info(f"Memory usage {context}: {memory_mb:.1f}MB")
                
                # Warning if memory usage is high
                if memory_mb > 1500:  # 1.5GB warning threshold
                    logger.warning(f"High memory usage detected: {memory_mb:.1f}MB")
                    
                return memory_mb
            except Exception as e:
                logger.debug(f"Could not get memory usage: {e}")
        return None
    
    def cleanup_memory(self):
        """Force garbage collection and log memory cleanup"""
        self.log_memory_usage("before cleanup")
        gc.collect()
        self.log_memory_usage("after cleanup")
    
    def extract_text_from_pdf(self, pdf_content: bytes, filename: str) -> Dict:
        """
        Extract text from PDF using the best available method with memory management
        
        Args:
            pdf_content: PDF file content as bytes
            filename: Name of the PDF file for logging
            
        Returns:
            Dict containing extraction results and metadata with standardized format
        """
        self.log_memory_usage(f"starting extraction for {filename}")
        
        if not pdf_content:
            return self._create_error_result(filename, "No PDF content provided")
        
        result = self._create_base_result(filename)
        
        if not self.available_methods:
            return self._create_error_result(filename, "No PDF extraction methods available")
        
        # Try extraction methods in order of preference
        for method in self.available_methods:
            try:
                logger.info(f"Attempting {method} extraction for {filename}")
                
                if method == "pdfplumber" and self.pdfplumber:
                    if self._extract_with_pdfplumber(pdf_content, result):
                        self.cleanup_memory()
                        return result
                elif method == "pypdf2" and self.PyPDF2:
                    if self._extract_with_pypdf2(pdf_content, result):
                        self.cleanup_memory()
                        return result
                elif method == "ocr" and self.pytesseract:
                    if self._extract_with_ocr(pdf_content, result):
                        self.cleanup_memory()
                        return result
                        
                # Clean up after each attempt
                self.cleanup_memory()
                
            except Exception as e:
                logger.error(f"Exception in {method} extraction for {filename}: {e}")
                self.cleanup_memory()
                continue
        
        logger.error(f"All extraction methods failed for {filename}")
        return self._create_error_result(filename, "All extraction methods failed")
    
    def _extract_with_pdfplumber(self, pdf_content: bytes, result: Dict) -> bool:
        """Extract text using pdfplumber with memory monitoring"""
        try:
            logger.debug("Attempting pdfplumber extraction...")
            self.log_memory_usage("before pdfplumber")
            
            with self.pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                if not pdf.pages:
                    logger.debug("PDF has no pages")
                    return False
                
                full_text = ""
                tables_found = 0
                
                result["debug_info"]["total_pages"] = len(pdf.pages)
                logger.debug(f"PDF has {len(pdf.pages)} pages")
                
                # Process each page with memory monitoring
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Check memory before processing each page
                        if page_num % 10 == 0:  # Check every 10 pages
                            memory_mb = self.log_memory_usage(f"page {page_num + 1}")
                            if memory_mb and memory_mb > 1800:  # 1.8GB threshold
                                logger.warning(f"Memory limit approaching, stopping at page {page_num + 1}")
                                break
                        
                        page_header = f"\n{'='*50}\nPAGE {page_num + 1}\n{'='*50}\n"
                        
                        # Extract regular text
                        page_text = page.extract_text()
                        if page_text:
                            cleaned_text = page_text.replace('\n\n\n', '\n\n').replace('\t', ' ')
                            full_text += page_header + cleaned_text + "\n"
                        
                        # Extract tables
                        tables = page.extract_tables()
                        if tables:
                            tables_found += len(tables)
                            for table_num, table in enumerate(tables):
                                table_text = self._process_financial_table(table)
                                if table_text:
                                    table_header = f"\n--- TABLE {page_num + 1}.{table_num + 1} ---\n"
                                    full_text += table_header + table_text + "\n"
                        
                        # Clean up variables for this page
                        del page_text, tables
                        
                    except Exception as e:
                        logger.debug(f"Error processing page {page_num + 1}: {e}")
                        continue
                
                result["debug_info"]["text_length"] = len(full_text)
                result["debug_info"]["tables_found"] = tables_found
                
                logger.debug(f"Extracted {len(full_text)} characters of text")
                logger.debug(f"Found {tables_found} tables total")
                
                if len(full_text) > MIN_EXTRACTION_LENGTH:
                    result["raw_text"] = full_text
                    result["extraction_status"] = "success"
                    result["extraction_method"] = "pdfplumber"
                    result["debug_info"]["sample_text"] = full_text[:500]
                    logger.info(f"pdfplumber extraction successful")
                    return True
                else:
                    logger.debug(f"pdfplumber extracted minimal text ({len(full_text)} chars)")
                    return False
                    
        except Exception as e:
            logger.debug(f"pdfplumber extraction failed: {e}")
            return False
        finally:
            # Always clean up
            gc.collect()
    
    def _extract_with_pypdf2(self, pdf_content: bytes, result: Dict) -> bool:
        """Extract text using PyPDF2 with memory monitoring"""
        try:
            logger.debug("Attempting PyPDF2 extraction...")
            self.log_memory_usage("before PyPDF2")
            
            pdf_reader = self.PyPDF2.PdfReader(io.BytesIO(pdf_content))
            
            if not pdf_reader.pages:
                logger.debug("PDF has no pages")
                return False
            
            full_text = ""
            
            result["debug_info"]["total_pages"] = len(pdf_reader.pages)
            logger.debug(f"PDF has {len(pdf_reader.pages)} pages")
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    # Check memory periodically
                    if page_num % 20 == 0:
                        memory_mb = self.log_memory_usage(f"PyPDF2 page {page_num + 1}")
                        if memory_mb and memory_mb > 1800:
                            logger.warning(f"Memory limit approaching, stopping at page {page_num + 1}")
                            break
                    
                    page_header = f"\n{'='*50}\nPAGE {page_num + 1}\n{'='*50}\n"
                    page_text = page.extract_text()
                    if page_text:
                        cleaned_text = page_text.replace('\n\n\n', '\n\n').replace('\t', ' ')
                        full_text += page_header + cleaned_text + "\n"
                except Exception as e:
                    logger.debug(f"Error extracting page {page_num + 1}: {e}")
                    continue
            
            result["debug_info"]["text_length"] = len(full_text)
            logger.debug(f"Extracted {len(full_text)} characters of text")
            
            if len(full_text) > MIN_EXTRACTION_LENGTH:
                result["raw_text"] = full_text
                result["extraction_status"] = "success"
                result["extraction_method"] = "PyPDF2"
                result["debug_info"]["sample_text"] = full_text[:500]
                logger.info(f"PyPDF2 extraction successful")
                return True
            else:
                logger.debug(f"PyPDF2 extracted minimal text ({len(full_text)} chars)")
                return False
                
        except Exception as e:
            logger.debug(f"PyPDF2 extraction failed: {e}")
            return False
        finally:
            gc.collect()
    
    def _extract_with_ocr(self, pdf_content: bytes, result: Dict) -> bool:
        """Extract text using OCR with memory monitoring"""
        try:
            logger.debug("Attempting OCR extraction...")
            self.log_memory_usage("before OCR")
            result["debug_info"]["ocr_attempted"] = True
            
            # Convert PDF to images
            images = self.convert_from_bytes(
                pdf_content, 
                dpi=OCR_DPI, 
                first_page=1, 
                last_page=min(MAX_PAGES_OCR, MAX_PAGES_OCR)
            )
            
            if not images:
                logger.debug("No images could be generated from PDF")
                return False
            
            self.log_memory_usage("after PDF to images conversion")
            
            full_text = ""
            for page_num, image in enumerate(images):
                try:
                    # Check memory before each OCR operation
                    memory_mb = self.log_memory_usage(f"OCR page {page_num + 1}")
                    if memory_mb and memory_mb > 1700:  # Lower threshold for OCR
                        logger.warning(f"Memory limit approaching during OCR, stopping at page {page_num + 1}")
                        break
                    
                    page_header = f"\n{'='*50}\nPAGE {page_num + 1} (OCR)\n{'='*50}\n"
                    page_text = self.pytesseract.image_to_string(
                        image, 
                        lang='eng',
                        config=OCR_CONFIG
                    )
                    if page_text and page_text.strip():
                        cleaned_text = page_text.replace('\n\n\n', '\n\n').replace('\t', ' ')
                        full_text += page_header + cleaned_text + "\n"
                    
                    # Clean up the image from memory
                    del image
                    
                    # Force cleanup every few pages during OCR
                    if page_num % 3 == 0:
                        gc.collect()
                        
                except Exception as e:
                    logger.debug(f"OCR error on page {page_num + 1}: {e}")
                    continue
            
            # Clean up images list
            del images
            gc.collect()
            
            result["debug_info"]["text_length"] = len(full_text)
            logger.debug(f"OCR extracted {len(full_text)} characters of text")
            
            if len(full_text) > MIN_EXTRACTION_LENGTH:
                result["raw_text"] = full_text
                result["extraction_status"] = "success"
                result["extraction_method"] = "OCR"
                result["debug_info"]["sample_text"] = full_text[:500]
                logger.info(f"OCR extraction successful")
                return True
            else:
                logger.debug(f"OCR extracted minimal text ({len(full_text)} chars)")
                return False
                
        except Exception as e:
            logger.debug(f"OCR extraction failed: {e}")
            return False
        finally:
            # Always clean up after OCR
            gc.collect()
    
    # ... rest of the methods remain the same ...
    def _process_financial_table(self, table) -> str:
        """Convert table data to searchable text format"""
        if not table or not any(table):
            return ""
        
        table_text = ""
        
        try:
            for row_num, row in enumerate(table):
                if not row or not any(row):
                    continue
                
                # Clean and join cells
                clean_cells = []
                for cell in row:
                    if cell:
                        cell_str = str(cell).strip()
                        if cell_str and cell_str not in ['', 'None']:
                            clean_cells.append(cell_str)
                
                if clean_cells:
                    # Join with pipes for easy parsing
                    row_text = " | ".join(clean_cells)
                    table_text += f"ROW{row_num}: {row_text}\n"
        except Exception as e:
            logger.error(f"Error processing table: {e}")
            return ""
        
        return table_text
    
    def _create_base_result(self, filename: str) -> Dict:
        """Create base result structure"""
        return {
            "filename": filename,
            "extraction_status": "failed",
            "extraction_method": "none",
            "raw_text": "",
            "debug_info": {
                "total_pages": 0,
                "text_length": 0,
                "tables_found": 0,
                "sample_text": "",
                "ocr_attempted": False,
                "available_methods": self.available_methods
            }
        }
    
    def _create_error_result(self, filename: str, error_msg: str) -> Dict:
        """Create standardized error result"""
        return {
            "filename": filename,
            "extraction_status": "error",
            "extraction_method": "none",
            "raw_text": "",
            "error": error_msg,
            "debug_info": {
                "total_pages": 0,
                "text_length": 0,
                "tables_found": 0,
                "sample_text": "",
                "ocr_attempted": False,
                "available_methods": self.available_methods
            }
        }
    
    def extract_text_from_file(self, file_path: str) -> Dict:
        """
        Extract text from a PDF file on disk
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dict containing extraction results
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return self._create_error_result(
                    file_path.name, 
                    f"File not found: {file_path}"
                )
            
            if file_path.stat().st_size == 0:
                return self._create_error_result(
                    file_path.name, 
                    "File is empty"
                )
            
            with open(file_path, "rb") as f:
                pdf_content = f.read()
            
            return self.extract_text_from_pdf(pdf_content, file_path.name)
            
        except Exception as e:
            return self._create_error_result(
                Path(file_path).name, 
                str(e)
            )