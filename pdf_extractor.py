#!/usr/bin/env python3
"""
Enhanced PDF Extractor with Parallel OCR Processing
Save this as: pdf_extractor.py
Speeds up multi-document analysis significantly
"""

import logging
import concurrent.futures
import multiprocessing
import threading
import time
import gc
from typing import List, Dict, Any
import os
import tempfile

logger = logging.getLogger(__name__)

class ParallelPDFExtractor:
    """Enhanced PDF extractor with parallel OCR processing"""
    
    def __init__(self, max_workers=None):
        """
        Initialize parallel PDF extractor
        
        Args:
            max_workers: Maximum number of parallel workers (default: CPU count - 1)
        """
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"🚀 Parallel PDF Extractor initialized with {self.max_workers} workers")
        
    def extract_multiple_pdfs_parallel(self, file_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract text from multiple PDFs in parallel
        
        Args:
            file_list: List of file info dictionaries with 'path', 'filename', etc.
            
        Returns:
            List of extraction results
        """
        start_time = time.time()
        logger.info(f"🔄 Starting parallel extraction of {len(file_list)} files")
        
        # For single file, use sequential processing
        if len(file_list) == 1:
            return [self._extract_single_pdf(file_list[0])]
        
        results = []
        
        # Process files in parallel using ThreadPoolExecutor for I/O bound OCR
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(file_list))) as executor:
            # Submit all extraction tasks
            future_to_file = {
                executor.submit(self._extract_single_pdf_safe, file_info): file_info 
                for file_info in file_list
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                file_info = future_to_file[future]
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per file
                    if result:
                        results.append(result)
                        logger.info(f"✅ Completed: {file_info['filename']} - {len(result.get('content', '')):,} chars")
                    else:
                        logger.error(f"❌ Failed: {file_info['filename']}")
                        
                except concurrent.futures.TimeoutError:
                    logger.error(f"⏰ Timeout: {file_info['filename']} (>5 minutes)")
                except Exception as e:
                    logger.error(f"❌ Error extracting {file_info['filename']}: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"🎉 Parallel extraction completed: {len(results)}/{len(file_list)} files in {total_time:.1f}s")
        logger.info(f"⚡ Average time per file: {total_time/len(file_list):.1f}s (parallel)")
        
        return results

    def _extract_single_pdf_safe(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thread-safe wrapper for single PDF extraction
        
        Args:
            file_info: File information dictionary
            
        Returns:
            Extraction result dictionary
        """
        try:
            return self._extract_single_pdf(file_info)
        except Exception as e:
            logger.error(f"❌ Safe extraction failed for {file_info.get('filename', 'unknown')}: {e}")
            return None
        finally:
            # Aggressive cleanup in parallel processing
            gc.collect()

    def _extract_single_pdf(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract text from a single PDF file
        
        Args:
            file_info: Dictionary containing file path, filename, etc.
            
        Returns:
            Dictionary with extraction results
        """
        filename = file_info.get('filename', 'unknown')
        file_path = file_info.get('path', '')
        
        logger.info(f"📄 Starting extraction: {filename}")
        
        try:
            # Read file content
            with open(file_path, 'rb') as f:
                pdf_content = f.read()
            
            # Try different extraction methods in order of speed/efficiency
            result = self._try_extraction_methods(pdf_content, filename)
            
            if result and result.get("extraction_status") == "success":
                content = result.get("raw_text", "")
                if len(content.strip()) > 100:
                    return {
                        'filename': filename,
                        'date': file_info.get('date'),
                        'content': content,
                        'metadata': {
                            'file_size': file_info.get('size', 0),
                            'extraction_method': result.get("extraction_method", "unknown"),
                            'processing_time': result.get("processing_time", 0)
                        }
                    }
            
            logger.warning(f"⚠️ Insufficient content extracted from {filename}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error extracting {filename}: {e}")
            return None

    def _try_extraction_methods(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Try different PDF extraction methods in order of efficiency
        
        Args:
            pdf_content: PDF file content as bytes
            filename: Filename for logging
            
        Returns:
            Extraction result dictionary
        """
        start_time = time.time()
        
        # Method 1: PyPDF2 (fastest)
        result = self._try_pypdf2(pdf_content, filename)
        if result["extraction_status"] == "success":
            result["processing_time"] = time.time() - start_time
            return result
        
        # Method 2: pdfplumber (medium speed, better accuracy)
        result = self._try_pdfplumber(pdf_content, filename)
        if result["extraction_status"] == "success":
            result["processing_time"] = time.time() - start_time
            return result
        
        # Method 3: OCR (slowest, most comprehensive)
        result = self._try_ocr_optimized(pdf_content, filename)
        result["processing_time"] = time.time() - start_time
        return result

    def _try_pypdf2(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Try PyPDF2 extraction (fastest method)"""
        try:
            import PyPDF2
            import io
            
            with io.BytesIO(pdf_content) as pdf_stream:
                pdf_reader = PyPDF2.PdfReader(pdf_stream)
                text_parts = []
                
                for page_num, page in enumerate(pdf_reader.pages[:50]):  # Limit to 50 pages
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_parts.append(page_text)
                    except Exception:
                        continue
                
                combined_text = "\n".join(text_parts)
                
                if len(combined_text.strip()) > 500:  # Minimum threshold
                    return {
                        "extraction_status": "success",
                        "raw_text": combined_text,
                        "extraction_method": "pypdf2"
                    }
        
        except Exception as e:
            logger.debug(f"PyPDF2 failed for {filename}: {e}")
        
        return {"extraction_status": "failed", "raw_text": "", "extraction_method": "pypdf2"}

    def _try_pdfplumber(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Try pdfplumber extraction (medium speed, better accuracy)"""
        try:
            import pdfplumber
            import io
            
            text_parts = []
            
            with io.BytesIO(pdf_content) as pdf_stream:
                with pdfplumber.open(pdf_stream) as pdf:
                    for page_num, page in enumerate(pdf.pages[:50]):  # Limit to 50 pages
                        try:
                            page_text = page.extract_text()
                            if page_text and page_text.strip():
                                text_parts.append(page_text)
                        except Exception:
                            continue
            
            combined_text = "\n".join(text_parts)
            
            if len(combined_text.strip()) > 500:
                return {
                    "extraction_status": "success",
                    "raw_text": combined_text,
                    "extraction_method": "pdfplumber"
                }
                
        except Exception as e:
            logger.debug(f"pdfplumber failed for {filename}: {e}")
        
        return {"extraction_status": "failed", "raw_text": "", "extraction_method": "pdfplumber"}

    def _try_ocr_optimized(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Optimized OCR extraction for parallel processing
        """
        try:
            import pytesseract
            from pdf2image import convert_from_bytes
            import gc
            
            logger.info(f"🔄 Starting OCR for {filename}")
            
            # Convert PDF to images with optimized settings for speed
            images = convert_from_bytes(
                pdf_content,
                dpi=150,  # Lower DPI for speed
                first_page=1,
                last_page=20,  # Limit pages for speed
                fmt='jpeg',
                jpegopt={'quality': 80, 'progressive': True}
            )
            
            text_parts = []
            
            # Process images with OCR
            for i, image in enumerate(images):
                try:
                    # Use faster OCR configuration
                    text = pytesseract.image_to_string(
                        image, 
                        config='--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?()-:;'
                    )
                    
                    if text and text.strip():
                        text_parts.append(text)
                    
                    # Cleanup image immediately
                    del image
                    
                    # Memory management for parallel processing
                    if i % 5 == 0:
                        gc.collect()
                        
                except Exception as page_error:
                    logger.debug(f"OCR page {i+1} failed: {page_error}")
                    continue
            
            # Cleanup images
            del images
            gc.collect()
            
            combined_text = "\n".join(text_parts)
            
            if len(combined_text.strip()) > 100:
                logger.info(f"✅ OCR successful for {filename}: {len(combined_text)} chars")
                return {
                    "extraction_status": "success",
                    "raw_text": combined_text,
                    "extraction_method": "ocr_optimized"
                }
            else:
                logger.warning(f"⚠️ OCR produced insufficient text for {filename}")
                
        except Exception as e:
            logger.error(f"❌ OCR failed for {filename}: {e}")
        finally:
            gc.collect()
        
        return {
            "extraction_status": "failed",
            "raw_text": "",
            "extraction_method": "ocr_optimized"
        }


class PDFExtractor:
    """
    Legacy single-file PDF extractor for backwards compatibility
    """
    
    def __init__(self):
        self.parallel_extractor = ParallelPDFExtractor()
    
    def extract_text_from_pdf(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Extract text from a single PDF - legacy method
        
        Args:
            pdf_content: PDF file content as bytes
            filename: Filename for logging
            
        Returns:
            Extraction result dictionary
        """
        # Use the parallel extractor's method for consistency
        return self.parallel_extractor._try_extraction_methods(pdf_content, filename)


# Integration functions for main.py
def extract_multiple_files_parallel(file_list: List[Dict[str, Any]], max_workers: int = None) -> List[Dict[str, Any]]:
    """
    Parallel extraction function to replace your sequential extraction
    
    Args:
        file_list: List of file dictionaries from your download process
        max_workers: Number of parallel workers (default: CPU count - 1)
        
    Returns:
        List of extracted content dictionaries
    """
    extractor = ParallelPDFExtractor(max_workers=max_workers)
    return extractor.extract_multiple_pdfs_parallel(file_list)


def extract_files_in_batches(file_list: List[Dict[str, Any]], batch_size: int = 4, max_workers: int = None) -> List[Dict[str, Any]]:
    """
    Process files in batches to manage memory usage
    
    Args:
        file_list: List of files to process
        batch_size: Number of files per batch
        max_workers: Workers per batch
        
    Returns:
        List of all extraction results
    """
    all_results = []
    total_files = len(file_list)
    
    logger.info(f"📦 Processing {total_files} files in batches of {batch_size}")
    
    for i in range(0, total_files, batch_size):
        batch = file_list[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_files + batch_size - 1) // batch_size
        
        logger.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} files)")
        
        batch_results = extract_multiple_files_parallel(batch, max_workers=max_workers)
        all_results.extend(batch_results)
        
        # Memory cleanup between batches
        gc.collect()
        
        logger.info(f"✅ Batch {batch_num} completed: {len(batch_results)}/{len(batch)} files successful")
    
    logger.info(f"🎉 All batches completed: {len(all_results)}/{total_files} files total")
    return all_results


if __name__ == "__main__":
    # Test the parallel extractor
    print("Testing Parallel PDF Extractor...")
    
    # This would be replaced with actual file paths in production
    test_files = [
        {
            'filename': 'test1.pdf',
            'path': '/path/to/test1.pdf',
            'size': 1000000,
            'date': '2024-01-01'
        },
        {
            'filename': 'test2.pdf', 
            'path': '/path/to/test2.pdf',
            'size': 1500000,
            'date': '2024-01-01'
        }
    ]
    
    # Test parallel extraction
    results = extract_multiple_files_parallel(test_files)
    print(f"Parallel extraction completed: {len(results)} files processed")
    
    print("Parallel PDF Extractor test completed.")