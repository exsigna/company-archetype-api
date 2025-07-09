def extract_content_from_files(downloaded_files):
    """
    Extract content from downloaded files using parallel processing when beneficial
    """
    try:
        logger.info(f"📄 Starting content extraction from {len(downloaded_files)} files")
        
        # Use parallel extraction for multiple files, sequential for single files
        if len(downloaded_files) > 1:
            logger.info(f"🚀 Using parallel extraction for {len(downloaded_files)} files")
            extracted_content = extract_multiple_files_parallel(downloaded_files)
        else:
            logger.info("📄 Using sequential extraction for single file")
            # Use the parallel extractor's single file method for consistency
            extractor = ParallelPDFExtractor()
            result = extractor._extract_single_pdf(downloaded_files[0])
            extracted_content = [result] if result else []
        
        # Filter out None results and log statistics
        valid_content = [content for content in extracted_content if content is not None]
        
        logger.info(f"✅ Content extraction completed: {len(valid_content)}/{len(downloaded_files)} files successful")
        
        if len(valid_content) != len(downloaded_files):
            failed_count = len(downloaded_files) - len(valid_content)
            logger.warning(f"⚠️ {failed_count} files failed extraction")
        
        return valid_content
        
    except Exception as e:
        logger.error(f"❌ Error in content extraction: {e}")
        return []


def extract_content_from_files_legacy(downloaded_files):
    """
    Legacy sequential extraction method - kept for fallback
    """
    extracted_content = []
    
    for file_info in downloaded_files:
        try:
            logger.info(f"📄 Extracting content from: {file_info['filename']}")
            
            # Read PDF content
            with open(file_info['path'], 'rb') as f:
                pdf_content = f.read()
            
            # Extract using PDFExtractor
            extraction_result = pdf_extractor.extract_text_from_pdf(
                pdf_content, file_info['filename']
            )
            
            if extraction_result["extraction_status"] == "success":
                content = extraction_result.get("raw_text", "")
                if content and len(content.strip()) > 100:  # Minimum content check
                    extracted_content.append({
                        'filename': file_info['filename'],
                        'date': file_info['date'],
                        'content': content,
                        'metadata': {
                            'file_size': file_info['size'],
                            'extraction_method': extraction_result["extraction_method"]
                        }
                    })
                    logger.info(f"✅ Successfully extracted {len(content):,} characters from {file_info['filename']}")
                else:
                    logger.warning(f"⚠️ Insufficient content extracted from {file_info['filename']}")
            else:
                logger.error(f"❌ Extraction failed for {file_info['filename']}")
                
        except Exception as e:
            logger.error(f"❌ Error extracting content from {file_info['filename']}: {e}")
    
    return extracted_content