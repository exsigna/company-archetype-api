#!/usr/bin/env python3
"""
Content Processor for Strategic Analysis
Processes extracted PDF content and categorizes it into strategy, governance, risk, and audit themes
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from config import (
    STRATEGY_KEYWORDS, GOVERNANCE_KEYWORDS, RISK_KEYWORDS, AUDIT_KEYWORDS,
    MAX_TOKENS_PER_CHUNK, TOTAL_TOKEN_LIMIT, MIN_CONTENT_LENGTH,
    MAX_CONTENT_SECTIONS_PER_CATEGORY
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentProcessor:
    """
    Processes and categorizes extracted content from company documents
    """
    
    def __init__(self):
        """Initialize the content processor"""
        self.keyword_categories = {
            "strategy": STRATEGY_KEYWORDS,
            "governance": GOVERNANCE_KEYWORDS,
            "risk": RISK_KEYWORDS,
            "audit": AUDIT_KEYWORDS
        }
        
        # Compile regex patterns for efficiency
        self.keyword_patterns = {}
        for category, keywords in self.keyword_categories.items():
            pattern = r'\b(?:' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            self.keyword_patterns[category] = re.compile(pattern, re.IGNORECASE)
        
        logger.info("Content processor initialized")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\n\s*Page \d+.*?\n', '\n', text, flags=re.IGNORECASE)
        
        # Remove common PDF artifacts
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        # Normalize quotes and dashes (using unicode escapes for safety)
        text = re.sub(r'[\u201c\u201d]', '"', text)  # Smart double quotes " "
        text = re.sub(r'[\u2018\u2019]', "'", text)  # Smart single quotes ' '
        text = re.sub(r'[\u2013\u2014]', '-', text)  # Em and en dashes – —
        
        # Clean up multiple periods
        text = re.sub(r'\.{3,}', '...', text)
        
        return text.strip()
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text
        
        Args:
            text: Text to split into sentences
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting on periods, exclamation marks, and question marks
        # followed by whitespace and capital letter or end of string
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) >= MIN_CONTENT_LENGTH:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def extract_paragraphs(self, text: str) -> List[str]:
        """
        Extract paragraphs from text
        
        Args:
            text: Text to split into paragraphs
            
        Returns:
            List of paragraphs
        """
        # Split on double newlines or multiple whitespace
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
        
        # Clean and filter paragraphs
        cleaned_paragraphs = []
        for para in paragraphs:
            para = self.clean_text(para)
            if len(para) >= MIN_CONTENT_LENGTH:
                cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs
    
    def categorize_content(self, text: str) -> Dict[str, List[str]]:
        """
        Categorize text content by theme (strategy, governance, risk, audit)
        
        Args:
            text: Text to categorize
            
        Returns:
            Dictionary with categorized content
        """
        categorized = {
            "strategy": [],
            "governance": [],
            "risk": [],
            "audit": []
        }
        
        # Extract paragraphs for categorization
        paragraphs = self.extract_paragraphs(text)
        
        for paragraph in paragraphs:
            # Count keyword matches for each category
            category_scores = {}
            
            for category, pattern in self.keyword_patterns.items():
                matches = pattern.findall(paragraph)
                category_scores[category] = len(matches)
            
            # Assign paragraph to category with highest score
            if any(score > 0 for score in category_scores.values()):
                best_category = max(category_scores, key=category_scores.get)
                
                # Limit sections per category to avoid token overload
                if len(categorized[best_category]) < MAX_CONTENT_SECTIONS_PER_CATEGORY:
                    categorized[best_category].append(paragraph)
        
        return categorized
    
    def extract_key_metrics(self, text: str) -> Dict[str, Any]:
        """
        Extract key financial and business metrics from text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of extracted metrics
        """
        metrics = {
            "financial_figures": [],
            "percentages": [],
            "dates": [],
            "ratios": [],
            "currencies": []
        }
        
        # Financial figures (£, $, €, millions, billions)
        money_pattern = r'[£$€]\s*\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:million|billion|m|bn))?'
        metrics["financial_figures"] = re.findall(money_pattern, text, re.IGNORECASE)
        
        # Percentages
        percentage_pattern = r'\d+(?:\.\d+)?%'
        metrics["percentages"] = re.findall(percentage_pattern, text)
        
        # Dates (various formats)
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{4}',  # DD/MM/YYYY or DD-MM-YYYY
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',  # YYYY/MM/DD or YYYY-MM-DD
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
            r'\d{4}'  # Just years
        ]
        
        for pattern in date_patterns:
            metrics["dates"].extend(re.findall(pattern, text, re.IGNORECASE))
        
        # Financial ratios
        ratio_pattern = r'\d+(?:\.\d+)?:\d+(?:\.\d+)?'
        metrics["ratios"] = re.findall(ratio_pattern, text)
        
        # Remove duplicates and limit results
        for key in metrics:
            metrics[key] = list(set(metrics[key]))[:20]  # Limit to 20 items each
        
        return metrics
    
    def analyze_sentiment_indicators(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment indicators in the text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis
        """
        positive_words = [
            "growth", "increase", "improve", "strong", "robust", "positive", 
            "successful", "expand", "opportunity", "confident", "optimistic",
            "profitable", "efficient", "innovative", "competitive", "resilient"
        ]
        
        negative_words = [
            "decline", "decrease", "reduce", "weak", "poor", "negative",
            "challenge", "risk", "concern", "difficult", "loss", "falling",
            "uncertainty", "volatile", "pressure", "constraint", "adverse"
        ]
        
        risk_words = [
            "risk", "threat", "vulnerability", "exposure", "uncertainty",
            "volatility", "compliance", "regulatory", "operational", "credit",
            "market", "liquidity", "cyber", "reputation", "strategic"
        ]
        
        # Count occurrences
        text_lower = text.lower()
        
        positive_count = sum(text_lower.count(word) for word in positive_words)
        negative_count = sum(text_lower.count(word) for word in negative_words)
        risk_count = sum(text_lower.count(word) for word in risk_words)
        
        total_words = len(text.split())
        
        sentiment_analysis = {
            "positive_indicators": positive_count,
            "negative_indicators": negative_count,
            "risk_indicators": risk_count,
            "total_words": total_words,
            "sentiment_ratio": (positive_count - negative_count) / max(total_words, 1),
            "risk_density": risk_count / max(total_words, 1)
        }
        
        return sentiment_analysis
    
    def extract_executive_summary_sections(self, text: str) -> List[str]:
        """
        Identify and extract sections that appear to be executive summaries
        
        Args:
            text: Full document text
            
        Returns:
            List of potential executive summary sections
        """
        summary_sections = []
        
        # Common section headers for executive summaries (using safer patterns)
        summary_headers = [
            r'executive summary',
            r'summary', 
            r'overview',
            r'key highlights',
            r'chairman.{0,2}s statement',      # chairman's or chairmans statement
            r'chief executive.{0,2}s review',  # chief executive's review
            r'ceo.{0,2}s statement',           # ceo's statement
            r'business review',
            r'strategic report'
        ]
        
        # Create pattern to find these sections
        header_pattern = r'\n\s*(' + '|'.join(summary_headers) + r')\s*\n'
        
        # Split text by potential summary headers
        sections = re.split(header_pattern, text, flags=re.IGNORECASE)
        
        for i, section in enumerate(sections):
            # Look for sections that follow summary headers
            if i > 0 and re.match(r'|'.join(summary_headers), sections[i-1], re.IGNORECASE):
                # Take the first reasonable chunk (up to 2000 words)
                words = section.split()
                if len(words) > 50:  # Minimum threshold
                    summary_text = ' '.join(words[:2000])
                    summary_sections.append(self.clean_text(summary_text))
        
        return summary_sections
    
    def process_document_content(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process complete document content and extract structured information
        
        Args:
            content: Full document text content
            metadata: Optional metadata about the document
            
        Returns:
            Comprehensive analysis results
        """
        if not content:
            logger.warning("Empty content provided for processing")
            return self._empty_analysis_result()
        
        logger.info(f"Processing document content ({len(content)} characters)")
        
        # Clean the content
        cleaned_content = self.clean_text(content)
        
        if len(cleaned_content) < MIN_CONTENT_LENGTH:
            logger.warning("Content too short after cleaning")
            return self._empty_analysis_result()
        
        # Perform various analyses
        analysis_result = {
            "metadata": metadata or {},
            "content_stats": {
                "original_length": len(content),
                "cleaned_length": len(cleaned_content),
                "word_count": len(cleaned_content.split()),
                "paragraph_count": len(self.extract_paragraphs(cleaned_content)),
                "sentence_count": len(self.extract_sentences(cleaned_content))
            },
            "categorized_content": self.categorize_content(cleaned_content),
            "key_metrics": self.extract_key_metrics(cleaned_content),
            "sentiment_analysis": self.analyze_sentiment_indicators(cleaned_content),
            "executive_summaries": self.extract_executive_summary_sections(cleaned_content),
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # Add category statistics
        category_stats = {}
        for category, sections in analysis_result["categorized_content"].items():
            category_stats[category] = {
                "section_count": len(sections),
                "total_words": sum(len(section.split()) for section in sections),
                "avg_section_length": sum(len(section.split()) for section in sections) / max(len(sections), 1)
            }
        
        analysis_result["category_stats"] = category_stats
        
        logger.info("Document processing completed successfully")
        return analysis_result
    
    def _empty_analysis_result(self) -> Dict[str, Any]:
        """
        Return empty analysis result structure
        
        Returns:
            Empty analysis result dictionary
        """
        return {
            "metadata": {},
            "content_stats": {
                "original_length": 0,
                "cleaned_length": 0,
                "word_count": 0,
                "paragraph_count": 0,
                "sentence_count": 0
            },
            "categorized_content": {
                "strategy": [],
                "governance": [],
                "risk": [],
                "audit": []
            },
            "key_metrics": {
                "financial_figures": [],
                "percentages": [],
                "dates": [],
                "ratios": [],
                "currencies": []
            },
            "sentiment_analysis": {
                "positive_indicators": 0,
                "negative_indicators": 0,
                "risk_indicators": 0,
                "total_words": 0,
                "sentiment_ratio": 0.0,
                "risk_density": 0.0
            },
            "executive_summaries": [],
            "category_stats": {
                "strategy": {"section_count": 0, "total_words": 0, "avg_section_length": 0},
                "governance": {"section_count": 0, "total_words": 0, "avg_section_length": 0},
                "risk": {"section_count": 0, "total_words": 0, "avg_section_length": 0},
                "audit": {"section_count": 0, "total_words": 0, "avg_section_length": 0}
            },
            "processing_timestamp": datetime.now().isoformat()
        }
    
    def combine_multiple_documents(self, document_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine analysis results from multiple documents
        
        Args:
            document_analyses: List of individual document analysis results
            
        Returns:
            Combined analysis result
        """
        if not document_analyses:
            return self._empty_analysis_result()
        
        logger.info(f"Combining {len(document_analyses)} document analyses")
        
        combined = {
            "metadata": {
                "document_count": len(document_analyses),
                "combined_timestamp": datetime.now().isoformat()
            },
            "content_stats": {
                "total_original_length": 0,
                "total_cleaned_length": 0,
                "total_word_count": 0,
                "total_paragraph_count": 0,
                "total_sentence_count": 0
            },
            "categorized_content": {
                "strategy": [],
                "governance": [],
                "risk": [],
                "audit": []
            },
            "key_metrics": {
                "financial_figures": [],
                "percentages": [],
                "dates": [],
                "ratios": [],
                "currencies": []
            },
            "sentiment_analysis": {
                "positive_indicators": 0,
                "negative_indicators": 0,
                "risk_indicators": 0,
                "total_words": 0,
                "sentiment_ratio": 0.0,
                "risk_density": 0.0
            },
            "executive_summaries": [],
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # Combine data from all documents
        for doc_analysis in document_analyses:
            # Content stats
            stats = doc_analysis.get("content_stats", {})
            combined["content_stats"]["total_original_length"] += stats.get("original_length", 0)
            combined["content_stats"]["total_cleaned_length"] += stats.get("cleaned_length", 0)
            combined["content_stats"]["total_word_count"] += stats.get("word_count", 0)
            combined["content_stats"]["total_paragraph_count"] += stats.get("paragraph_count", 0)
            combined["content_stats"]["total_sentence_count"] += stats.get("sentence_count", 0)
            
            # Categorized content
            categorized = doc_analysis.get("categorized_content", {})
            for category in combined["categorized_content"]:
                combined["categorized_content"][category].extend(categorized.get(category, []))
            
            # Key metrics
            metrics = doc_analysis.get("key_metrics", {})
            for metric_type in combined["key_metrics"]:
                combined["key_metrics"][metric_type].extend(metrics.get(metric_type, []))
            
            # Sentiment analysis
            sentiment = doc_analysis.get("sentiment_analysis", {})
            combined["sentiment_analysis"]["positive_indicators"] += sentiment.get("positive_indicators", 0)
            combined["sentiment_analysis"]["negative_indicators"] += sentiment.get("negative_indicators", 0)
            combined["sentiment_analysis"]["risk_indicators"] += sentiment.get("risk_indicators", 0)
            combined["sentiment_analysis"]["total_words"] += sentiment.get("total_words", 0)
            
            # Executive summaries
            combined["executive_summaries"].extend(doc_analysis.get("executive_summaries", []))
        
        # Calculate combined sentiment ratios
        total_words = combined["sentiment_analysis"]["total_words"]
        if total_words > 0:
            pos = combined["sentiment_analysis"]["positive_indicators"]
            neg = combined["sentiment_analysis"]["negative_indicators"]
            risk = combined["sentiment_analysis"]["risk_indicators"]
            
            combined["sentiment_analysis"]["sentiment_ratio"] = (pos - neg) / total_words
            combined["sentiment_analysis"]["risk_density"] = risk / total_words
        
        # Remove duplicates from metrics
        for metric_type in combined["key_metrics"]:
            combined["key_metrics"][metric_type] = list(set(combined["key_metrics"][metric_type]))
        
        # Limit content sections to avoid token overflow
        for category in combined["categorized_content"]:
            sections = combined["categorized_content"][category]
            if len(sections) > MAX_CONTENT_SECTIONS_PER_CATEGORY:
                combined["categorized_content"][category] = sections[:MAX_CONTENT_SECTIONS_PER_CATEGORY]
        
        logger.info("Document combination completed successfully")
        return combined
    
    def save_analysis_to_file(self, analysis: Dict[str, Any], filepath: Path) -> bool:
        """
        Save analysis results to JSON file
        
        Args:
            analysis: Analysis results to save
            filepath: Path to save the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Analysis saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save analysis to {filepath}: {e}")
            return False
    
    def load_analysis_from_file(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """
        Load analysis results from JSON file
        
        Args:
            filepath: Path to the file to load
            
        Returns:
            Analysis results or None if failed
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
            
            logger.info(f"Analysis loaded from {filepath}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to load analysis from {filepath}: {e}")
            return None


# Convenience function for quick content processing
def process_text_content(text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Process text content with default settings
    
    Args:
        text: Text content to process
        metadata: Optional metadata
        
    Returns:
        Analysis results
    """
    processor = ContentProcessor()
    return processor.process_document_content(text, metadata)


if __name__ == "__main__":
    # Test the content processor
    test_text = """
    Executive Summary
    
    The company has demonstrated strong strategic growth in the digital lending sector.
    Our risk management framework has been enhanced to address operational challenges.
    The board of directors has provided effective governance oversight throughout the year.
    Internal audit functions have identified key areas for improvement in our control environment.
    
    Financial Performance
    
    Revenue increased by 15.2% to £45.6 million, reflecting our strategic expansion.
    The risk-adjusted return on equity improved to 12.5%, demonstrating effective risk management.
    Our governance committee met quarterly to review strategic initiatives.
    """
    
    processor = ContentProcessor()
    results = processor.process_document_content(test_text, {"test": True})
    
    print("Content Processing Test Results:")
    print(f"Word count: {results['content_stats']['word_count']}")
    print(f"Categories found: {list(results['categorized_content'].keys())}")
    
    for category, sections in results['categorized_content'].items():
        if sections:
            print(f"{category.title()}: {len(sections)} sections")
    
    print(f"Financial figures found: {results['key_metrics']['financial_figures']}")
    print(f"Percentages found: {results['key_metrics']['percentages']}")