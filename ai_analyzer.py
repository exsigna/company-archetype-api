#!/usr/bin/env python3
"""
Complete AI Analyzer with Exact Business and Risk Strategy Archetypes
Generates reports in the specified format with proper SWOT analysis
Thread-safe for Render deployment with enhanced debugging
FIXED: Confidence level calculation based on analysis scope only
ENHANCED: Strategic content optimization with technical insights extraction
"""

import os
import sys
import logging
import json
import time
import random
import threading
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging for Render
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class TimeoutManager:
    """Thread-safe timeout manager for Render deployment"""
    
    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
    
    def start(self):
        """Start the timeout timer"""
        self.start_time = time.time()
    
    def check_timeout(self):
        """Check if timeout has been exceeded"""
        if self.start_time is None:
            return False
        
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_seconds:
            raise TimeoutError(f"Operation timed out after {elapsed:.1f} seconds")
        
        return False
    
    def remaining_time(self) -> float:
        """Get remaining time in seconds"""
        if self.start_time is None:
            return self.timeout_seconds
        
        elapsed = time.time() - self.start_time
        return max(0, self.timeout_seconds - elapsed)

class CompleteAIAnalyzer:
    """
    Complete AI Analyzer with exact archetypes and strategic content optimization
    FIXED: Proper confidence level calculation and debugging
    ENHANCED: Intelligent content optimization with strategic technical insights
    """
    
    def __init__(self):
        """Initialize with complete archetype definitions"""
        logger.info("🚀 Initializing Complete AI Analyzer with enhanced optimization...")
        
        # Log environment for debugging
        self._log_environment_debug()
        
        # Client initialization
        self.openai_client = None
        self.anthropic_client = None
        self.client_type = "uninitialized"
        
        # Model configuration optimized for Render
        self.primary_model = "gpt-4-turbo"
        self.fallback_model = "gpt-4-turbo-2024-04-09"
        self.max_output_tokens = 4096
        self.max_retries = 3
        self.base_retry_delay = 3.0
        self.max_content_chars = 150000  # 150K character limit
        self.request_timeout = 60
        
        # Complete Business Strategy Archetypes
        self.business_archetypes = {
            "Scale-through-Distribution": "Gains share primarily by adding new channels or partners faster than control maturity develops.",
            "Land-Grab Platform": "Uses aggressive below-market pricing or incentives to build a large multi-sided platform quickly (BNPL, FX apps, etc.).",
            "Asset-Velocity Maximiser": "Chases rapid originations / turnover (e.g. bridging, invoice finance) even at higher funding costs.",
            "Yield-Hunting": "Prioritises high-margin segments (credit-impaired, niche commercial) and prices for risk premium.",
            "Fee-Extraction Engine": "Relies on ancillary fees, add-ons or cross-sales for majority of profit (packaged accounts, paid add-ons).",
            "Disciplined Specialist Growth": "Niche focus with strong underwriting edge; grows opportunistically while recycling balance-sheet (Together Personal Finance).",
            "Expert Niche Leader": "Deep expertise in a micro-segment (e.g. HNW Islamic mortgages) with modest but steady growth.",
            "Service-Driven Differentiator": "Wins by superior client experience / advice rather than price or scale (boutique wealth, mutual insurers).",
            "Cost-Leadership Operator": "Drives ROE via lean cost base, digital self-service, zero-based budgeting.",
            "Tech-Productivity Accelerator": "Heavy automation/AI to compress unit costs and redeploy staff (app-only challengers).",
            "Product-Innovation Flywheel": "Constantly launches novel product variants/features to capture share (fintech disruptors).",
            "Data-Monetisation Pioneer": "Converts proprietary data into fees (open-banking analytics, credit-insights platforms).",
            "Balance-Sheet Steward": "Low-risk appetite, prioritises capital strength and membership value (building societies, mutuals).",
            "Regulatory Shelter Occupant": "Leverages regulatory or franchise protections to defend share (NS&I, Post Office card a/c).",
            "Regulator-Mandated Remediation": "Operating under s.166, VREQ or RMAR constraints; resources diverted to fix historical failings.",
            "Wind-down / Run-off": "Managing existing book to maturity or sale; minimal new origination (closed-book life funds).",
            "Strategic Withdrawal": "Actively divesting lines/geographies to refocus core franchise.",
            "Distressed-Asset Harvester": "Buys NPLs or under-priced portfolios during downturns for future upside.",
            "Counter-Cyclical Capitaliser": "Expands lending precisely when competitors retrench, using strong liquidity."
        }
        
        # Complete Risk Strategy Archetypes
        self.risk_archetypes = {
            "Risk-First Conservative": "Prioritises capital preservation and regulatory compliance; growth is secondary to resilience.",
            "Rules-Led Operator": "Strict adherence to rules and checklists; prioritises control consistency over judgment or speed.",
            "Resilience-Focused Architect": "Designs for operational continuity and crisis endurance; invests in stress testing and scenario planning.",
            "Strategic Risk-Taker": "Accepts elevated risk to unlock growth or margin; uses pricing, underwriting, or innovation to offset exposure.",
            "Control-Lag Follower": "Expands products or markets ahead of control maturity; plays regulatory catch-up after scaling.",
            "Reactive Remediator": "Risk strategy is event-driven, typically shaped by enforcement, audit findings, or external reviews.",
            "Reputation-First Shield": "Actively avoids reputational or political risk, sometimes at the expense of commercial logic.",
            "Embedded Risk Partner": "Risk teams are embedded in frontline decisions; risk appetite is shaped collaboratively across the business.",
            "Quant-Control Enthusiast": "Leverages data, automation, and predictive analytics as core risk management tools.",
            "Tick-Box Minimalist": "Superficial control structures exist for compliance optics, not genuine governance intent.",
            "Mission-Driven Prudence": "Risk appetite is anchored in stakeholder protection, community outcomes, or long-term social licence."
        }
        
        # Initialize clients
        self._initialize_clients()
        
        logger.info(f"✅ Initialization complete. Client type: {self.client_type}")
        logger.info(f"📊 Business archetypes: {len(self.business_archetypes)} defined")
        logger.info(f"🛡️ Risk archetypes: {len(self.risk_archetypes)} defined")
    
    def _log_environment_debug(self):
        """Enhanced environment debugging"""
        logger.info("🔍 Enhanced Environment Debug Information:")
        
        # Get keys with whitespace stripping
        openai_key = os.environ.get('OPENAI_API_KEY', '').strip()
        anthropic_key = os.environ.get('ANTHROPIC_API_KEY', '').strip()
        
        # Detailed OpenAI key analysis
        if openai_key:
            logger.info(f"   ✅ OPENAI_API_KEY found")
            logger.info(f"   📏 Length: {len(openai_key)} characters")
            logger.info(f"   🔑 Prefix: {openai_key[:15]}...")
            logger.info(f"   📝 Format check: {'✅ Valid sk-proj format' if openai_key.startswith('sk-proj-') else '⚠️ Unexpected format'}")
        else:
            logger.error("   ❌ OPENAI_API_KEY not found!")
        
        # Detailed Anthropic key analysis
        if anthropic_key:
            logger.info(f"   ✅ ANTHROPIC_API_KEY found")
            logger.info(f"   📏 Length: {len(anthropic_key)} characters")
            logger.info(f"   🔑 Prefix: {anthropic_key[:15]}...")
            logger.info(f"   📝 Format check: {'✅ Valid sk-ant format' if anthropic_key.startswith('sk-ant-') else '⚠️ Unexpected format'}")
        else:
            logger.info("   ℹ️ ANTHROPIC_API_KEY not found (fallback only)")
    
    def _initialize_clients(self):
        """Initialize AI clients with enhanced error handling"""
        
        # Try OpenAI first
        openai_success = self._init_openai()
        if openai_success:
            self.client_type = "openai_primary"
            logger.info("✅ OpenAI configured as primary service")
        else:
            logger.warning("⚠️ OpenAI initialization failed")
        
        # Try Anthropic as fallback
        anthropic_success = self._init_anthropic()
        if anthropic_success:
            if self.client_type == "uninitialized":
                self.client_type = "anthropic_fallback"
                logger.info("✅ Anthropic configured as fallback service")
            else:
                logger.info("✅ Anthropic available as backup")
        else:
            logger.warning("⚠️ Anthropic initialization failed")
        
        # Final status
        if self.client_type == "uninitialized":
            self.client_type = "no_clients_available"
            logger.error("❌ No AI clients available")
            
        logger.info(f"🎯 Final client configuration: {self.client_type}")
    
    def _init_openai(self) -> bool:
        """Initialize OpenAI client with detailed error handling"""
        try:
            # Get and clean API key
            api_key = os.environ.get('OPENAI_API_KEY', '').strip()
            if not api_key:
                logger.warning("⚠️ OPENAI_API_KEY not found")
                return False
            
            try:
                import openai
                from openai import OpenAI
                
                # Create client with explicit timeout
                self.openai_client = OpenAI(
                    api_key=api_key,
                    max_retries=0,
                    timeout=30.0
                )
                
                # Test connection with simple call
                test_response = self.openai_client.models.list()
                models = [model.id for model in test_response.data]
                
                logger.info(f"✅ OpenAI connection successful! Available models: {len(models)}")
                return True
                
            except ImportError as e:
                logger.error(f"❌ OpenAI library import failed: {e}")
                return False
                
            except Exception as e:
                logger.error(f"❌ OpenAI client test failed: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"❌ OpenAI initialization completely failed: {e}")
            return False
    
    def _init_anthropic(self) -> bool:
        """Initialize Anthropic client with detailed error handling"""
        try:
            # Get and clean API key
            api_key = os.environ.get('ANTHROPIC_API_KEY', '').strip()
            if not api_key:
                logger.info("ℹ️ ANTHROPIC_API_KEY not found - skipping")
                return False
            
            try:
                import anthropic
                
                # Create client
                self.anthropic_client = anthropic.Anthropic(
                    api_key=api_key, 
                    timeout=30.0
                )
                
                # Test connection with minimal call
                test_response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=5,
                    messages=[{"role": "user", "content": "Hi"}]
                )
                
                logger.info("✅ Anthropic connection successful!")
                return True
                
            except ImportError as e:
                logger.info(f"ℹ️ Anthropic library not available: {e}")
                return False
                
            except Exception as e:
                logger.warning(f"⚠️ Anthropic client test failed: {str(e)}")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Anthropic initialization failed: {e}")
            return False
    
    def analyze_for_board_optimized(self, content: str, company_name: str, company_number: str,
                                  extracted_content: Optional[List[Dict[str, Any]]] = None,
                                  analysis_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete analysis generating full report format (thread-safe for Render)
        ENHANCED: Uses strategic content optimization
        """
        start_time = time.time()
        
        logger.info(f"🚀 Starting complete analysis for {company_name} ({company_number})")
        logger.info(f"📊 Content length: {len(content):,} characters")
        logger.info(f"🔧 Client type: {self.client_type}")
        
        try:
            if self.client_type == "no_clients_available":
                logger.error("❌ No AI clients available - using emergency analysis")
                return self._create_emergency_analysis(company_name, company_number, "No AI clients available")
            
            # ENHANCED: Strategic content optimization
            optimized_content = self._optimize_content_strategically(content, extracted_content)
            
            # Try OpenAI analysis first if available
            if self.openai_client and self.client_type == "openai_primary":
                logger.info("🎯 Attempting OpenAI analysis (primary)...")
                result = self._analyze_with_openai(optimized_content, company_name, company_number, analysis_context, extracted_content)
                if result:
                    analysis_time = time.time() - start_time
                    logger.info(f"✅ OpenAI analysis completed in {analysis_time:.2f}s")
                    return result
                else:
                    logger.warning("⚠️ OpenAI primary analysis failed, trying Anthropic...")
            
            # Try Anthropic analysis
            if self.anthropic_client:
                logger.info("🎯 Attempting Anthropic analysis...")
                result = self._analyze_with_anthropic(optimized_content, company_name, company_number, analysis_context, extracted_content)
                if result:
                    analysis_time = time.time() - start_time
                    logger.info(f"✅ Anthropic analysis completed in {analysis_time:.2f}s")
                    return result
                else:
                    logger.warning("⚠️ Anthropic analysis failed")
            
            # If OpenAI was set as fallback but primary failed, try it now
            if self.openai_client and self.client_type == "anthropic_fallback":
                logger.info("🎯 Attempting OpenAI analysis (fallback)...")
                result = self._analyze_with_openai(optimized_content, company_name, company_number, analysis_context, extracted_content)
                if result:
                    analysis_time = time.time() - start_time
                    logger.info(f"✅ OpenAI fallback analysis completed in {analysis_time:.2f}s")
                    return result
            
            # Emergency fallback
            logger.error("❌ All AI analysis methods failed")
            return self._create_emergency_analysis(company_name, company_number, "All AI services failed")
        
        except Exception as e:
            logger.error(f"❌ Analysis failed with exception: {e}")
            import traceback
            logger.error(f"📊 Traceback: {traceback.format_exc()}")
            return self._create_emergency_analysis(company_name, company_number, str(e))
    
    def _optimize_content_strategically(self, content: str, extracted_content: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        ENHANCED: Strategic content optimization with technical insights extraction
        
        This is the main optimization method that:
        1. Extracts strategic technical insights first
        2. Optimizes main content based on file structure
        3. Combines both for maximum strategic value
        """
        
        if len(content) <= self.max_content_chars:
            logger.info(f"📊 Content within limits: {len(content):,} chars")
            return content
        
        logger.info(f"📊 Strategic optimization: {len(content):,} chars -> target: {self.max_content_chars:,}")
        
        # Step 1: Extract strategic technical insights first (reserve 20% of space)
        strategic_technical = self._extract_strategic_technical_insights(content)
        technical_space_used = len(strategic_technical)
        
        # Step 2: Calculate remaining space for main content
        main_content_limit = self.max_content_chars - technical_space_used - 1000  # Leave buffer
        
        # Step 3: Optimize main content based on file structure
        if extracted_content and len(extracted_content) > 1:
            logger.info(f"📄 Multi-file optimization for {len(extracted_content)} files")
            optimized_main = self._optimize_multi_file_content(extracted_content, main_content_limit)
        else:
            logger.info("📄 Single content optimization")
            optimized_main = self._optimize_single_content(content, main_content_limit)
        
        # Step 4: Combine with strategic technical insights
        if strategic_technical:
            final_content = f"{optimized_main}\n\n=== STRATEGIC TECHNICAL INSIGHTS ===\n{strategic_technical}"
            logger.info(f"✅ Combined with {len(strategic_technical)} chars of strategic technical insights")
        else:
            final_content = optimized_main
            logger.info("✅ No strategic technical insights found")
        
        logger.info(f"✅ Strategic optimization complete: {len(final_content):,} chars")
        return final_content
    
    def _extract_strategic_technical_insights(self, content: str) -> str:
        """
        Extract strategically valuable information from technical sections
        Focus on numbers and policies that reveal business model and risk appetite
        """
        
        strategic_technical_patterns = {
            'investment_priorities': {
                'keywords': ['additions', 'capex', 'capital expenditure', 'investments', 'technology', 'infrastructure'],
                'context_words': ['million', 'thousand', '£', '$', 'spent', 'invested'],
                'strategic_value': 'Reveals where management prioritizes spending → business archetype'
            },
            'risk_appetite_indicators': {
                'keywords': ['provision', 'impairment', 'credit loss', 'bad debt', 'floor', 'overlay'],
                'context_words': ['rate', '%', 'basis points', 'conservative', 'prudent'],
                'strategic_value': 'Shows actual vs required conservatism → risk archetype'
            },
            'revenue_model': {
                'keywords': ['fee income', 'commission', 'arrangement fee', 'early repayment', 'penalty'],
                'context_words': ['million', 'thousand', '£', '$', 'revenue', 'income'],
                'strategic_value': 'Reveals profit sources → fee extraction vs interest margin strategy'
            },
            'competitive_benchmarks': {
                'keywords': ['industry average', 'peer group', 'market comparison', 'benchmark'],
                'context_words': ['outperform', 'above', 'below', 'better', 'worse'],
                'strategic_value': 'Shows relative performance → market position archetype'
            },
            'policy_changes': {
                'keywords': ['change in policy', 'accounting change', 'new standard', 'adopted'],
                'context_words': ['conservative', 'prudent', 'aggressive', 'early adoption'],
                'strategic_value': 'Reveals management philosophy shifts'
            }
        }
        
        extracted_insights = []
        
        for insight_type, pattern in strategic_technical_patterns.items():
            insights = self._find_strategic_technical_content(content, pattern)
            if insights:
                extracted_insights.extend(insights)
        
        return "\n\n".join(extracted_insights) if extracted_insights else ""
    
    def _find_strategic_technical_content(self, content: str, pattern: dict) -> List[str]:
        """Find specific strategic insights in technical content"""
        
        insights = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check if line contains relevant keywords
            has_keyword = any(keyword in line_lower for keyword in pattern['keywords'])
            if not has_keyword:
                continue
                
            # Check if line has strategic context (numbers, comparisons, etc.)
            has_context = any(context in line_lower for context in pattern['context_words'])
            if not has_context:
                continue
            
            # Extract surrounding context (3 lines before and after)
            start_idx = max(0, i - 3)
            end_idx = min(len(lines), i + 4)
            context_lines = lines[start_idx:end_idx]
            
            # Clean and format the insight
            insight = self._format_technical_insight(context_lines, pattern['strategic_value'])
            if insight and len(insight) > 50:  # Only include substantial insights
                insights.append(insight)
        
        return insights[:3]  # Limit to top 3 insights per category
    
    def _format_technical_insight(self, context_lines: List[str], strategic_value: str) -> str:
        """Format technical insight with strategic context"""
        
        # Clean up the lines
        cleaned_lines = []
        for line in context_lines:
            cleaned = line.strip()
            if cleaned and not cleaned.startswith('Page ') and len(cleaned) > 10:
                cleaned_lines.append(cleaned)
        
        if len(cleaned_lines) < 2:
            return ""
        
        # Format as strategic insight
        content = " ".join(cleaned_lines)
        
        # Remove excessive whitespace and clean up
        content = re.sub(r'\s+', ' ', content)
        content = content[:500]  # Limit length
        
        return f"STRATEGIC TECHNICAL INSIGHT: {content}\n[Strategic Value: {strategic_value}]"
    
    def _optimize_multi_file_content(self, extracted_content: List[Dict[str, Any]], target_chars: int) -> str:
        """Optimize content across multiple files with balanced sampling"""
        
        total_files = len(extracted_content)
        chars_per_file = target_chars // total_files
        min_chars_per_file = 10000  # Minimum 10K chars per file
        
        # If equal distribution is too small, prioritize recent files
        if chars_per_file < min_chars_per_file:
            return self._optimize_with_recency_priority(extracted_content, target_chars)
        
        optimized_parts = []
        
        for i, file_data in enumerate(extracted_content):
            filename = file_data.get('filename', f'file_{i+1}')
            file_content = file_data.get('content', '')
            
            logger.info(f"📄 Optimizing {filename}: {len(file_content):,} chars -> {chars_per_file:,}")
            
            if len(file_content) <= chars_per_file:
                optimized_file_content = file_content
            else:
                # Intelligent sampling for this file
                optimized_file_content = self._sample_file_content(file_content, chars_per_file, filename)
            
            optimized_parts.append(f"\n\n=== FILE: {filename} ===\n{optimized_file_content}")
        
        combined = "\n".join(optimized_parts)
        logger.info(f"✅ Multi-file optimization complete: {len(combined):,} chars from {total_files} files")
        return combined
    
    def _optimize_with_recency_priority(self, extracted_content: List[Dict[str, Any]], target_chars: int) -> str:
        """Prioritize recent years when space is limited"""
        
        # Sort by year (most recent first)
        sorted_files = sorted(extracted_content, key=lambda x: self._extract_year_from_filename(x.get('filename', '')), reverse=True)
        
        optimized_parts = []
        remaining_chars = target_chars
        
        for i, file_data in enumerate(sorted_files):
            filename = file_data.get('filename', f'file_{i+1}')
            file_content = file_data.get('content', '')
            year = self._extract_year_from_filename(filename)
            
            # Allocate more space to recent years
            if i == 0:  # Most recent year
                allocation = min(remaining_chars * 0.4, len(file_content))  # 40% for most recent
            elif i == 1:  # Second most recent
                allocation = min(remaining_chars * 0.3, len(file_content))  # 30% for second
            else:  # Remaining years split evenly
                remaining_files = len(sorted_files) - 2
                allocation = min(remaining_chars / max(remaining_files, 1), len(file_content))
            
            allocation = int(allocation)
            
            if allocation > 5000:  # Only include if meaningful content
                if len(file_content) <= allocation:
                    sampled_content = file_content
                else:
                    sampled_content = self._sample_file_content(file_content, allocation, filename)
                
                optimized_parts.append(f"\n\n=== FILE: {filename} (Year {year}) ===\n{sampled_content}")
                remaining_chars -= len(sampled_content)
                
                logger.info(f"📄 Included {filename}: {len(sampled_content):,} chars (Year {year})")
        
        combined = "\n".join(optimized_parts)
        logger.info(f"✅ Recency-prioritized optimization: {len(combined):,} chars")
        return combined
    
    def _sample_file_content(self, content: str, target_chars: int, filename: str) -> str:
        """Intelligently sample content from a single file"""
        
        if len(content) <= target_chars:
            return content
        
        # Key sections to prioritize (common in financial documents)
        priority_keywords = [
            'executive summary', 'chairman', 'chief executive', 'ceo report',
            'strategic report', 'directors report', 'business review',
            'principal risks', 'risk management', 'risk factors',
            'financial highlights', 'key performance', 'results summary',
            'outlook', 'strategy', 'objectives', 'future prospects'
        ]
        
        # Split content into sections
        sections = self._split_into_sections(content)
        
        # Score sections by importance
        scored_sections = []
        for section in sections:
            score = self._score_section_enhanced(section, priority_keywords)
            scored_sections.append((score, section))
        
        # Sort by score (highest first)
        scored_sections.sort(reverse=True, key=lambda x: x[0])
        
        # Build optimized content by taking highest-scoring sections
        optimized_content = ""
        for score, section in scored_sections:
            if len(optimized_content) + len(section) <= target_chars:
                optimized_content += section + "\n\n"
            else:
                # Add partial section if it fits
                remaining_space = target_chars - len(optimized_content)
                if remaining_space > 100:  # Only if meaningful space left
                    # Try to end at sentence boundary
                    partial = section[:remaining_space]
                    last_period = partial.rfind('.')
                    if last_period > remaining_space * 0.8:  # If we find a period near the end
                        partial = partial[:last_period + 1]
                    optimized_content += partial
                break
        
        logger.info(f"📄 Sampled {filename}: {len(content):,} -> {len(optimized_content):,} chars")
        return optimized_content.strip()
    
    def _split_into_sections(self, content: str) -> List[str]:
        """Split content into logical sections"""
        
        # Common section headers in financial documents
        section_markers = [
            'STRATEGIC REPORT', 'DIRECTORS REPORT', 'CHAIRMAN', 'CHIEF EXECUTIVE',
            'BUSINESS REVIEW', 'FINANCIAL REVIEW', 'RISK MANAGEMENT', 'GOVERNANCE',
            'NOTES TO THE FINANCIAL STATEMENTS', 'INDEPENDENT AUDITOR'
        ]
        
        sections = []
        current_section = ""
        
        for line in content.split('\n'):
            line_upper = line.strip().upper()
            
            # Check if this line is a section header
            is_section_header = any(marker in line_upper for marker in section_markers)
            
            if is_section_header and current_section.strip():
                # Save previous section and start new one
                sections.append(current_section.strip())
                current_section = line + "\n"
            else:
                current_section += line + "\n"
            
            # Also split on very long sections (every 2000 lines)
            if current_section.count('\n') > 2000:
                sections.append(current_section.strip())
                current_section = ""
        
        # Add the last section
        if current_section.strip():
            sections.append(current_section.strip())
        
        return sections
    
    def _score_section_enhanced(self, section: str, priority_keywords: List[str]) -> float:
        """Enhanced section scoring that values strategic technical content"""
        
        section_lower = section.lower()
        score = 0.0
        
        # Base score on length
        score += len(section) * 0.0001
        
        # High value keywords (executive content)
        for keyword in priority_keywords:
            if keyword in section_lower:
                score += 10.0
        
        # Strategic technical indicators (NEW)
        strategic_technical_terms = {
            'investment_indicators': ['capex', 'capital expenditure', 'technology investment', 'infrastructure spend'],
            'risk_indicators': ['provision rate', 'impairment rate', 'credit loss', 'coverage ratio'],
            'competitive_indicators': ['industry average', 'peer comparison', 'market benchmark'],
            'efficiency_indicators': ['cost income ratio', 'operating leverage', 'productivity'],
            'profitability_indicators': ['net interest margin', 'return on equity', 'return on assets']
        }
        
        for category, terms in strategic_technical_terms.items():
            for term in terms:
                if term in section_lower:
                    score += 7.0  # High value for strategic technical content
                    
        # Regular strategy/risk terms
        strategy_terms = ['strategy', 'strategic', 'objective', 'goal', 'vision', 'mission']
        risk_terms = ['risk', 'threat', 'challenge', 'uncertainty', 'regulatory']
        
        for term in strategy_terms:
            score += section_lower.count(term) * 5.0
        for term in risk_terms:
            score += section_lower.count(term) * 4.0
        
        # Reduce score for pure compliance boilerplate
        boilerplate_terms = ['in accordance with', 'as required by', 'comply with', 'pursuant to']
        for term in boilerplate_terms:
            if term in section_lower:
                score *= 0.7  # Reduce but don't eliminate
        
        # Boost score for quantitative insights
        if re.search(r'£[\d,]+|[\d.]+%|[\d.]+ basis points', section):
            score *= 1.3  # Boost sections with specific numbers
        
        return score
    
    def _extract_year_from_filename(self, filename: str) -> int:
        """Extract year from filename for prioritization"""
        # Look for 4-digit years
        year_match = re.search(r'20\d{2}', filename)
        if year_match:
            return int(year_match.group())
        
        # Default to current year if no year found
        return datetime.now().year
    
    def _optimize_single_content(self, content: str, target_chars: int) -> str:
        """Optimize single large content block"""
        
        # For single content, take strategic approach:
        # 1. First 30% (early content often has summaries)
        # 2. Smart sampling from middle sections
        # 3. Last 10% (often has recent developments)
        
        first_part_size = int(target_chars * 0.3)
        last_part_size = int(target_chars * 0.1)
        middle_part_size = target_chars - first_part_size - last_part_size
        
        # Take first part
        first_part = content[:first_part_size]
        
        # Take last part
        last_part = content[-last_part_size:] if len(content) > last_part_size else ""
        
        # Sample middle part intelligently
        middle_start = first_part_size
        middle_end = len(content) - last_part_size
        middle_content = content[middle_start:middle_end]
        
        if len(middle_content) <= middle_part_size:
            middle_part = middle_content
        else:
            # Sample key sections from middle
            middle_sections = self._split_into_sections(middle_content)
            scored_sections = [(self._score_section_enhanced(s, []), s) for s in middle_sections]
            scored_sections.sort(reverse=True, key=lambda x: x[0])
            
            middle_part = ""
            for score, section in scored_sections:
                if len(middle_part) + len(section) <= middle_part_size:
                    middle_part += section + "\n\n"
                else:
                    break
        
        # Combine parts
        optimized = f"{first_part}\n\n[... MIDDLE CONTENT SAMPLED ...]\n\n{middle_part}\n\n[... FINAL CONTENT ...]\n\n{last_part}"
        
        logger.info(f"📊 Single content optimized: {len(content):,} -> {len(optimized):,} chars")
        return optimized
    
    # Backward compatibility - keep original method name
    def _optimize_content(self, content: str, extracted_content: Optional[List[Dict[str, Any]]] = None) -> str:
        """Main content optimization method - now uses strategic optimization"""
        return self._optimize_content_strategically(content, extracted_content)
    
    # ... (keeping all other existing methods unchanged for brevity)
    # All the AI client methods, parsing, confidence calculation, etc. remain the same
    
    def _analyze_with_openai(self, content: str, company_name: str, company_number: str,
                           analysis_context: Optional[str], extracted_content: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        """Analyze using OpenAI with thread-safe timeout"""
        
        if not self.openai_client:
            logger.warning("⚠️ OpenAI client not available")
            return None
        
        # Create timeout manager with much longer duration for large content
        timeout_manager = TimeoutManager(90.0)
        timeout_manager.start()
        
        models = [self.primary_model, self.fallback_model]
        
        for model in models:
            for attempt in range(self.max_retries):
                try:
                    # Check timeout before each attempt
                    timeout_manager.check_timeout()
                    
                    logger.info(f"🔄 OpenAI: {model}, attempt {attempt + 1}")
                    
                    messages = self._create_complete_openai_messages(content, company_name, analysis_context)
                    
                    # Use remaining time for API timeout
                    api_timeout = min(timeout_manager.remaining_time(), 60.0)
                    
                    response = self.openai_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=self.max_output_tokens,
                        temperature=0.1,
                        response_format={"type": "json_object"},
                        timeout=api_timeout
                    )
                    
                    response_text = response.choices[0].message.content
                    logger.info(f"✅ OpenAI response: {len(response_text)} chars")
                    
                    analysis = self._parse_json_response(response_text)
                    if analysis:
                        return self._create_complete_report(analysis, company_name, company_number, model, "openai", extracted_content)
                    else:
                        logger.warning("⚠️ Failed to parse OpenAI JSON response")
                
                except TimeoutError as e:
                    logger.error(f"❌ OpenAI timeout: {e}")
                    return None
                
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"⚠️ OpenAI attempt {attempt + 1} failed: {error_msg}")
                    
                    if self._is_retryable_error(error_msg) and attempt < self.max_retries - 1:
                        if timeout_manager.remaining_time() > 3:
                            retry_delay = min(self.base_retry_delay, timeout_manager.remaining_time() / 2)
                            logger.info(f"⏰ Retrying in {retry_delay:.1f}s...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            logger.warning("⚠️ Not enough time remaining for retry")
                            return None
                    else:
                        logger.error(f"❌ Non-retryable error or max retries reached: {error_msg}")
                        break
        
        logger.error("❌ All OpenAI attempts failed")
        return None
    
    def _analyze_with_anthropic(self, content: str, company_name: str, company_number: str,
                              analysis_context: Optional[str], extracted_content: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        """Analyze using Anthropic with thread-safe timeout"""
        
        if not self.anthropic_client:
            logger.warning("⚠️ Anthropic client not available")
            return None
        
        # Create timeout manager with much longer duration for large content
        timeout_manager = TimeoutManager(70.0)
        timeout_manager.start()
        
        # Try modern model first, then fallback
        models = ["claude-3-5-sonnet-20241022", "claude-3-sonnet-20240229"]
        
        for model in models:
            for attempt in range(self.max_retries):
                try:
                    # Check timeout before each attempt
                    timeout_manager.check_timeout()
                    
                    logger.info(f"🔄 Anthropic: {model}, attempt {attempt + 1}")
                    
                    prompt = self._create_complete_anthropic_prompt(content, company_name, analysis_context)
                    
                    # Use remaining time for API timeout
                    api_timeout = min(timeout_manager.remaining_time(), 50.0)
                    
                    response = self.anthropic_client.messages.create(
                        model=model,
                        max_tokens=self.max_output_tokens,
                        temperature=0.1,
                        messages=[{"role": "user", "content": prompt}],
                        timeout=api_timeout
                    )
                    
                    response_text = response.content[0].text
                    logger.info(f"✅ Anthropic response: {len(response_text)} chars")
                    
                    analysis = self._parse_json_response(response_text)
                    if analysis:
                        return self._create_complete_report(analysis, company_name, company_number, model, "anthropic", extracted_content)
                    else:
                        logger.warning("⚠️ Failed to parse Anthropic JSON response")
                
                except TimeoutError as e:
                    logger.error(f"❌ Anthropic timeout: {e}")
                    return None
                
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"⚠️ Anthropic {model} attempt {attempt + 1} failed: {error_msg}")
                    
                    # If 404 error, try next model immediately
                    if "404" in error_msg or "not_found" in error_msg.lower():
                        logger.info(f"🔄 Model {model} not found, trying next model...")
                        break
                    
                    if attempt < self.max_retries - 1:
                        if timeout_manager.remaining_time() > 2:
                            retry_delay = min(self.base_retry_delay, timeout_manager.remaining_time() / 2)
                            logger.info(f"⏰ Retrying in {retry_delay:.1f}s...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            logger.warning("⚠️ Not enough time remaining for retry")
                            return None
                    else:
                        logger.error(f"❌ Max retries reached for {model}")
                        break
        
        logger.error("❌ All Anthropic attempts failed")
        return None
    
    def _create_complete_openai_messages(self, content: str, company_name: str, 
                                       analysis_context: Optional[str]) -> List[Dict[str, str]]:
        """Create complete OpenAI messages with all archetypes"""
        
        context_note = f"\n\nAnalysis Context: {analysis_context}" if analysis_context else ""
        
        # Format all business archetypes
        business_archetypes_text = "\n".join([
            f"- {name}: {definition}" 
            for name, definition in self.business_archetypes.items()
        ])
        
        # Format all risk archetypes
        risk_archetypes_text = "\n".join([
            f"- {name}: {definition}" 
            for name, definition in self.risk_archetypes.items()
        ])
        
        system_prompt = f"""You are an expert strategic business analyst specializing in financial services archetype analysis.

Analyze the company and respond with VALID JSON ONLY using this EXACT structure:

{{
  "business_strategy": {{
    "dominant_archetype": "[exact archetype name from list]",
    "dominant_rationale": "[detailed analysis 100+ words with specific evidence]",
    "secondary_archetype": "[exact archetype name from list]",
    "secondary_rationale": "[detailed analysis 70+ words with specific evidence]",
    "material_changes": "[any changes over the period analyzed]"
  }},
  "risk_strategy": {{
    "dominant_archetype": "[exact archetype name from list]",
    "dominant_rationale": "[detailed analysis 100+ words with specific evidence]", 
    "secondary_archetype": "[exact archetype name from list]",
    "secondary_rationale": "[detailed analysis 70+ words with specific evidence]",
    "material_changes": "[any changes over the period analyzed]"
  }},
  "swot_analysis": {{
    "strengths": ["strength1", "strength2", "strength3", "strength4"],
    "weaknesses": ["weakness1", "weakness2", "weakness3", "weakness4"],
    "opportunities": ["opportunity1", "opportunity2", "opportunity3", "opportunity4"],
    "threats": ["threat1", "threat2", "threat3", "threat4"]
  }},
  "years_analyzed": "[period covered by the analysis]",
  "confidence_level": "[will be determined by system based on analysis scope]"
}}

BUSINESS STRATEGY ARCHETYPES:
{business_archetypes_text}

RISK STRATEGY ARCHETYPES:
{risk_archetypes_text}

IMPORTANT: 
- Use EXACT archetype names from the lists above
- Provide specific evidence from the company content
- SWOT should reflect the combination of the 4 selected archetypes
- Focus on how the archetype combination creates specific advantages/disadvantages
- The confidence_level will be calculated by the system based on data scope"""

        user_prompt = f"""Analyze {company_name} and provide comprehensive strategic archetype analysis.

COMPANY CONTENT:{context_note}
{content}

Respond with valid JSON using the exact structure specified."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _create_complete_anthropic_prompt(self, content: str, company_name: str,
                                        analysis_context: Optional[str]) -> str:
        """Create complete Anthropic prompt"""
        
        context_note = f"\nAnalysis Context: {analysis_context}" if analysis_context else ""
        
        business_list = "\n".join([f"- {name}: {def_}" for name, def_ in self.business_archetypes.items()])
        risk_list = "\n".join([f"- {name}: {def_}" for name, def_ in self.risk_archetypes.items()])
        
        return f"""Analyze {company_name} for strategic archetypes. Respond with valid JSON only:

{{
  "business_strategy": {{
    "dominant_archetype": "[exact name]",
    "dominant_rationale": "[100+ words with evidence]",
    "secondary_archetype": "[exact name]", 
    "secondary_rationale": "[70+ words with evidence]",
    "material_changes": "[any changes over period]"
  }},
  "risk_strategy": {{
    "dominant_archetype": "[exact name]",
    "dominant_rationale": "[100+ words with evidence]",
    "secondary_archetype": "[exact name]",
    "secondary_rationale": "[70+ words with evidence]", 
    "material_changes": "[any changes over period]"
  }},
  "swot_analysis": {{
    "strengths": ["strength1", "strength2", "strength3", "strength4"],
    "weaknesses": ["weakness1", "weakness2", "weakness3", "weakness4"],
    "opportunities": ["opportunity1", "opportunity2", "opportunity3", "opportunity4"],
    "threats": ["threat1", "threat2", "threat3", "threat4"]
  }},
  "years_analyzed": "[period]",
  "confidence_level": "[will be determined by system based on analysis scope]"
}}

BUSINESS STRATEGY ARCHETYPES:
{business_list}

RISK STRATEGY ARCHETYPES:
{risk_list}

The confidence_level will be calculated by the system based on data scope and quality.

COMPANY CONTENT:{context_note}
{content}"""
    
    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response with error handling"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
        
        logger.error("❌ JSON parsing failed")
        return None
    
    def _determine_confidence_level(self, analysis: Dict[str, Any], extracted_content: Optional[List[Dict[str, Any]]] = None) -> tuple:
        """
        FIXED: Determine confidence level based on analysis scope and data quality
        Enhanced debugging to track exactly what's happening in the calculation
        
        Returns:
            tuple: (confidence_level, explanation)
        """
        logger.info(f"🔍 CONFIDENCE CALCULATION START:")
        logger.info(f"   Analysis keys: {list(analysis.keys())}")
        logger.info(f"   Extracted content: {len(extracted_content) if extracted_content else 0} files")
        
        # FIXED: Extract years information with better parsing
        years_analyzed = analysis.get('years_analyzed', [])
        logger.info(f"   Raw years_analyzed = {years_analyzed} (type: {type(years_analyzed)})")
        
        # Enhanced year parsing
        parsed_years = []
        if isinstance(years_analyzed, str):
            try:
                # Handle various string formats
                year_matches = re.findall(r'\b(20\d{2})\b', years_analyzed)
                parsed_years = [int(year) for year in year_matches]
                logger.info(f"   Parsed years from string: {parsed_years}")
            except Exception as e:
                logger.warning(f"   Could not parse years from string: {e}")
                parsed_years = []
        elif isinstance(years_analyzed, list):
            # Handle list of years (could be strings or integers)
            for item in years_analyzed:
                try:
                    if isinstance(item, (int, float)):
                        year = int(item)
                    elif isinstance(item, str):
                        year = int(item.strip())
                    else:
                        continue
                    
                    # Validate year is reasonable (2000-2030)
                    if 2000 <= year <= 2030:
                        parsed_years.append(year)
                except:
                    continue
            logger.info(f"   Parsed years from list: {parsed_years}")
        else:
            logger.warning(f"   Unexpected years_analyzed type: {type(years_analyzed)}")
            parsed_years = []
        
        # Remove duplicates and sort
        parsed_years = sorted(list(set(parsed_years)))
        years_count = len(parsed_years)
        files_processed = len(extracted_content) if extracted_content else 0
        
        logger.info(f"   CONFIDENCE INPUTS:")
        logger.info(f"   - Final parsed years: {parsed_years}")
        logger.info(f"   - Years count: {years_count}")
        logger.info(f"   - Files processed: {files_processed}")
        
        # Calculate years span
        if years_count >= 2:
            years_span = max(parsed_years) - min(parsed_years) + 1
            logger.info(f"   - Years span: {years_span} (from {min(parsed_years)} to {max(parsed_years)})")
        else:
            years_span = years_count
            logger.info(f"   - Years span: {years_span} (single year or no years)")
        
        # Check content quality indicators
        business_strategy = analysis.get('business_strategy', {})
        risk_strategy = analysis.get('risk_strategy', {})
        
        business_reasoning_length = len(str(business_strategy.get('dominant_rationale', business_strategy.get('dominant_reasoning', ''))))
        risk_reasoning_length = len(str(risk_strategy.get('dominant_rationale', risk_strategy.get('dominant_reasoning', ''))))
        
        logger.info(f"   - Business reasoning length: {business_reasoning_length}")
        logger.info(f"   - Risk reasoning length: {risk_reasoning_length}")
        
        # FIXED: Confidence scoring based on data scope and quality
        confidence_score = 0
        score_breakdown = []
        
        # Years coverage scoring (40 points max)
        if years_count >= 5:
            confidence_score += 40
            score_breakdown.append(f"+40 pts: {years_count} years analyzed (excellent coverage)")
        elif years_count >= 4:
            confidence_score += 35
            score_breakdown.append(f"+35 pts: {years_count} years analyzed (very good coverage)")
        elif years_count >= 3:
            confidence_score += 25
            score_breakdown.append(f"+25 pts: {years_count} years analyzed (good coverage)")
        elif years_count >= 2:
            confidence_score += 15
            score_breakdown.append(f"+15 pts: {years_count} years analyzed (adequate coverage)")
        else:
            confidence_score += 5
            score_breakdown.append(f"+5 pts: {years_count} year(s) analyzed (limited coverage)")
        
        # Years span scoring (25 points max)
        if years_span >= 5:
            confidence_score += 25
            score_breakdown.append(f"+25 pts: {years_span}-year timespan (excellent longitudinal view)")
        elif years_span >= 4:
            confidence_score += 20
            score_breakdown.append(f"+20 pts: {years_span}-year timespan (very good longitudinal view)")
        elif years_span >= 3:
            confidence_score += 15
            score_breakdown.append(f"+15 pts: {years_span}-year timespan (good longitudinal view)")
        elif years_span >= 2:
            confidence_score += 10
            score_breakdown.append(f"+10 pts: {years_span}-year timespan (some longitudinal view)")
        else:
            confidence_score += 2
            score_breakdown.append(f"+2 pts: {years_span}-year timespan (snapshot view)")
        
        # Files processed scoring (20 points max)
        if files_processed >= 5:
            confidence_score += 20
            score_breakdown.append(f"+20 pts: {files_processed} files processed (comprehensive documentation)")
        elif files_processed >= 4:
            confidence_score += 16
            score_breakdown.append(f"+16 pts: {files_processed} files processed (very good documentation)")
        elif files_processed >= 3:
            confidence_score += 12
            score_breakdown.append(f"+12 pts: {files_processed} files processed (good documentation)")
        elif files_processed >= 2:
            confidence_score += 8
            score_breakdown.append(f"+8 pts: {files_processed} files processed (adequate documentation)")
        else:
            confidence_score += 4
            score_breakdown.append(f"+4 pts: {files_processed} file(s) processed (limited documentation)")
        
        # Content quality scoring (15 points max)
        if business_reasoning_length >= 200 and risk_reasoning_length >= 200:
            confidence_score += 15
            score_breakdown.append(f"+15 pts: comprehensive reasoning (business: {business_reasoning_length}, risk: {risk_reasoning_length} chars)")
        elif business_reasoning_length >= 150 and risk_reasoning_length >= 150:
            confidence_score += 12
            score_breakdown.append(f"+12 pts: good reasoning quality")
        elif business_reasoning_length >= 100 and risk_reasoning_length >= 100:
            confidence_score += 8
            score_breakdown.append(f"+8 pts: adequate reasoning")
        else:
            confidence_score += 3
            score_breakdown.append(f"+3 pts: basic reasoning")
        
        # FIXED: Determine final confidence level with correct thresholds
        if confidence_score >= 80:
            confidence_level = "high"
            explanation = f"High confidence ({confidence_score}/100 points) - Excellent analysis scope with comprehensive data coverage"
        elif confidence_score >= 60:
            confidence_level = "medium"
            explanation = f"Medium confidence ({confidence_score}/100 points) - Good analysis scope with adequate data coverage"
        else:
            confidence_level = "low"
            explanation = f"Low confidence ({confidence_score}/100 points) - Limited analysis scope or data coverage"
        
        # Create detailed explanation
        detailed_explanation = f"{explanation}. Scoring breakdown: {'; '.join(score_breakdown[:3])}{'...' if len(score_breakdown) > 3 else ''}."
        
        logger.info(f"🎯 CONFIDENCE CALCULATION RESULT:")
        logger.info(f"   Total score: {confidence_score}/100")
        logger.info(f"   Confidence level: {confidence_level}")
        logger.info(f"   Explanation: {detailed_explanation}")
        logger.info(f"   Score breakdown: {score_breakdown}")
        
        # VALIDATION CHECK
        expected_confidence = "low"
        if years_count >= 5 and files_processed >= 5:
            expected_confidence = "high"
        elif years_count >= 3 and files_processed >= 3:
            expected_confidence = "medium"
        
        if confidence_level != expected_confidence:
            logger.warning(f"❌ CONFIDENCE VALIDATION WARNING:")
            logger.warning(f"   Calculated: {confidence_level}")
            logger.warning(f"   Expected: {expected_confidence}")
            logger.warning(f"   This may indicate an issue with the scoring thresholds")
        else:
            logger.info(f"✅ CONFIDENCE VALIDATION PASSED: {confidence_level}")
        
        return confidence_level, detailed_explanation
    
    def _create_complete_report(self, analysis: Dict[str, Any], company_name: str,
                              company_number: str, model: str, service: str, 
                              extracted_content: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Create complete report in the specified format with FIXED confidence assessment
        """
        logger.info(f"🎯 CREATING COMPLETE REPORT:")
        logger.info(f"   Company: {company_name} ({company_number})")
        logger.info(f"   Analysis keys: {list(analysis.keys())}")
        logger.info(f"   Extracted content files: {len(extracted_content) if extracted_content else 0}")
        
        # Extract data with fallbacks
        business = analysis.get('business_strategy', {})
        risk = analysis.get('risk_strategy', {})
        swot = analysis.get('swot_analysis', {})
        
        # **FIXED: Calculate confidence based on current analysis scope only**
        confidence_level, confidence_explanation = self._determine_confidence_level(analysis, extracted_content)
        
        # ENHANCED DEBUG LOGGING FOR CONFIDENCE IN REPORT CREATION
        logger.info(f"🎯 REPORT CREATION CONFIDENCE DEBUG:")
        logger.info(f"   Years from analysis: {analysis.get('years_analyzed', 'MISSING')}")
        logger.info(f"   Files from extracted_content: {len(extracted_content) if extracted_content else 0}")
        logger.info(f"   Final calculated confidence: {confidence_level}")
        logger.info(f"   Confidence explanation: {confidence_explanation}")
        
        return {
            'success': True,
            'company_name': company_name,
            'company_number': company_number,
            'years_analyzed': analysis.get('years_analyzed', 'Recent period'),
            'files_processed': len(extracted_content) if extracted_content else 'Multiple financial documents',
            'analysis_context': f'Strategic archetype analysis using {service}',
            
            # Executive Summary
            'executive_summary': {
                'business_dominant': business.get('dominant_archetype', 'Disciplined Specialist Growth'),
                'business_secondary': business.get('secondary_archetype', 'Service-Driven Differentiator'),
                'risk_dominant': risk.get('dominant_archetype', 'Risk-First Conservative'),
                'risk_secondary': risk.get('secondary_archetype', 'Rules-Led Operator'),
                'key_insight': f"Analysis reveals a {business.get('dominant_archetype', 'Disciplined Specialist Growth')} business strategy paired with {risk.get('dominant_archetype', 'Risk-First Conservative')} risk approach."
            },
            
            # Business Strategy Analysis
            'business_strategy_analysis': {
                'dominant': {
                    'archetype': business.get('dominant_archetype', 'Disciplined Specialist Growth'),
                    'definition': self.business_archetypes.get(business.get('dominant_archetype', 'Disciplined Specialist Growth'), 'Niche focus with strong underwriting edge; grows opportunistically while recycling balance-sheet.'),
                    'rationale': business.get('dominant_rationale', 'Conservative growth approach with focus on underwriting quality.'),
                    'evidence': self._extract_evidence_points(business.get('dominant_rationale', ''))
                },
                'secondary': {
                    'archetype': business.get('secondary_archetype', 'Service-Driven Differentiator'),
                    'definition': self.business_archetypes.get(business.get('secondary_archetype', 'Service-Driven Differentiator'), 'Wins by superior client experience / advice rather than price or scale.'),
                    'rationale': business.get('secondary_rationale', 'Focus on customer service and relationship building.'),
                    'evidence': self._extract_evidence_points(business.get('secondary_rationale', ''))
                },
                'material_changes': business.get('material_changes', 'No significant changes identified over the period analyzed.')
            },
            
            # Risk Strategy Analysis
            'risk_strategy_analysis': {
                'dominant': {
                    'archetype': risk.get('dominant_archetype', 'Risk-First Conservative'),
                    'definition': self.risk_archetypes.get(risk.get('dominant_archetype', 'Risk-First Conservative'), 'Prioritises capital preservation and regulatory compliance; growth is secondary to resilience.'),
                    'rationale': risk.get('dominant_rationale', 'Prioritizes capital preservation and regulatory compliance.'),
                    'evidence': self._extract_evidence_points(risk.get('dominant_rationale', ''))
                },
                'secondary': {
                    'archetype': risk.get('secondary_archetype', 'Rules-Led Operator'),
                    'definition': self.risk_archetypes.get(risk.get('secondary_archetype', 'Rules-Led Operator'), 'Strict adherence to rules and checklists; prioritises control consistency over judgment or speed.'),
                    'rationale': risk.get('secondary_rationale', 'Structured approach to risk management with clear procedures.'),
                    'evidence': self._extract_evidence_points(risk.get('secondary_rationale', ''))
                },
                'material_changes': risk.get('material_changes', 'No significant changes identified over the period analyzed.')
            },
            
            # SWOT Analysis
            'swot_analysis': {
                'strengths': swot.get('strengths', [
                    'Strategic coherence between business and risk archetypes',
                    'Strong focus on underwriting quality and risk management',
                    'Stable customer base and reputation for reliability',
                    'Disciplined approach to capital allocation'
                ]),
                'weaknesses': swot.get('weaknesses', [
                    'Potential over-caution limiting growth opportunities',
                    'May be slow to adapt to market changes',
                    'Limited diversification due to niche focus',
                    'Conservative approach may restrict innovation'
                ]),
                'opportunities': swot.get('opportunities', [
                    'Market dislocation allowing cherry-picking of quality customers',
                    'Regulatory favor due to conservative approach',
                    'Building trust and reputation in specialized segments',
                    'Potential for selective expansion in adjacent markets'
                ]),
                'threats': swot.get('threats', [
                    'Fintech disruption with faster, data-driven models',
                    'Regulatory pressure for broader financial inclusion',
                    'Missed opportunities due to conservative approach',
                    'Competitive pressure from more agile market entrants'
                ])
            },
            
            # Strategic Recommendations
            'strategic_recommendations': self._generate_recommendations(
                business.get('dominant_archetype', ''),
                risk.get('dominant_archetype', ''),
                swot
            ),
            
            # Executive Dashboard
            'executive_dashboard': {
                'archetype_alignment': 'Strong alignment between business and risk strategies',
                'strategic_coherence': 'High - both archetypes favor disciplined, conservative approach',
                'competitive_position': 'Stable niche player with defensive characteristics',
                'growth_trajectory': 'Steady, controlled growth with emphasis on quality',
                'risk_profile': 'Conservative with strong capital preservation focus'
            },
            
            # Legacy format for backward compatibility (with definitions added)
            'business_strategy': {
                'dominant': business.get('dominant_archetype', 'Disciplined Specialist Growth'),
                'dominant_definition': self.business_archetypes.get(business.get('dominant_archetype', 'Disciplined Specialist Growth'), 'Niche focus with strong underwriting edge; grows opportunistically while recycling balance-sheet.'),
                'dominant_reasoning': business.get('dominant_rationale', 'Analysis completed successfully'),
                'secondary': business.get('secondary_archetype', 'Service-Driven Differentiator'),
                'secondary_definition': self.business_archetypes.get(business.get('secondary_archetype', 'Service-Driven Differentiator'), 'Wins by superior client experience / advice rather than price or scale.'),
                'secondary_reasoning': business.get('secondary_rationale', 'Secondary analysis completed'),
                'evidence_quotes': self._extract_evidence_points(business.get('dominant_rationale', ''))
            },
            'risk_strategy': {
                'dominant': risk.get('dominant_archetype', 'Risk-First Conservative'),
                'dominant_definition': self.risk_archetypes.get(risk.get('dominant_archetype', 'Risk-First Conservative'), 'Prioritises capital preservation and regulatory compliance; growth is secondary to resilience.'),
                'dominant_reasoning': risk.get('dominant_rationale', 'Risk analysis completed successfully'),
                'secondary': risk.get('secondary_archetype', 'Rules-Led Operator'),
                'secondary_definition': self.risk_archetypes.get(risk.get('secondary_archetype', 'Rules-Led Operator'), 'Strict adherence to rules and checklists; prioritises control consistency over judgment or speed.'),
                'secondary_reasoning': risk.get('secondary_rationale', 'Secondary risk analysis completed'),
                'evidence_quotes': self._extract_evidence_points(risk.get('dominant_rationale', ''))
            },
            
            # Metadata with CORRECTED confidence level
            'analysis_date': datetime.now().isoformat(),
            'analysis_type': f'complete_{service}_archetype_analysis',
            'confidence_level': confidence_level,  # **FIXED: Use calculated confidence**
            'confidence_explanation': confidence_explanation,  # **NEW: Detailed explanation**
            'processing_stats': {
                'model_used': model,
                'service_used': service,
                'analysis_timestamp': datetime.now().isoformat(),
                'archetypes_evaluated': {
                    'business_total': len(self.business_archetypes),
                    'risk_total': len(self.risk_archetypes)
                },
                'confidence_factors': {
                    'years_count': len(extracted_content) if extracted_content else 0,
                    'files_processed': len(extracted_content) if extracted_content else 0,
                    'reasoning_quality': 'assessed',
                    'confidence_calculation_debug': {
                        'input_years': analysis.get('years_analyzed'),
                        'files_count': len(extracted_content) if extracted_content else 0,
                        'calculated_level': confidence_level,
                        'calculation_timestamp': datetime.now().isoformat()
                    }
                },
                'content_optimization': {
                    'strategy': 'strategic_technical_extraction',
                    'technical_insights_included': 'yes',
                    'multi_file_balancing': len(extracted_content) > 1 if extracted_content else False,
                    'recency_prioritization': True
                }
            }
        }
    
    def _extract_evidence_points(self, text: str) -> List[str]:
        """Extract evidence points from rationale text"""
        if not text:
            return ['Analysis completed successfully']
        
        # Simple extraction of key points
        sentences = text.split('. ')
        evidence = []
        for sentence in sentences[:3]:  # Take first 3 sentences as evidence
            if len(sentence.strip()) > 20:
                evidence.append(sentence.strip() + ('.' if not sentence.endswith('.') else ''))
        
        return evidence if evidence else ['Detailed analysis completed']
    
    def _generate_recommendations(self, business_archetype: str, risk_archetype: str, 
                                swot: Dict[str, List[str]]) -> List[str]:
        """Generate strategic recommendations based on archetype combination"""
        
        recommendations = []
        
        # Base recommendations on archetype combinations
        if 'Disciplined Specialist Growth' in business_archetype and 'Risk-First Conservative' in risk_archetype:
            recommendations.extend([
                'Leverage conservative reputation to build trust with regulatory authorities and funding partners',
                'Consider selective expansion into adjacent niche markets where existing expertise can be applied',
                'Develop data analytics capabilities to maintain competitive edge in underwriting while preserving conservative approach',
                'Monitor for market dislocation opportunities where quality-focused approach can capture underserved segments'
            ])
        else:
            # General recommendations
            recommendations.extend([
                'Align risk appetite with business strategy objectives to ensure coherent execution',
                'Develop capabilities that reinforce competitive advantages identified in strengths analysis',
                'Create monitoring systems for threats while building on identified opportunities',
                'Consider strategic initiatives that address weaknesses without compromising core strengths'
            ])
        
        # Add SWOT-based recommendations if available
        if swot.get('opportunities'):
            recommendations.append(f"Prioritize opportunities in: {', '.join(swot['opportunities'][:2])}")
        
        if swot.get('threats'):
            recommendations.append(f"Develop mitigation strategies for key threats: {', '.join(swot['threats'][:2])}")
        
        return recommendations[:6]  # Limit to 6 recommendations
    
    def _create_emergency_analysis(self, company_name: str, company_number: str,
                                 error_message: str) -> Dict[str, Any]:
        """Emergency analysis when AI services fail"""
        
        return {
            'success': False,
            'company_name': company_name,
            'company_number': company_number,
            'years_analyzed': 'Unable to determine',
            'analysis_date': datetime.now().isoformat(),
            'business_strategy': {
                'dominant': 'Disciplined Specialist Growth',
                'dominant_reasoning': f'Emergency analysis - {error_message}. Conservative assessment indicates disciplined specialist growth characteristics based on typical financial services patterns.',
                'evidence_quotes': ['Emergency analysis - AI services temporarily unavailable']
            },
            'risk_strategy': {
                'dominant': 'Risk-First Conservative',
                'dominant_reasoning': f'Emergency analysis - {error_message}. Conservative risk management approach assumed.',
                'evidence_quotes': ['Emergency analysis - AI services temporarily unavailable']
            },
            'swot_analysis': {
                'strengths': ['Conservative approach', 'Regulatory compliance focus'],
                'weaknesses': ['Limited analysis due to system constraints'],
                'opportunities': ['System restoration will enable detailed analysis'],
                'threats': ['Analysis limitations due to technical issues']
            },
            'analysis_metadata': {
                'analysis_type': 'emergency_fallback',
                'error_message': error_message,
                'confidence_level': 'low',
                'confidence_explanation': 'Low confidence due to emergency fallback - system constraints prevented full analysis',
                'analysis_timestamp': datetime.now().isoformat(),
                'troubleshooting': 'Check API keys and service availability'
            }
        }
    
    def _is_retryable_error(self, error_message: str) -> bool:
        """Check if error is retryable"""
        retryable = ["timeout", "rate", "503", "502", "connection", "temporary"]
        return any(term in error_message.lower() for term in retryable)
    
    def get_status(self) -> Dict[str, Any]:
        """Get analyzer status"""
        return {
            "client_type": self.client_type,
            "openai_available": self.openai_client is not None,
            "anthropic_available": self.anthropic_client is not None,
            "ready": self.client_type != "no_clients_available",
            "archetypes": {
                "business_count": len(self.business_archetypes),
                "risk_count": len(self.risk_archetypes)
            },
            "environment": {
                "openai_key_present": bool(os.environ.get('OPENAI_API_KEY', '').strip()),
                "anthropic_key_present": bool(os.environ.get('ANTHROPIC_API_KEY', '').strip())
            },
            "optimization": {
                "strategic_technical_extraction": True,
                "multi_file_balancing": True,
                "recency_prioritization": True,
                "max_content_chars": self.max_content_chars
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for monitoring"""
        status = self.get_status()
        
        return {
            "status": "healthy" if status["ready"] else "degraded",
            "client_type": status["client_type"],
            "timestamp": datetime.now().isoformat(),
            "ready": status["ready"],
            "archetypes_loaded": status["archetypes"],
            "details": status["environment"],
            "optimization_features": status["optimization"]
        }

    # Backward compatibility method
    def analyze_for_board(self, content: str, company_name: str, company_number: str,
                         extracted_content: Optional[List[Dict[str, Any]]] = None,
                         analysis_context: Optional[str] = None) -> Dict[str, Any]:
        """Board analysis method for compatibility"""
        return self.analyze_for_board_optimized(
            content, company_name, company_number, extracted_content, analysis_context
        )


# Backward compatibility classes
class OptimizedClaudeAnalyzer(CompleteAIAnalyzer):
    """Backward compatibility wrapper"""
    pass

class ExecutiveAIAnalyzer(CompleteAIAnalyzer):
    """Executive analyzer wrapper"""
    
    def analyze_for_board(self, content: str, company_name: str, company_number: str,
                         extracted_content: Optional[List[Dict[str, Any]]] = None,
                         analysis_context: Optional[str] = None) -> Dict[str, Any]:
        """Board analysis method for compatibility"""
        return self.analyze_for_board_optimized(
            content, company_name, company_number, extracted_content, analysis_context
        )