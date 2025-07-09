#!/usr/bin/env python3
"""
AI Analyzer - Complete Enhanced Version with Multi-File Analysis Support
Ensures all files are properly analyzed and taken into account for final archetype classification
"""

import logging
import json
import os
import re
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List

from config import (
    DEFAULT_OPENAI_MODEL, AI_MAX_TOKENS, AI_TEMPERATURE
)

logger = logging.getLogger(__name__)

class AIArchetypeAnalyzer:
    """Enhanced AI-powered analyzer with multi-file support and comprehensive analysis"""
    
    def __init__(self):
        """Initialize the AI analyzer with detailed debugging"""
        logger.info("🚀 AIArchetypeAnalyzer.__init__() starting...")
        self.client = None
        self.client_type = "fallback"
        
        # Business Strategy Archetypes with definitions
        self.business_archetypes = {
            'Scale-through-Distribution': 'Gains share primarily by adding new channels or partners faster than control maturity develops.',
            'Land-Grab Platform': 'Uses aggressive below-market pricing or incentives to build a large multi-sided platform quickly (BNPL, FX apps, etc.).',
            'Asset-Velocity Maximiser': 'Chases rapid originations / turnover (e.g. bridging, invoice finance) even at higher funding costs.',
            'Yield-Hunting': 'Prioritises high-margin segments (credit-impaired, niche commercial) and prices for risk premium.',
            'Fee-Extraction Engine': 'Relies on ancillary fees, add-ons or cross-sales for majority of profit (packaged accounts, paid add-ons).',
            'Disciplined Specialist Growth': 'Niche focus with strong underwriting edge; grows opportunistically while recycling balance-sheet (Together Personal Finance).',
            'Expert Niche Leader': 'Deep expertise in a micro-segment (e.g. HNW Islamic mortgages) with modest but steady growth.',
            'Service-Driven Differentiator': 'Wins by superior client experience / advice rather than price or scale (boutique wealth, mutual insurers).',
            'Cost-Leadership Operator': 'Drives ROE via lean cost base, digital self-service, zero-based budgeting.',
            'Tech-Productivity Accelerator': 'Heavy automation/AI to compress unit costs and redeploy staff (app-only challengers).',
            'Product-Innovation Flywheel': 'Constantly launches novel product variants/features to capture share (fintech disruptors).',
            'Data-Monetisation Pioneer': 'Converts proprietary data into fees (open-banking analytics, credit-insights platforms).',
            'Balance-Sheet Steward': 'Low-risk appetite, prioritises capital strength and membership value (building societies, mutuals).',
            'Regulatory Shelter Occupant': 'Leverages regulatory or franchise protections to defend share (NS&I, Post Office card a/c).',
            'Regulator-Mandated Remediation': 'Operating under s.166, VREQ or RMAR constraints; resources diverted to fix historical failings.',
            'Wind-down / Run-off': 'Managing existing book to maturity or sale; minimal new origination (closed-book life funds).',
            'Strategic Withdrawal': 'Actively divesting lines/geographies to refocus core franchise.',
            'Distressed-Asset Harvester': 'Buys NPLs or under-priced portfolios during downturns for future upside.',
            'Counter-Cyclical Capitaliser': 'Expands lending precisely when competitors retrench, using strong liquidity.'
        }
        
        # Risk Strategy Archetypes with definitions
        self.risk_archetypes = {
            'Risk-First Conservative': 'Prioritises capital preservation and regulatory compliance; growth is secondary to resilience.',
            'Rules-Led Operator': 'Strict adherence to rules and checklists; prioritises control consistency over judgment or speed.',
            'Resilience-Focused Architect': 'Designs for operational continuity and crisis endurance; invests in stress testing and scenario planning.',
            'Strategic Risk-Taker': 'Accepts elevated risk to unlock growth or margin; uses pricing, underwriting, or innovation to offset exposure.',
            'Control-Lag Follower': 'Expands products or markets ahead of control maturity; plays regulatory catch-up after scaling.',
            'Reactive Remediator': 'Risk strategy is event-driven, typically shaped by enforcement, audit findings, or external reviews.',
            'Reputation-First Shield': 'Actively avoids reputational or political risk, sometimes at the expense of commercial logic.',
            'Embedded Risk Partner': 'Risk teams are embedded in frontline decisions; risk appetite is shaped collaboratively across the business.',
            'Quant-Control Enthusiast': 'Leverages data, automation, and predictive analytics as core risk management tools.',
            'Tick-Box Minimalist': 'Superficial control structures exist for compliance optics, not genuine governance intent.',
            'Mission-Driven Prudence': 'Risk appetite is anchored in stakeholder protection, community outcomes, or long-term social licence.'
        }
        
        logger.info("🔧 About to call _setup_client()...")
        self._setup_client()
        logger.info(f"✅ AIArchetypeAnalyzer.__init__() completed. Client type: {self.client_type}")

    def _setup_client(self):
        """Setup the AI client with comprehensive error handling and proxy fix"""
        try:
            logger.info("🔧 AI CLIENT SETUP - Starting initialization...")
            
            # Check for OpenAI API key first
            openai_key = os.getenv('OPENAI_API_KEY')
            logger.info(f"🔑 OpenAI API key check - Found: {bool(openai_key)}")
            
            if openai_key:
                logger.info(f"🔑 OpenAI key length: {len(openai_key)} characters")
                # Safe preview of the key
                if len(openai_key) > 10:
                    logger.info(f"🔑 Key preview: {openai_key[:10]}...{openai_key[-4:]}")
                else:
                    logger.info(f"🔑 Key preview: {openai_key[:5]}... (short key)")
                
                # Check if it's a placeholder
                if openai_key.startswith('your_'):
                    logger.warning("🚨 DETECTED: OpenAI API key is placeholder value!")
                    openai_key = None  # Treat as if not set
                elif openai_key.startswith('sk-'):
                    logger.info("✅ OpenAI key format looks correct (starts with sk-)")
                else:
                    logger.warning("⚠️ OpenAI key format may be incorrect (doesn't start with sk-)")
                    
            if openai_key and openai_key.strip():
                try:
                    logger.info("📦 ATTEMPTING: Import OpenAI library...")
                    import openai
                    logger.info(f"✅ SUCCESS: OpenAI library imported, version: {getattr(openai, '__version__', 'unknown')}")
                    
                    # FIXED: Clear any proxy environment variables that might interfere
                    logger.info("🔧 CLEARING: Proxy environment variables...")
                    original_proxy_vars = {}
                    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
                    
                    # Temporarily clear proxy environment variables
                    for var in proxy_vars:
                        if var in os.environ:
                            original_proxy_vars[var] = os.environ[var]
                            del os.environ[var]
                            logger.info(f"🔧 Temporarily cleared proxy var: {var}")
                    
                    if not original_proxy_vars:
                        logger.info("🔧 No proxy variables found to clear")
                    
                    try:
                        logger.info("🚀 ATTEMPTING: Initialize OpenAI client...")
                        # Initialize OpenAI client with only api_key
                        self.client = openai.OpenAI(api_key=openai_key.strip())
                        self.client_type = "openai"
                        logger.info("✅ SUCCESS: OpenAI client initialized")
                        
                        # Test API connection with a minimal call
                        try:
                            logger.info("🧪 TESTING: OpenAI API connection...")
                            test_response = self.client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "user", "content": "Hi"}],
                                max_tokens=1,
                                temperature=0
                            )
                            logger.info("✅ SUCCESS: OpenAI API test passed")
                            logger.info(f"🧪 Test response ID: {test_response.id}")
                            return  # Success! Exit here
                            
                        except Exception as api_test_error:
                            logger.error(f"🚨 FAILED: OpenAI API test - {type(api_test_error).__name__}: {str(api_test_error)}")
                            
                            # Check specific error types
                            error_str = str(api_test_error).lower()
                            if "api key" in error_str or "authentication" in error_str:
                                logger.error("💡 HINT: API key authentication failed - check if key is valid")
                            elif "quota" in error_str or "billing" in error_str:
                                logger.error("💡 HINT: Quota/billing issue - check OpenAI account status")
                            elif "rate limit" in error_str:
                                logger.error("💡 HINT: Rate limited - will continue with fallback")
                            else:
                                logger.error(f"💡 HINT: Unexpected API error: {str(api_test_error)}")
                            
                            # Keep client for analysis even if test fails
                            logger.warning("⚠️ CONTINUING: Will attempt analysis despite API test failure")
                            
                    finally:
                        # Restore original proxy environment variables
                        for var, value in original_proxy_vars.items():
                            os.environ[var] = value
                            logger.info(f"🔧 Restored proxy var: {var}")
                        
                except ImportError as import_error:
                    logger.error(f"🚨 FAILED: OpenAI import - {str(import_error)}")
                    logger.error("💡 HINT: Make sure 'openai' package is installed")
                except Exception as setup_error:
                    logger.error(f"🚨 FAILED: OpenAI setup - {type(setup_error).__name__}: {str(setup_error)}")
                    logger.error(f"🚨 Setup error details: {repr(setup_error)}")
                    logger.error(f"🚨 Setup traceback: {traceback.format_exc()}")
            else:
                if not openai_key:
                    logger.warning("⚠️ OpenAI API key not found in environment")
                    logger.warning("💡 HINT: Set OPENAI_API_KEY environment variable")
                else:
                    logger.warning("⚠️ OpenAI API key is empty or invalid")
        
            # If we get here, OpenAI failed
            logger.warning("⚠️ FALLBACK: No valid OpenAI client available, using pattern analysis")
            logger.info("💡 HINT: Check your OpenAI API key and network connectivity")
            self.client = None
            self.client_type = "fallback"
            
        except Exception as critical_error:
            logger.error(f"🚨 CRITICAL ERROR in _setup_client: {type(critical_error).__name__}: {str(critical_error)}")
            logger.error(f"🚨 CRITICAL traceback: {traceback.format_exc()}")
            self.client = None
            self.client_type = "fallback"

    def analyze_archetypes(self, content: str, company_name: str, company_number: str, 
                         extracted_content: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced archetype analysis that takes all files into account
        
        Args:
            content: Combined content (legacy parameter)
            company_name: Company name
            company_number: Company number
            extracted_content: List of individual file content data
        """
        try:
            logger.info(f"🏛️ ARCHETYPE ANALYSIS START - Company: {company_name}")
            logger.info(f"🔧 Client type available: {self.client_type}")
            logger.info(f"📊 Combined content length: {len(content):,} characters")
            
            if extracted_content:
                logger.info(f"📁 Individual files available: {len(extracted_content)} files")
                for i, file_data in enumerate(extracted_content):
                    logger.info(f"   File {i+1}: {file_data.get('filename', 'Unknown')} - {len(file_data.get('content', '')):,} chars")
            else:
                logger.info("📁 No individual file data available - using combined content only")
                
            logger.info(f"🤖 Client object: {type(self.client).__name__ if self.client else 'None'}")
            
            if self.client and self.client_type == "openai":
                logger.info(f"🚀 ATTEMPTING: AI-powered multi-file analysis using OpenAI")
                
                try:
                    # Enhanced multi-file analysis
                    if extracted_content and len(extracted_content) > 1:
                        logger.info("🎯 USING: Multi-file analysis approach")
                        business_analysis = self._analyze_multiple_files(
                            extracted_content, self.business_archetypes, "Business Strategy"
                        )
                        risk_analysis = self._analyze_multiple_files(
                            extracted_content, self.risk_archetypes, "Risk Strategy"
                        )
                    else:
                        logger.info("🎯 USING: Single content analysis approach")
                        business_analysis = self._classify_dominant_and_secondary_archetypes(
                            content, self.business_archetypes, "Business Strategy"
                        )
                        risk_analysis = self._classify_dominant_and_secondary_archetypes(
                            content, self.risk_archetypes, "Risk Strategy"
                        )
                    
                    logger.info(f"✅ COMPLETED: Business Strategy - {business_analysis.get('dominant', 'Unknown')}")
                    logger.info(f"✅ COMPLETED: Risk Strategy - {risk_analysis.get('dominant', 'Unknown')}")
                    
                    return self._create_success_result(
                        analysis_type="ai_multi_file_classification" if extracted_content and len(extracted_content) > 1 else "ai_archetype_classification",
                        company_name=company_name,
                        company_number=company_number,
                        business_strategy_archetypes=business_analysis,
                        risk_strategy_archetypes=risk_analysis,
                        model_used=f"openai_{DEFAULT_OPENAI_MODEL}",
                        files_analyzed=len(extracted_content) if extracted_content else 1,
                        total_content_chars=len(content)
                    )
                    
                except Exception as ai_error:
                    logger.error(f"🚨 AI ANALYSIS FAILED: {type(ai_error).__name__}: {str(ai_error)}")
                    logger.error(f"🚨 AI Error details: {repr(ai_error)}")
                    logger.error(f"🚨 AI Error traceback: {traceback.format_exc()}")
                    logger.warning("🔄 FALLING BACK: to pattern-based analysis due to AI failure")
                    return self._fallback_archetype_analysis(content, company_name, company_number, extracted_content)
            else:
                logger.warning(f"🔄 USING FALLBACK: No AI client available (type: {self.client_type})")
                return self._fallback_archetype_analysis(content, company_name, company_number, extracted_content)
                
        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR in analyze_archetypes: {type(e).__name__}: {str(e)}")
            logger.error(f"❌ CRITICAL traceback: {traceback.format_exc()}")
            return self._create_error_result(str(e))

    def _analyze_multiple_files(self, extracted_content: List[Dict[str, Any]], 
                               archetype_dict: Dict[str, str], label: str) -> Dict[str, Any]:
        """
        Analyze multiple files individually and synthesize results
        
        Args:
            extracted_content: List of file content data
            archetype_dict: Dictionary of archetypes
            label: Analysis label (Business Strategy or Risk Strategy)
        """
        logger.info(f"📊 MULTI-FILE ANALYSIS: {label} across {len(extracted_content)} files")
        
        # Analyze each file individually
        individual_analyses = []
        for i, file_data in enumerate(extracted_content):
            logger.info(f"🔍 Analyzing file {i+1}: {file_data.get('filename', 'Unknown')}")
            
            # Use larger content sample per file (up to 12,000 chars)
            file_content = file_data.get('content', '')
            content_sample = file_content[:12000]
            
            try:
                analysis = self._classify_dominant_and_secondary_archetypes(
                    content_sample, archetype_dict, f"{label} - {file_data.get('filename', f'File {i+1}')}"
                )
                analysis['source_file'] = file_data.get('filename', f'File {i+1}')
                analysis['file_date'] = file_data.get('date', 'Unknown')
                analysis['content_length'] = len(file_content)
                analysis['sample_length'] = len(content_sample)
                individual_analyses.append(analysis)
                logger.info(f"✅ File {i+1} analysis: {analysis.get('dominant', 'Unknown')}")
                
            except Exception as file_error:
                logger.error(f"❌ Error analyzing file {i+1}: {str(file_error)}")
                continue
        
        # Synthesize results from all files
        if individual_analyses:
            return self._synthesize_multiple_analyses(individual_analyses, label)
        else:
            logger.warning(f"⚠️ No successful individual analyses for {label}")
            # Fallback to combined analysis
            combined_content = "\n\n".join([f"=== {file_data.get('filename', f'File {i+1}')} ===\n{file_data.get('content', '')[:8000]}" 
                                          for i, file_data in enumerate(extracted_content)])
            return self._classify_dominant_and_secondary_archetypes(combined_content, archetype_dict, label)

    def _synthesize_multiple_analyses(self, individual_analyses: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
        """
        Synthesize results from multiple individual file analyses - ENHANCED to include full reasoning
        
        Args:
            individual_analyses: List of analysis results from individual files
            label: Analysis label
        """
        logger.info(f"🔄 SYNTHESIZING: {len(individual_analyses)} {label} analyses")
        
        # Count occurrences of each archetype
        dominant_counts = {}
        secondary_counts = {}
        all_reasoning = []
        
        for analysis in individual_analyses:
            dominant = analysis.get('dominant', '')
            secondary = analysis.get('secondary', '')
            
            if dominant:
                dominant_counts[dominant] = dominant_counts.get(dominant, 0) + 1
            if secondary:
                secondary_counts[secondary] = secondary_counts.get(secondary, 0) + 1
            
            # Collect FULL reasoning with file context (not truncated)
            file_name = analysis.get('source_file', 'Unknown file')
            reasoning = analysis.get('reasoning', '')
            if reasoning:
                # Include the full reasoning, not just a snippet
                all_reasoning.append(f"[{file_name}] {reasoning}")
        
        # Determine final dominant archetype (most frequent)
        final_dominant = max(dominant_counts.items(), key=lambda x: x[1])[0] if dominant_counts else "Balance-Sheet Steward"
        
        # Determine final secondary archetype
        final_secondary = ""
        if secondary_counts:
            # Get most frequent secondary that's different from dominant
            for archetype, count in sorted(secondary_counts.items(), key=lambda x: x[1], reverse=True):
                if archetype != final_dominant:
                    final_secondary = archetype
                    break
        
        # Create confidence score
        total_files = len(individual_analyses)
        dominant_frequency = dominant_counts.get(final_dominant, 0)
        confidence = dominant_frequency / total_files
        
        # Create comprehensive reasoning that includes ALL evidence
        synthesized_reasoning = f"Multi-file analysis across {total_files} documents shows {final_dominant} as the dominant archetype "
        synthesized_reasoning += f"(appears in {dominant_frequency}/{total_files} files, {confidence:.0%} confidence). "
        
        if final_secondary:
            secondary_frequency = secondary_counts.get(final_secondary, 0)
            synthesized_reasoning += f"Secondary archetype {final_secondary} identified in {secondary_frequency} files. "
        
        # Add ALL reasoning from individual files (not just top 3) - THIS IS THE KEY FIX
        if len(all_reasoning) > 0:
            synthesized_reasoning += "Key evidence: " + "; ".join(all_reasoning)  # Include ALL evidence
        
        result = {
            "dominant": final_dominant,
            "secondary": final_secondary,
            "reasoning": synthesized_reasoning,
            "confidence_score": confidence,
            "analysis_details": {
                "files_analyzed": total_files,
                "dominant_frequency": dominant_frequency,
                "all_dominant_counts": dominant_counts,
                "all_secondary_counts": secondary_counts,
                "individual_analyses": individual_analyses
            }
        }
        
        logger.info(f"🎯 SYNTHESIS RESULT: {final_dominant} ({confidence:.0%} confidence)")
        return result

    def _classify_dominant_and_secondary_archetypes(self, content: str, archetype_dict: Dict[str, str], label: str) -> Dict[str, Any]:
        """Classify archetypes using OpenAI API with enhanced content handling"""
        
        logger.info(f"🎯 CLASSIFYING: {label} using OpenAI API")
        
        # Enhanced content preparation - use more content but intelligently
        content_sample = self._create_intelligent_content_sample(content, max_chars=15000)
        logger.info(f"📝 Content sample length: {len(content_sample)} chars")
        
        # Format archetype definitions
        archetypes_text = "\n".join([f"- {name}: {definition}" for name, definition in archetype_dict.items()])
        
        prompt = f"""You are an expert analyst evaluating a UK financial services firm. Your task is to identify the dominant and secondary {label} Archetypes based on the annual report content below.

Available {label} Archetypes:
{archetypes_text}

Instructions:
1. Analyse the content to understand the firm's strategic approach
2. Select the DOMINANT archetype that most strongly characterises the firm
3. Select a SECONDARY archetype (or "None" if no clear secondary)
4. Provide detailed reasoning based on specific evidence from the content
5. Focus on actual business activities, not just stated intentions

Output format:
**Dominant:** <archetype_name>
**Secondary:** <archetype_name or "None">

**Reasoning:** <detailed explanation with specific evidence>

CONTENT TO ANALYSE:
{content_sample}"""

        try:
            logger.info(f"🚀 MAKING OpenAI API call for {label}...")
            logger.info(f"📝 Prompt length: {len(prompt)} characters")
            
            response = self.client.chat.completions.create(
                model=DEFAULT_OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": f"You are an expert {label.lower()} analyst for UK financial services firms. Focus on evidence-based analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,  # Increased for better reasoning
                temperature=AI_TEMPERATURE
            )
            
            response_text = response.choices[0].message.content
            logger.info(f"✅ OpenAI response received: {len(response_text)} characters")
            logger.info(f"📄 OpenAI response preview: {response_text[:200]}...")
            
            # Parse the response
            result = self._parse_archetype_response(response_text)
            logger.info(f"🎯 PARSED {label}: Dominant={result['dominant']}, Secondary={result.get('secondary', 'None')}")
            
            return result
            
        except Exception as api_error:
            logger.error(f"🚨 API ERROR for {label}: {type(api_error).__name__}: {str(api_error)}")
            logger.error(f"🚨 API Error repr: {repr(api_error)}")
            logger.error(f"🚨 API Error traceback: {traceback.format_exc()}")
            
            # Fallback to pattern analysis for this category
            logger.warning(f"🔄 FALLBACK: Using pattern analysis for {label}")
            return self._fallback_single_archetype_analysis(content, archetype_dict, label)

    def _create_intelligent_content_sample(self, content: str, max_chars: int = 15000) -> str:
        """
        Create intelligent content sample that preserves key strategic sections
        
        Args:
            content: Full content to sample
            max_chars: Maximum characters to include
        """
        if len(content) <= max_chars:
            return content
        
        # Priority sections to look for
        priority_sections = [
            'strategic report', 'business strategy', 'risk management', 'principal risks',
            'business model', 'strategy', 'outlook', 'governance', 'executive summary',
            'chairman', 'chief executive', 'performance', 'objectives'
        ]
        
        # Try to find and prioritize key sections
        content_lower = content.lower()
        important_sections = []
        
        for section in priority_sections:
            # Look for section headers
            patterns = [
                f'{section}',
                f'{section} report',
                f'{section} statement'
            ]
            
            for pattern in patterns:
                start_pos = content_lower.find(pattern)
                if start_pos != -1:
                    # Extract section (up to 3000 chars)
                    section_content = content[start_pos:start_pos + 3000]
                    important_sections.append(section_content)
                    break
        
        # Combine important sections
        if important_sections:
            combined_important = "\n\n".join(important_sections)
            remaining_chars = max_chars - len(combined_important)
            
            if remaining_chars > 1000:
                # Add beginning of document
                start_content = content[:remaining_chars]
                return start_content + "\n\n=== KEY SECTIONS ===\n\n" + combined_important
            else:
                return combined_important[:max_chars]
        
        # Fallback: take from beginning and end
        half_chars = max_chars // 2
        start_content = content[:half_chars]
        end_content = content[-half_chars:]
        
        return start_content + "\n\n=== [DOCUMENT END] ===\n\n" + end_content

    def _parse_archetype_response(self, response: str) -> Dict[str, str]:
        """Parse archetype response from AI - handles both regular and markdown formatting"""
        result = {"dominant": "", "secondary": "", "reasoning": ""}
        
        # Debug: Log the raw response
        logger.info(f"🔍 RAW RESPONSE TO PARSE: {repr(response[:500])}")
        
        lines = response.strip().split('\n')
        reasoning_started = False
        reasoning_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            logger.info(f"🔍 PARSING LINE {i}: {repr(line)}")
            
            # Handle both regular and bold markdown formatting
            if line.startswith("Dominant:") or line.startswith("**Dominant:**"):
                value = line.replace("Dominant:", "").replace("**Dominant:**", "").strip()
                # Clean any remaining asterisks and extra whitespace
                result["dominant"] = re.sub(r'\*+', '', value).strip()
                logger.info(f"🔍 FOUND DOMINANT: {result['dominant']}")
                
            elif line.startswith("Secondary:") or line.startswith("**Secondary:**"):
                value = line.replace("Secondary:", "").replace("**Secondary:**", "").strip()
                # Clean any remaining asterisks and extra whitespace
                value = re.sub(r'\*+', '', value).strip()
                result["secondary"] = value if value.lower() != "none" else ""
                logger.info(f"🔍 FOUND SECONDARY: {result['secondary']}")
                
            elif line.startswith("Reasoning:") or line.startswith("**Reasoning:**"):
                logger.info(f"🔍 FOUND REASONING HEADER")
                # Check if reasoning is on the same line
                value = line.replace("Reasoning:", "").replace("**Reasoning:**", "").strip()
                value = re.sub(r'\*+', '', value).strip()  # Clean asterisks first
                if value:
                    # Reasoning is on the same line
                    result["reasoning"] = value
                    logger.info(f"🔍 REASONING ON SAME LINE: {result['reasoning']}")
                else:
                    # Reasoning starts on the next line
                    reasoning_started = True
                    logger.info(f"🔍 REASONING STARTS ON NEXT LINE")
                    
            elif reasoning_started:
                # Collect all remaining lines as reasoning
                if line:  # Skip empty lines
                    reasoning_lines.append(line)
                    logger.info(f"🔍 ADDED TO REASONING: {line}")
        
        # If we collected reasoning lines, join them
        if reasoning_lines:
            reasoning_text = ' '.join(reasoning_lines)
            result["reasoning"] = re.sub(r'\*+', '', reasoning_text).strip()
            logger.info(f"🔍 FINAL REASONING: {result['reasoning'][:200]}...")
        
        logger.info(f"🔍 FINAL PARSED RESULT: {result}")
        return result

    def _fallback_archetype_analysis(self, content: str, company_name: str, company_number: str, 
                                   extracted_content: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhanced pattern-based fallback analysis"""
        logger.info("🔄 EXECUTING: Enhanced pattern-based archetype analysis")
        
        business_analysis = self._fallback_single_archetype_analysis(
            content, self.business_archetypes, "Business Strategy"
        )
        
        risk_