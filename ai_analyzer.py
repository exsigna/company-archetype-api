#!/usr/bin/env python3
"""
AI Analyzer - Fixed Version with Enhanced Debugging, Proxy Fix, and Parser Fix
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
    """AI-powered analyzer with extensive debugging"""
    
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

    def analyze_archetypes(self, content: str, company_name: str, company_number: str) -> Dict[str, Any]:
        """Analyze company archetypes with detailed logging"""
        try:
            logger.info(f"🏛️ ARCHETYPE ANALYSIS START - Company: {company_name}")
            logger.info(f"🔧 Client type available: {self.client_type}")
            logger.info(f"📊 Content length: {len(content):,} characters")
            logger.info(f"🤖 Client object: {type(self.client).__name__ if self.client else 'None'}")
            
            if self.client and self.client_type == "openai":
                logger.info(f"🚀 ATTEMPTING: AI-powered analysis using OpenAI")
                
                try:
                    # Test business strategy analysis
                    logger.info("🎯 STARTING: Business Strategy archetype classification...")
                    business_analysis = self._classify_dominant_and_secondary_archetypes(
                        content, self.business_archetypes, "Business Strategy"
                    )
                    logger.info(f"✅ COMPLETED: Business Strategy - {business_analysis.get('dominant', 'Unknown')}")
                    
                    # Test risk strategy analysis
                    logger.info("🎯 STARTING: Risk Strategy archetype classification...")
                    risk_analysis = self._classify_dominant_and_secondary_archetypes(
                        content, self.risk_archetypes, "Risk Strategy"
                    )
                    logger.info(f"✅ COMPLETED: Risk Strategy - {risk_analysis.get('dominant', 'Unknown')}")
                    
                    return self._create_success_result(
                        analysis_type="ai_archetype_classification",
                        company_name=company_name,
                        company_number=company_number,
                        business_strategy_archetypes=business_analysis,
                        risk_strategy_archetypes=risk_analysis,
                        model_used=f"openai_{DEFAULT_OPENAI_MODEL}"
                    )
                    
                except Exception as ai_error:
                    logger.error(f"🚨 AI ANALYSIS FAILED: {type(ai_error).__name__}: {str(ai_error)}")
                    logger.error(f"🚨 AI Error details: {repr(ai_error)}")
                    logger.error(f"🚨 AI Error traceback: {traceback.format_exc()}")
                    logger.warning("🔄 FALLING BACK: to pattern-based analysis due to AI failure")
                    return self._fallback_archetype_analysis(content, company_name, company_number)
            else:
                logger.warning(f"🔄 USING FALLBACK: No AI client available (type: {self.client_type})")
                return self._fallback_archetype_analysis(content, company_name, company_number)
                
        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR in analyze_archetypes: {type(e).__name__}: {str(e)}")
            logger.error(f"❌ CRITICAL traceback: {traceback.format_exc()}")
            return self._create_error_result(str(e))

    def _classify_dominant_and_secondary_archetypes(self, content: str, archetype_dict: Dict[str, str], label: str) -> Dict[str, Any]:
        """Classify archetypes using OpenAI API"""
        
        logger.info(f"🎯 CLASSIFYING: {label} using OpenAI API")
        
        # Prepare content (truncate for API limits)
        content_sample = content[:8000]
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
4. Provide detailed reasoning based on specific evidence

Output format:
Dominant: <archetype_name>
Secondary: <archetype_name or "None">
Reasoning: <detailed explanation>

TEXT TO ANALYSE:
{content_sample}"""

        try:
            logger.info(f"🚀 MAKING OpenAI API call for {label}...")
            logger.info(f"📝 Prompt length: {len(prompt)} characters")
            
            response = self.client.chat.completions.create(
                model=DEFAULT_OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": f"You are an expert {label.lower()} analyst for UK financial services firms."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
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

    def _fallback_archetype_analysis(self, content: str, company_name: str, company_number: str) -> Dict[str, Any]:
        """Pattern-based fallback analysis"""
        logger.info("🔄 EXECUTING: Pattern-based archetype analysis")
        
        business_analysis = self._fallback_single_archetype_analysis(
            content, self.business_archetypes, "Business Strategy"
        )
        
        risk_analysis = self._fallback_single_archetype_analysis(
            content, self.risk_archetypes, "Risk Strategy"
        )
        
        return self._create_success_result(
            analysis_type="pattern_archetype_classification",
            company_name=company_name,
            company_number=company_number,
            business_strategy_archetypes=business_analysis,
            risk_strategy_archetypes=risk_analysis
        )

    def _fallback_single_archetype_analysis(self, content: str, archetype_dict: Dict[str, str], label: str) -> Dict[str, str]:
        """Pattern-based analysis for single category"""
        content_lower = content.lower()
        archetype_scores = {}
        
        # Keyword patterns for each archetype
        if label == "Business Strategy":
            keyword_patterns = {
                'Disciplined Specialist Growth': ['specialist', 'niche', 'underwriting', 'opportunistic', 'recycling'],
                'Balance-Sheet Steward': ['capital.*strength', 'prudent', 'conservative', 'steward', 'membership'],
                'Service-Driven Differentiator': ['customer.*experience', 'service.*quality', 'advice', 'client.*experience'],
                'Cost-Leadership Operator': ['efficiency', 'cost.*base', 'lean', 'digital.*self.*service'],
                'Expert Niche Leader': ['expertise', 'micro.*segment', 'deep.*knowledge', 'specialist.*knowledge'],
                'Tech-Productivity Accelerator': ['automation', 'technology', 'digital', 'efficiency.*gains'],
                'Regulatory Shelter Occupant': ['regulatory.*protection', 'franchise.*protection'],
                'Regulator-Mandated Remediation': ['remediation', 'regulatory.*action', 'enforcement', 'improvement.*programme']
            }
        else:  # Risk Strategy
            keyword_patterns = {
                'Rules-Led Operator': ['compliance', 'regulatory.*requirements', 'procedures', 'controls'],
                'Risk-First Conservative': ['capital.*preservation', 'regulatory.*compliance', 'resilience', 'conservative'],
                'Resilience-Focused Architect': ['operational.*continuity', 'stress.*testing', 'scenario.*planning', 'crisis'],
                'Embedded Risk Partner': ['embedded.*risk', 'collaborative', 'integrated.*risk'],
                'Reputation-First Shield': ['reputation', 'reputational.*risk', 'brand.*protection'],
                'Strategic Risk-Taker': ['risk.*appetite', 'growth.*oriented', 'calculated.*risk'],
                'Mission-Driven Prudence': ['stakeholder.*protection', 'community', 'social.*licence', 'mission']
            }
        
        # Score each archetype
        for archetype, patterns in keyword_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content_lower))
                score += matches
            archetype_scores[archetype] = score
        
        # Select top archetypes
        sorted_archetypes = sorted(archetype_scores.items(), key=lambda x: x[1], reverse=True)
        
        dominant = sorted_archetypes[0][0] if sorted_archetypes and sorted_archetypes[0][1] > 0 else "Balance-Sheet Steward"
        secondary = sorted_archetypes[1][0] if len(sorted_archetypes) > 1 and sorted_archetypes[1][1] > 0 else ""
        
        reasoning = f"Pattern-based analysis identified {dominant} as the dominant archetype"
        if secondary:
            reasoning += f" with {secondary} as secondary archetype"
        reasoning += f" based on keyword frequency analysis of the annual report content."
        
        return {
            "dominant": dominant,
            "secondary": secondary,
            "reasoning": reasoning
        }

    def _create_success_result(self, analysis_type: str, company_name: str, 
                             company_number: str, business_strategy_archetypes: Dict[str, str],
                             risk_strategy_archetypes: Dict[str, str], model_used: str = None) -> Dict[str, Any]:
        """Create success result"""
        result = {
            "success": True,
            "analysis_type": analysis_type,
            "company_name": company_name,
            "company_number": company_number,
            "business_strategy_archetypes": business_strategy_archetypes,
            "risk_strategy_archetypes": risk_strategy_archetypes,
            "timestamp": datetime.now().isoformat(),
        }
        
        if model_used:
            result["model_used"] = model_used
            
        return result

    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            "success": False,
            "error": error_msg,
            "analysis_type": "error",
            "timestamp": datetime.now().isoformat()
        }