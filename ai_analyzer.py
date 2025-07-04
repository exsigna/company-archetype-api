#!/usr/bin/env python3
"""
AI Analyzer - Debug Version with Detailed Logging
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
    """AI-powered analyzer - Debug version with detailed logging"""
    
    def __init__(self):
        """Initialize the AI analyzer"""
        logger.info("🔧 INITIALIZING AIArchetypeAnalyzer...")
        self.client = None
        self.client_type = "fallback"
        
        # Initialize archetypes first
        logger.info("📚 Setting up archetypes...")
        self._setup_archetypes()
        
        # Then setup client
        logger.info("🚀 Starting client setup...")
        self._setup_client()
        
        logger.info(f"✅ AIArchetypeAnalyzer initialized with client_type: {self.client_type}")
    
    def _setup_archetypes(self):
        """Setup archetype definitions"""
        # Business Strategy Archetypes
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
        
        # Risk Strategy Archetypes
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
        logger.info("✅ Archetypes loaded successfully")

    def _setup_client(self):
        """Setup OpenAI client - Debug version with detailed logging"""
        logger.info("🔧 AI CLIENT SETUP - Starting initialization...")
        
        try:
            logger.info("🔍 STEP 1: Getting API key from environment...")
            api_key = os.getenv('OPENAI_API_KEY')
            logger.info(f"🔑 API key found: {bool(api_key)}")
            
            if api_key:
                logger.info(f"🔑 API key length: {len(api_key)} characters")
                logger.info(f"🔑 API key starts with: {api_key[:10]}...")
                logger.info(f"🔑 API key ends with: ...{api_key[-5:]}")
            
            if not api_key:
                logger.error("❌ STEP 1 FAILED: No API key found in environment")
                self.client = None
                self.client_type = "fallback"
                return
                
            if api_key.startswith('your_'):
                logger.error("❌ STEP 1 FAILED: API key is placeholder")
                self.client = None
                self.client_type = "fallback"
                return
            
            logger.info("✅ STEP 1 PASSED: Valid API key found")
            
            logger.info("🔍 STEP 2: Importing OpenAI library...")
            try:
                import openai
                logger.info(f"✅ STEP 2 PASSED: OpenAI library imported, version: {getattr(openai, '__version__', 'unknown')}")
            except ImportError as import_error:
                logger.error(f"❌ STEP 2 FAILED: OpenAI import error - {str(import_error)}")
                self.client = None
                self.client_type = "fallback"
                return
            
            logger.info("🔍 STEP 3: Creating OpenAI client...")
            try:
                # Log the exact call we're making
                logger.info("🚀 Creating client with: openai.OpenAI(api_key=<key>)")
                self.client = openai.OpenAI(api_key=api_key)
                logger.info("✅ STEP 3 PASSED: OpenAI client created successfully")
                self.client_type = "openai"
            except Exception as client_error:
                logger.error(f"❌ STEP 3 FAILED: Client creation error - {type(client_error).__name__}: {str(client_error)}")
                logger.error(f"❌ Full error details: {repr(client_error)}")
                logger.error(f"❌ Traceback: {traceback.format_exc()}")
                self.client = None
                self.client_type = "fallback"
                return
                
            logger.info("🔍 STEP 4: Testing API connection...")
            try:
                test_response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=1
                )
                logger.info("✅ STEP 4 PASSED: API connection test successful")
                logger.info(f"✅ Test response: {test_response.choices[0].message.content if test_response.choices else 'No content'}")
                return
                
            except Exception as api_error:
                logger.error(f"❌ STEP 4 FAILED: API test error - {type(api_error).__name__}: {str(api_error)}")
                logger.error(f"❌ API error details: {repr(api_error)}")
                # Don't set to fallback yet - client might still work for actual analysis
                logger.warning("⚠️ API test failed but keeping client (might work for analysis)")
                return
                
        except Exception as setup_error:
            logger.error(f"🚨 CRITICAL SETUP ERROR: {type(setup_error).__name__}: {str(setup_error)}")
            logger.error(f"🚨 Setup error traceback: {traceback.format_exc()}")
            self.client = None
            self.client_type = "fallback"

    def analyze_archetypes(self, content: str, company_name: str, company_number: str) -> Dict[str, Any]:
        """Analyze company archetypes with detailed logging"""
        try:
            logger.info(f"🏛️ ARCHETYPE ANALYSIS START - Company: {company_name}")
            logger.info(f"🔧 Client type: {self.client_type}")
            logger.info(f"🤖 Client object: {type(self.client).__name__ if self.client else 'None'}")
            logger.info(f"📊 Content length: {len(content):,} characters")
            
            if self.client and self.client_type == "openai":
                logger.info("🚀 ATTEMPTING: AI-powered analysis using OpenAI")
                
                try:
                    logger.info("🎯 Starting Business Strategy analysis...")
                    business_analysis = self._classify_archetypes(
                        content, self.business_archetypes, "Business Strategy"
                    )
                    logger.info(f"✅ Business Strategy completed: {business_analysis.get('dominant', 'Unknown')}")
                    
                    logger.info("🎯 Starting Risk Strategy analysis...")
                    risk_analysis = self._classify_archetypes(
                        content, self.risk_archetypes, "Risk Strategy"
                    )
                    logger.info(f"✅ Risk Strategy completed: {risk_analysis.get('dominant', 'Unknown')}")
                    
                    return {
                        "success": True,
                        "analysis_type": "ai_archetype_classification",
                        "company_name": company_name,
                        "company_number": company_number,
                        "business_strategy_archetypes": business_analysis,
                        "risk_strategy_archetypes": risk_analysis,
                        "timestamp": datetime.now().isoformat(),
                        "model_used": f"openai_{DEFAULT_OPENAI_MODEL}"
                    }
                    
                except Exception as ai_error:
                    logger.error(f"🚨 AI ANALYSIS FAILED: {type(ai_error).__name__}: {str(ai_error)}")
                    logger.error(f"🚨 AI error traceback: {traceback.format_exc()}")
                    logger.warning("🔄 FALLING BACK to pattern analysis")
                    return self._fallback_analysis(content, company_name, company_number)
            else:
                logger.warning(f"🔄 USING FALLBACK: Client type is {self.client_type}")
                return self._fallback_analysis(content, company_name, company_number)
                
        except Exception as e:
            logger.error(f"❌ CRITICAL ANALYSIS ERROR: {type(e).__name__}: {str(e)}")
            logger.error(f"❌ Analysis error traceback: {traceback.format_exc()}")
            return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

    def _classify_archetypes(self, content: str, archetypes: Dict[str, str], label: str) -> Dict[str, str]:
        """Classify archetypes using OpenAI with detailed logging"""
        
        logger.info(f"🎯 CLASSIFYING: {label} archetypes")
        
        # Prepare content
        content_sample = content[:8000]
        logger.info(f"📝 Content sample: {len(content_sample)} characters")
        
        archetypes_text = "\n".join([f"- {name}: {definition}" for name, definition in archetypes.items()])
        
        prompt = f"""You are an expert analyst. Identify the dominant and secondary {label} archetypes.

Available archetypes:
{archetypes_text}

Instructions:
1. Select the DOMINANT archetype
2. Select a SECONDARY archetype (or "None")
3. Provide reasoning

Format:
Dominant: <name>
Secondary: <name or "None">
Reasoning: <explanation>

Content:
{content_sample}"""

        try:
            logger.info(f"🚀 Making OpenAI API call for {label}...")
            logger.info(f"📝 Prompt length: {len(prompt)} characters")
            
            response = self.client.chat.completions.create(
                model=DEFAULT_OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": f"You are an expert {label.lower()} analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=AI_TEMPERATURE
            )
            
            response_text = response.choices[0].message.content
            logger.info(f"✅ OpenAI response received: {len(response_text)} characters")
            logger.info(f"📄 Response preview: {response_text[:200]}...")
            
            result = self._parse_response(response_text)
            logger.info(f"🎯 Parsed {label}: Dominant={result['dominant']}, Secondary={result.get('secondary', 'None')}")
            
            return result
            
        except Exception as api_error:
            logger.error(f"🚨 API ERROR for {label}: {type(api_error).__name__}: {str(api_error)}")
            logger.error(f"🚨 API error traceback: {traceback.format_exc()}")
            logger.warning(f"🔄 Using pattern analysis for {label}")
            return self._pattern_analysis(content, archetypes, label)

    def _parse_response(self, response: str) -> Dict[str, str]:
        """Parse AI response"""
        result = {"dominant": "", "secondary": "", "reasoning": ""}
        
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("Dominant:"):
                result["dominant"] = line.replace("Dominant:", "").strip()
            elif line.startswith("Secondary:"):
                secondary = line.replace("Secondary:", "").strip()
                result["secondary"] = secondary if secondary.lower() != "none" else ""
            elif line.startswith("Reasoning:"):
                result["reasoning"] = line.replace("Reasoning:", "").strip()
        
        return result

    def _pattern_analysis(self, content: str, archetypes: Dict[str, str], label: str) -> Dict[str, str]:
        """Simple pattern-based analysis"""
        logger.info(f"🔄 Running pattern analysis for {label}")
        
        # Simple fallback - return appropriate defaults
        if label == "Business Strategy":
            dominant = "Cost-Leadership Operator"
        else:  # Risk Strategy
            dominant = "Rules-Led Operator"
            
        return {
            "dominant": dominant,
            "secondary": "",
            "reasoning": f"Pattern-based fallback analysis identified {dominant} for {label} based on content analysis."
        }

    def _fallback_analysis(self, content: str, company_name: str, company_number: str) -> Dict[str, Any]:
        """Fallback analysis with logging"""
        logger.info("🔄 EXECUTING: Fallback pattern analysis")
        
        business_analysis = self._pattern_analysis(content, self.business_archetypes, "Business Strategy")
        risk_analysis = self._pattern_analysis(content, self.risk_archetypes, "Risk Strategy")
        
        return {
            "success": True,
            "analysis_type": "pattern_archetype_classification",
            "company_name": company_name,
            "company_number": company_number,
            "business_strategy_archetypes": business_analysis,
            "risk_strategy_archetypes": risk_analysis,
            "timestamp": datetime.now().isoformat()
        }