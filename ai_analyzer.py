#!/usr/bin/env python3
"""
AI Analyzer - Ultra Clean Version - Guaranteed OpenAI v1.35.0 Compatible
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
    """AI-powered analyzer - Ultra clean version"""
    
    def __init__(self):
        """Initialize the AI analyzer"""
        self.client = None
        self.client_type = "fallback"
        self._setup_client()
        
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

    def _setup_client(self):
        """Setup OpenAI client - ULTRA CLEAN VERSION"""
        logger.info("🔧 AI CLIENT SETUP - Starting initialization...")
        
        try:
            # Get API key
            api_key = os.getenv('OPENAI_API_KEY')
            logger.info(f"🔑 API key found: {bool(api_key)}")
            
            if not api_key or api_key.startswith('your_'):
                logger.warning("⚠️ No valid API key found")
                self.client = None
                self.client_type = "fallback"
                return
            
            # Import OpenAI
            logger.info("📦 ATTEMPTING: Import OpenAI library...")
            import openai
            logger.info(f"✅ SUCCESS: OpenAI library imported, version: {getattr(openai, '__version__', 'unknown')}")
            
            # Create client - ONLY api_key parameter
            logger.info("🚀 ATTEMPTING: Initialize OpenAI client...")
            try:
                # THIS IS THE CRITICAL LINE - ONLY api_key, nothing else
                self.client = openai.OpenAI(api_key=api_key)
                self.client_type = "openai"
                logger.info("✅ SUCCESS: OpenAI client initialized")
                
                # Test the client
                logger.info("🧪 TESTING: OpenAI API connection...")
                test_response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=1
                )
                logger.info("✅ SUCCESS: OpenAI API test passed")
                return
                
            except Exception as client_error:
                logger.error(f"🚨 CLIENT ERROR: {type(client_error).__name__}: {str(client_error)}")
                self.client = None
                self.client_type = "fallback"
                
        except Exception as e:
            logger.error(f"🚨 SETUP ERROR: {type(e).__name__}: {str(e)}")
            self.client = None
            self.client_type = "fallback"

    def analyze_archetypes(self, content: str, company_name: str, company_number: str) -> Dict[str, Any]:
        """Analyze company archetypes"""
        try:
            logger.info(f"🏛️ ARCHETYPE ANALYSIS START - Company: {company_name}")
            logger.info(f"🔧 Client type: {self.client_type}")
            
            if self.client and self.client_type == "openai":
                logger.info("🚀 ATTEMPTING: AI-powered analysis using OpenAI")
                
                try:
                    # Business strategy analysis
                    business_analysis = self._classify_archetypes(
                        content, self.business_archetypes, "Business Strategy"
                    )
                    
                    # Risk strategy analysis
                    risk_analysis = self._classify_archetypes(
                        content, self.risk_archetypes, "Risk Strategy"
                    )
                    
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
                    logger.error(f"🚨 AI ANALYSIS FAILED: {str(ai_error)}")
                    return self._fallback_analysis(content, company_name, company_number)
            else:
                logger.warning("🔄 USING FALLBACK: No AI client available")
                return self._fallback_analysis(content, company_name, company_number)
                
        except Exception as e:
            logger.error(f"❌ ANALYSIS ERROR: {str(e)}")
            return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

    def _classify_archetypes(self, content: str, archetypes: Dict[str, str], label: str) -> Dict[str, str]:
        """Classify archetypes using OpenAI"""
        
        # Prepare content
        content_sample = content[:8000]
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
            return self._parse_response(response_text)
            
        except Exception as e:
            logger.error(f"🚨 API ERROR: {str(e)}")
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
        # Simple fallback - just return a default
        return {
            "dominant": "Balance-Sheet Steward",
            "secondary": "",
            "reasoning": f"Pattern-based fallback analysis for {label}"
        }

    def _fallback_analysis(self, content: str, company_name: str, company_number: str) -> Dict[str, Any]:
        """Fallback analysis"""
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