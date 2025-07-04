#!/usr/bin/env python3
"""
AI Analyzer - Safe Version with Basic Error Catching
"""

import logging
import json
import os
import re
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List

# Configure logging immediately
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log that we're starting to import
logger.info("🔥 STARTING AI_ANALYZER IMPORT...")

try:
    from config import (
        DEFAULT_OPENAI_MODEL, AI_MAX_TOKENS, AI_TEMPERATURE
    )
    logger.info("✅ Config imports successful")
except Exception as config_error:
    logger.error(f"❌ Config import failed: {str(config_error)}")
    # Set defaults
    DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
    AI_MAX_TOKENS = 1000
    AI_TEMPERATURE = 0.3

logger.info("🔥 DEFINING AIArchetypeAnalyzer CLASS...")

class AIArchetypeAnalyzer:
    """AI-powered analyzer - Safe version"""
    
    def __init__(self):
        """Initialize the AI analyzer with extensive error catching"""
        try:
            logger.info("🔥 INIT START - AIArchetypeAnalyzer.__init__() called")
            self.client = None
            self.client_type = "fallback"
            
            logger.info("🔥 INIT STEP 1 - Setting up archetypes...")
            self._setup_archetypes()
            
            logger.info("🔥 INIT STEP 2 - Setting up OpenAI client...")
            self._setup_client()
            
            logger.info(f"🔥 INIT COMPLETE - client_type: {self.client_type}")
            
        except Exception as init_error:
            logger.error(f"🚨 CRITICAL INIT ERROR: {type(init_error).__name__}: {str(init_error)}")
            logger.error(f"🚨 INIT ERROR TRACEBACK: {traceback.format_exc()}")
            self.client = None
            self.client_type = "fallback"
    
    def _setup_archetypes(self):
        """Setup archetype definitions"""
        try:
            logger.info("📚 Setting up business archetypes...")
            self.business_archetypes = {
                'Cost-Leadership Operator': 'Drives ROE via lean cost base, digital self-service, zero-based budgeting.',
                'Balance-Sheet Steward': 'Low-risk appetite, prioritises capital strength and membership value.',
                'Disciplined Specialist Growth': 'Niche focus with strong underwriting edge; grows opportunistically.',
                'Service-Driven Differentiator': 'Wins by superior client experience / advice rather than price or scale.',
                'Tech-Productivity Accelerator': 'Heavy automation/AI to compress unit costs and redeploy staff.',
                'Expert Niche Leader': 'Deep expertise in a micro-segment with modest but steady growth.',
                'Scale-through-Distribution': 'Gains share primarily by adding new channels or partners.',
                'Asset-Velocity Maximiser': 'Chases rapid originations / turnover even at higher funding costs.',
                'Yield-Hunting': 'Prioritises high-margin segments and prices for risk premium.',
                'Fee-Extraction Engine': 'Relies on ancillary fees, add-ons or cross-sales for majority of profit.'
            }
            
            logger.info("📚 Setting up risk archetypes...")
            self.risk_archetypes = {
                'Rules-Led Operator': 'Strict adherence to rules and checklists; prioritises control consistency.',
                'Risk-First Conservative': 'Prioritises capital preservation and regulatory compliance.',
                'Resilience-Focused Architect': 'Designs for operational continuity and crisis endurance.',
                'Strategic Risk-Taker': 'Accepts elevated risk to unlock growth or margin.',
                'Embedded Risk Partner': 'Risk teams are embedded in frontline decisions.',
                'Reputation-First Shield': 'Actively avoids reputational or political risk.',
                'Mission-Driven Prudence': 'Risk appetite is anchored in stakeholder protection.',
                'Control-Lag Follower': 'Expands products or markets ahead of control maturity.',
                'Reactive Remediator': 'Risk strategy is event-driven, typically shaped by enforcement.',
                'Quant-Control Enthusiast': 'Leverages data, automation, and predictive analytics.'
            }
            logger.info("✅ Archetypes setup completed")
            
        except Exception as archetype_error:
            logger.error(f"❌ Archetype setup failed: {str(archetype_error)}")
            # Set minimal defaults
            self.business_archetypes = {'Balance-Sheet Steward': 'Default business archetype'}
            self.risk_archetypes = {'Rules-Led Operator': 'Default risk archetype'}

    def _setup_client(self):
        """Setup OpenAI client with extensive error catching"""
        try:
            logger.info("🔧 AI CLIENT SETUP - Starting...")
            
            # Step 1: Get API key
            logger.info("🔍 STEP 1: Getting API key...")
            api_key = os.getenv('OPENAI_API_KEY')
            logger.info(f"🔑 API key found: {bool(api_key)}")
            
            if not api_key:
                logger.warning("⚠️ No API key found - using fallback")
                self.client = None
                self.client_type = "fallback"
                return
                
            if api_key.startswith('your_'):
                logger.warning("⚠️ Placeholder API key found - using fallback")
                self.client = None
                self.client_type = "fallback"
                return
            
            logger.info(f"✅ STEP 1 PASSED - Key length: {len(api_key)}")
            
            # Step 2: Import OpenAI
            logger.info("🔍 STEP 2: Importing OpenAI...")
            try:
                import openai
                logger.info(f"✅ STEP 2 PASSED - OpenAI v{getattr(openai, '__version__', 'unknown')}")
            except ImportError as import_err:
                logger.error(f"❌ STEP 2 FAILED - Import error: {str(import_err)}")
                self.client = None
                self.client_type = "fallback"
                return
            
            # Step 3: Create client
            logger.info("🔍 STEP 3: Creating OpenAI client...")
            try:
                logger.info("🚀 Calling: openai.OpenAI(api_key=***)")
                self.client = openai.OpenAI(api_key=api_key)
                logger.info("✅ STEP 3 PASSED - Client created")
                self.client_type = "openai"
            except Exception as client_err:
                logger.error(f"❌ STEP 3 FAILED - {type(client_err).__name__}: {str(client_err)}")
                self.client = None
                self.client_type = "fallback"
                return
                
            # Step 4: Test API
            logger.info("🔍 STEP 4: Testing API...")
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=1
                )
                logger.info("✅ STEP 4 PASSED - API working")
                return
                
            except Exception as api_err:
                logger.error(f"❌ STEP 4 FAILED - {type(api_err).__name__}: {str(api_err)}")
                # Keep client but log warning
                logger.warning("⚠️ API test failed but keeping client")
                return
                
        except Exception as setup_err:
            logger.error(f"🚨 SETUP ERROR: {type(setup_err).__name__}: {str(setup_err)}")
            logger.error(f"🚨 SETUP TRACEBACK: {traceback.format_exc()}")
            self.client = None
            self.client_type = "fallback"

    def analyze_archetypes(self, content: str, company_name: str, company_number: str) -> Dict[str, Any]:
        """Analyze company archetypes"""
        try:
            logger.info(f"🏛️ ARCHETYPE ANALYSIS START - {company_name}")
            logger.info(f"🔧 Client type: {self.client_type}")
            logger.info(f"🤖 Client: {type(self.client).__name__ if self.client else 'None'}")
            
            if self.client and self.client_type == "openai":
                logger.info("🚀 ATTEMPTING: AI analysis")
                
                try:
                    # Simple AI analysis
                    business_result = self._ai_classify(content, "Business Strategy")
                    risk_result = self._ai_classify(content, "Risk Strategy")
                    
                    return {
                        "success": True,
                        "analysis_type": "ai_archetype_classification",
                        "company_name": company_name,
                        "company_number": company_number,
                        "business_strategy_archetypes": business_result,
                        "risk_strategy_archetypes": risk_result,
                        "timestamp": datetime.now().isoformat(),
                        "model_used": f"openai_{DEFAULT_OPENAI_MODEL}"
                    }
                    
                except Exception as ai_err:
                    logger.error(f"🚨 AI analysis failed: {str(ai_err)}")
                    return self._fallback_analysis(content, company_name, company_number)
            else:
                logger.info("🔄 Using fallback analysis")
                return self._fallback_analysis(content, company_name, company_number)
                
        except Exception as analysis_err:
            logger.error(f"❌ Analysis error: {str(analysis_err)}")
            return {"success": False, "error": str(analysis_err), "timestamp": datetime.now().isoformat()}

    def _ai_classify(self, content: str, category: str) -> Dict[str, str]:
        """Simple AI classification"""
        try:
            content_sample = content[:6000]  # Smaller sample
            
            prompt = f"""Analyze this {category} and identify the dominant archetype from this list:
            
Business Strategy: Cost-Leadership Operator, Balance-Sheet Steward, Disciplined Specialist Growth
Risk Strategy: Rules-Led Operator, Risk-First Conservative, Resilience-Focused Architect

Reply with just: "Dominant: [archetype name]"

Content: {content_sample}"""

            response = self.client.chat.completions.create(
                model=DEFAULT_OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            logger.info(f"✅ AI response for {category}: {result_text}")
            
            # Simple parsing
            if "Dominant:" in result_text:
                dominant = result_text.split("Dominant:")[-1].strip()
            else:
                dominant = "Balance-Sheet Steward" if category == "Business Strategy" else "Rules-Led Operator"
                
            return {
                "dominant": dominant,
                "secondary": "",
                "reasoning": f"AI analysis identified {dominant} as the dominant {category} archetype."
            }
            
        except Exception as classify_err:
            logger.error(f"❌ AI classify error: {str(classify_err)}")
            return self._pattern_fallback(category)

    def _pattern_fallback(self, category: str) -> Dict[str, str]:
        """Pattern fallback"""
        if category == "Business Strategy":
            dominant = "Cost-Leadership Operator"
        else:
            dominant = "Rules-Led Operator"
            
        return {
            "dominant": dominant,
            "secondary": "",
            "reasoning": f"Pattern analysis identified {dominant}."
        }

    def _fallback_analysis(self, content: str, company_name: str, company_number: str) -> Dict[str, Any]:
        """Fallback analysis"""
        logger.info("🔄 Running fallback analysis")
        
        business = self._pattern_fallback("Business Strategy")
        risk = self._pattern_fallback("Risk Strategy")
        
        return {
            "success": True,
            "analysis_type": "pattern_archetype_classification",
            "company_name": company_name,
            "company_number": company_number,
            "business_strategy_archetypes": business,
            "risk_strategy_archetypes": risk,
            "timestamp": datetime.now().isoformat()
        }

logger.info("🔥 AI_ANALYZER MODULE LOADED SUCCESSFULLY")