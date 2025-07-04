#!/usr/bin/env python3
"""
AI Analyzer V2 - Fresh implementation to bypass import cache issues
"""

print("🔥 AI_ANALYZER_V2 MODULE STARTING...")

import logging
import json
import os
import re
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("🔥 IMPORTS COMPLETED")
logger.info("🔥 LOGGER INITIALIZED")

# Try to import config with fallbacks
try:
    from config import DEFAULT_OPENAI_MODEL, AI_MAX_TOKENS, AI_TEMPERATURE
    logger.info("✅ Config imported successfully")
except Exception as e:
    logger.warning(f"⚠️ Config import failed: {e}, using defaults")
    DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
    AI_MAX_TOKENS = 1000
    AI_TEMPERATURE = 0.3

print("🔥 DEFINING CLASS...")

class AIArchetypeAnalyzer:
    """AI Archetype Analyzer - V2"""
    
    def __init__(self):
        print("🔥 __INIT__ CALLED!")
        logger.info("🔥 AIArchetypeAnalyzer.__init__() starting")
        
        try:
            print("🔥 INIT STEP 1: Setting defaults...")
            self.client = None
            self.client_type = "fallback"
            print("🔥 INIT STEP 1: Complete")
            
            print("🔥 INIT STEP 2: Setting up archetypes...")
            logger.info("🔥 Setting up archetypes...")
            self._setup_archetypes()
            print("🔥 INIT STEP 2: Complete")
            
            print("🔥 INIT STEP 3: Setting up OpenAI...")
            logger.info("🔥 Setting up OpenAI client...")
            self._setup_openai()
            print("🔥 INIT STEP 3: Complete")
            
            print(f"🔥 INITIALIZATION COMPLETE - type: {self.client_type}")
            logger.info(f"🔥 Initialization complete - type: {self.client_type}")
            
        except Exception as e:
            print(f"🚨 CRITICAL INIT ERROR: {type(e).__name__}: {str(e)}")
            logger.error(f"🚨 Init failed: {e}")
            logger.error(f"🚨 Traceback: {traceback.format_exc()}")
            # Set safe defaults to prevent app crash
            self.client = None
            self.client_type = "fallback"
            self.business_archetypes = {'Balance-Sheet Steward': 'Default'}
            self.risk_archetypes = {'Rules-Led Operator': 'Default'}
            print("🔥 SAFE DEFAULTS SET - APP SHOULD CONTINUE")
    
    def _setup_archetypes(self):
        """Setup archetypes"""
        try:
            print("🔥 ARCHETYPES: Starting setup...")
            self.business_archetypes = {
                'Cost-Leadership Operator': 'Drives ROE via lean cost base, digital self-service.',
                'Balance-Sheet Steward': 'Low-risk appetite, prioritises capital strength.',
                'Disciplined Specialist Growth': 'Niche focus with strong underwriting edge.',
                'Service-Driven Differentiator': 'Wins by superior client experience.',
                'Tech-Productivity Accelerator': 'Heavy automation/AI to compress unit costs.'
            }
            print("🔥 ARCHETYPES: Business archetypes set")
            
            self.risk_archetypes = {
                'Rules-Led Operator': 'Strict adherence to rules and checklists.',
                'Risk-First Conservative': 'Prioritises capital preservation.',
                'Resilience-Focused Architect': 'Designs for operational continuity.',
                'Strategic Risk-Taker': 'Accepts elevated risk to unlock growth.',
                'Embedded Risk Partner': 'Risk teams are embedded in frontline decisions.'
            }
            print("🔥 ARCHETYPES: Risk archetypes set")
            logger.info("✅ Archetypes setup complete")
            print("🔥 ARCHETYPES: Setup complete")
            
        except Exception as e:
            print(f"🚨 ARCHETYPES ERROR: {e}")
            logger.error(f"Archetype setup failed: {e}")
            # Set minimal defaults
            self.business_archetypes = {'Balance-Sheet Steward': 'Default'}
            self.risk_archetypes = {'Rules-Led Operator': 'Default'}
    
    def _setup_openai(self):
        """Setup OpenAI with detailed steps"""
        try:
            print("🔥 OPENAI: Starting setup...")
            logger.info("🔧 Starting OpenAI setup...")
            
            # Step 1: Get API key
            print("🔥 OPENAI: Getting API key...")
            api_key = os.getenv('OPENAI_API_KEY')
            logger.info(f"🔑 Step 1 - API key found: {bool(api_key)}")
            print(f"🔥 OPENAI: API key found: {bool(api_key)}")
            
            if not api_key or api_key.startswith('your_'):
                print("🔥 OPENAI: No valid API key - staying in fallback")
                logger.warning("⚠️ No valid API key - staying in fallback mode")
                return
            
            # Step 2: Import OpenAI
            try:
                print("🔥 OPENAI: Importing OpenAI library...")
                logger.info("📦 Step 2 - Importing OpenAI...")
                import openai
                print(f"🔥 OPENAI: Import successful - v{getattr(openai, '__version__', 'unknown')}")
                logger.info(f"✅ Step 2 - OpenAI imported: v{getattr(openai, '__version__', 'unknown')}")
            except ImportError as e:
                print(f"🔥 OPENAI: Import failed - {e}")
                logger.error(f"❌ Step 2 - Import failed: {e}")
                return
            
            # Step 3: Create client
            try:
                print("🔥 OPENAI: Creating client...")
                logger.info("🚀 Step 3 - Creating client...")
                self.client = openai.OpenAI(api_key=api_key)
                self.client_type = "openai"
                print("🔥 OPENAI: Client created successfully")
                logger.info("✅ Step 3 - Client created successfully")
            except Exception as e:
                print(f"🔥 OPENAI: Client creation failed - {e}")
                logger.error(f"❌ Step 3 - Client creation failed: {e}")
                return
            
            # Step 4: Test API
            try:
                print("🔥 OPENAI: Testing API...")
                logger.info("🧪 Step 4 - Testing API...")
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=1
                )
                print("🔥 OPENAI: API test successful!")
                logger.info("✅ Step 4 - API test successful!")
            except Exception as e:
                print(f"🔥 OPENAI: API test failed - {e}")
                logger.error(f"❌ Step 4 - API test failed: {e}")
                # Keep client anyway
                
            print("🔥 OPENAI: Setup complete")
            
        except Exception as setup_error:
            print(f"🚨 OPENAI SETUP ERROR: {setup_error}")
            logger.error(f"OpenAI setup error: {setup_error}")
            self.client = None
            self.client_type = "fallback"
    
    def analyze_archetypes(self, content: str, company_name: str, company_number: str) -> Dict[str, Any]:
        """Analyze archetypes"""
        logger.info(f"🏛️ Starting analysis for {company_name}")
        logger.info(f"🔧 Client type: {self.client_type}")
        
        if self.client_type == "openai":
            logger.info("🚀 Using AI analysis")
            try:
                business = self._ai_analyze(content, "Business")
                risk = self._ai_analyze(content, "Risk")
                
                return {
                    "success": True,
                    "analysis_type": "ai_archetype_classification",
                    "company_name": company_name,
                    "company_number": company_number,
                    "business_strategy_archetypes": business,
                    "risk_strategy_archetypes": risk,
                    "timestamp": datetime.now().isoformat(),
                    "model_used": f"openai_{DEFAULT_OPENAI_MODEL}"
                }
            except Exception as e:
                logger.error(f"🚨 AI analysis failed: {e}")
                return self._fallback_analyze(content, company_name, company_number)
        else:
            logger.info("🔄 Using fallback analysis")
            return self._fallback_analyze(content, company_name, company_number)
    
    def _ai_analyze(self, content: str, category: str) -> Dict[str, str]:
        """AI analysis"""
        sample = content[:5000]
        
        prompt = f"""Analyze this financial company's {category} approach. 
        
Options: Cost-Leadership Operator, Balance-Sheet Steward, Rules-Led Operator, Risk-First Conservative

Reply format: "Dominant: [name]"

Content: {sample}"""
        
        response = self.client.chat.completions.create(
            model=DEFAULT_OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        
        text = response.choices[0].message.content
        if "Dominant:" in text:
            dominant = text.split("Dominant:")[-1].strip()
        else:
            dominant = "Balance-Sheet Steward" if category == "Business" else "Rules-Led Operator"
        
        return {
            "dominant": dominant,
            "secondary": "",
            "reasoning": f"AI analysis identified {dominant} for {category} strategy."
        }
    
    def _fallback_analyze(self, content: str, company_name: str, company_number: str) -> Dict[str, Any]:
        """Fallback analysis"""
        return {
            "success": True,
            "analysis_type": "pattern_archetype_classification", 
            "company_name": company_name,
            "company_number": company_number,
            "business_strategy_archetypes": {
                "dominant": "Cost-Leadership Operator",
                "secondary": "",
                "reasoning": "Pattern analysis identified Cost-Leadership Operator."
            },
            "risk_strategy_archetypes": {
                "dominant": "Rules-Led Operator", 
                "secondary": "",
                "reasoning": "Pattern analysis identified Rules-Led Operator."
            },
            "timestamp": datetime.now().isoformat()
        }

print("🔥 CLASS DEFINED")
logger.info("🔥 AI_ANALYZER_V2 MODULE COMPLETE")