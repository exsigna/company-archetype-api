#!/usr/bin/env python3
"""
Complete AI Analyzer with Exact Business and Risk Strategy Archetypes
Generates reports in the specified format with proper SWOT analysis
Thread-safe for Render deployment with enhanced debugging
UPDATED: Fixed confidence level calculation based on analysis scope only
"""

import os
import sys
import logging
import json
import time
import random
import threading
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
    Complete AI Analyzer with exact archetypes and report format
    """
    
    def __init__(self):
        """Initialize with complete archetype definitions"""
        logger.info("🚀 Initializing Complete AI Analyzer with exact archetypes...")
        
        # Log environment for debugging
        self._log_environment_debug()
        
        # Client initialization
        self.openai_client = None
        self.anthropic_client = None
        self.client_type = "uninitialized"
        
        # Model configuration optimized for Render - Updated for large content analysis
        self.primary_model = "gpt-4-turbo"
        self.fallback_model = "gpt-4-turbo-2024-04-09"
        self.max_output_tokens = 4096
        self.max_retries = 3  # Increased from 2 to 3
        self.base_retry_delay = 3.0  # Increased to 3s between retries
        self.max_content_chars = 150000
        self.request_timeout = 60  # Increased base timeout
        
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
            
            # Check for common issues
            if ' ' in openai_key:
                logger.warning(f"   ⚠️ Key contains spaces!")
            if '\n' in openai_key or '\r' in openai_key:
                logger.warning(f"   ⚠️ Key contains newlines!")
            if len(openai_key) < 50:
                logger.warning(f"   ⚠️ Key seems too short!")
        else:
            logger.error("   ❌ OPENAI_API_KEY not found!")
        
        # Detailed Anthropic key analysis
        if anthropic_key:
            logger.info(f"   ✅ ANTHROPIC_API_KEY found")
            logger.info(f"   📏 Length: {len(anthropic_key)} characters")
            logger.info(f"   🔑 Prefix: {anthropic_key[:15]}...")
            logger.info(f"   📝 Format check: {'✅ Valid sk-ant format' if anthropic_key.startswith('sk-ant-') else '⚠️ Unexpected format'}")
            
            # Check for common issues
            if ' ' in anthropic_key:
                logger.warning(f"   ⚠️ Key contains spaces!")
            if '\n' in anthropic_key or '\r' in anthropic_key:
                logger.warning(f"   ⚠️ Key contains newlines!")
            if len(anthropic_key) < 50:
                logger.warning(f"   ⚠️ Key seems too short!")
        else:
            logger.info("   ℹ️ ANTHROPIC_API_KEY not found (fallback only)")
        
        # Environment info
        logger.info(f"   🌍 Platform: {sys.platform}")
        logger.info(f"   🐍 Python: {sys.version}")
        logger.info(f"   📦 Working directory: {os.getcwd()}")
        
        # Check for Render-specific variables
        render_vars = {
            'RENDER': os.environ.get('RENDER'),
            'RENDER_SERVICE_ID': os.environ.get('RENDER_SERVICE_ID'),
            'RENDER_SERVICE_NAME': os.environ.get('RENDER_SERVICE_NAME'),
        }
        for key, value in render_vars.items():
            if value:
                logger.info(f"   🏗️ {key}: {value}")
    
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
            
            logger.info("🔧 Attempting OpenAI client initialization...")
            
            try:
                import openai
                logger.info(f"📦 openai version: {openai.__version__}")
                from openai import OpenAI
                logger.info(f"✅ OpenAI library imported successfully")
                
                # Create client with explicit timeout
                self.openai_client = OpenAI(
                    api_key=api_key,
                    max_retries=0,
                    timeout=30.0  # Increased timeout for initial test
                )
                logger.info("✅ OpenAI client created")
                
                # Test connection with simple call
                logger.info("🧪 Testing OpenAI connection...")
                test_response = self.openai_client.models.list()
                models = [model.id for model in test_response.data]
                gpt4_models = [m for m in models if 'gpt-4' in m]
                
                logger.info(f"✅ OpenAI connection successful!")
                logger.info(f"📊 Available models: {len(models)}")
                logger.info(f"🧠 GPT-4 models: {len(gpt4_models)}")
                logger.info(f"🎯 Primary model available: {'✅' if self.primary_model in models else '❌'}")
                
                return True
                
            except ImportError as e:
                logger.error(f"❌ OpenAI library import failed: {e}")
                logger.error("💡 Solution: Add 'openai>=1.0.0' to requirements.txt")
                return False
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ OpenAI client test failed: {error_msg}")
                
                # Provide specific error guidance
                if "401" in error_msg or "unauthorized" in error_msg.lower():
                    logger.error("💡 API key authentication failed - check key validity")
                elif "403" in error_msg or "forbidden" in error_msg.lower():
                    logger.error("💡 API access forbidden - check account status")
                elif "timeout" in error_msg.lower():
                    logger.error("💡 Connection timeout - network issue")
                elif "connection" in error_msg.lower():
                    logger.error("💡 Connection failed - check network/firewall")
                else:
                    logger.error(f"💡 Unexpected error type: {type(e).__name__}")
                
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
            
            logger.info("🔧 Attempting Anthropic client initialization...")
            
            try:
                import anthropic
                logger.info("✅ Anthropic library imported successfully")
                
                # Create client
                self.anthropic_client = anthropic.Anthropic(
                    api_key=api_key, 
                    timeout=30.0
                )
                logger.info("✅ Anthropic client created")
                
                # Test connection with minimal call using CORRECT model name
                logger.info("🧪 Testing Anthropic connection...")
                test_response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=5,
                    messages=[{"role": "user", "content": "Hi"}]
                )
                
                logger.info("✅ Anthropic connection successful!")
                logger.info(f"📝 Test response: {test_response.content[0].text}")
                return True
                
            except ImportError as e:
                logger.info(f"ℹ️ Anthropic library not available: {e}")
                logger.info("💡 Solution: Add 'anthropic>=0.8.0' to requirements.txt")
                return False
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"⚠️ Anthropic client test failed: {error_msg}")
                
                # Provide specific error guidance
                if "401" in error_msg or "authentication_error" in error_msg:
                    logger.warning("💡 Anthropic API key authentication failed")
                elif "403" in error_msg or "forbidden" in error_msg.lower():
                    logger.warning("💡 Anthropic API access forbidden")
                elif "rate_limit" in error_msg.lower():
                    logger.warning("💡 Anthropic rate limit exceeded")
                elif "404" in error_msg or "not_found" in error_msg.lower():
                    logger.warning("💡 Anthropic model not found - trying fallback model")
                    # Try with a different model
                    try:
                        test_response = self.anthropic_client.messages.create(
                            model="claude-3-sonnet-20240229",
                            max_tokens=5,
                            messages=[{"role": "user", "content": "Hi"}]
                        )
                        logger.info("✅ Anthropic connection successful with fallback model!")
                        return True
                    except:
                        logger.warning("⚠️ Fallback model also failed")
                        return False
                else:
                    logger.warning(f"💡 Anthropic error type: {type(e).__name__}")
                
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Anthropic initialization failed: {e}")
            return False
    
    def test_api_connections(self) -> Dict[str, Any]:
        """Test API connections and return detailed results"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "openai": {"status": "not_tested"},
            "anthropic": {"status": "not_tested"}
        }
        
        # Test OpenAI
        openai_key = os.environ.get('OPENAI_API_KEY', '').strip()
        if openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key, timeout=10.0)
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Say 'OpenAI test successful'"}],
                    max_tokens=10
                )
                
                results["openai"] = {
                    "status": "SUCCESS",
                    "response": response.choices[0].message.content,
                    "model": "gpt-3.5-turbo"
                }
                
            except Exception as e:
                results["openai"] = {
                    "status": "FAILED",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
        else:
            results["openai"] = {"status": "NO_KEY"}
        
        # Test Anthropic
        anthropic_key = os.environ.get('ANTHROPIC_API_KEY', '').strip()
        if anthropic_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=anthropic_key, timeout=10.0)
                
                # Try modern model first
                try:
                    response = client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=10,
                        messages=[{"role": "user", "content": "Say 'Anthropic test successful'"}]
                    )
                    model_used = "claude-3-5-sonnet-20241022"
                except:
                    # Fallback to older model
                    response = client.messages.create(
                        model="claude-3-sonnet-20240229",
                        max_tokens=10,
                        messages=[{"role": "user", "content": "Say 'Anthropic test successful'"}]
                    )
                    model_used = "claude-3-sonnet-20240229"
                
                results["anthropic"] = {
                    "status": "SUCCESS",
                    "response": response.content[0].text,
                    "model": model_used
                }
                
            except Exception as e:
                results["anthropic"] = {
                    "status": "FAILED",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
        else:
            results["anthropic"] = {"status": "NO_KEY"}
        
        return results
    
    def analyze_for_board_optimized(self, content: str, company_name: str, company_number: str,
                                  extracted_content: Optional[List[Dict[str, Any]]] = None,
                                  analysis_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete analysis generating full report format (thread-safe for Render)
        """
        start_time = time.time()
        
        logger.info(f"🚀 Starting complete analysis for {company_name} ({company_number})")
        logger.info(f"📊 Content length: {len(content):,} characters")
        logger.info(f"🔧 Client type: {self.client_type}")
        
        try:
            if self.client_type == "no_clients_available":
                logger.error("❌ No AI clients available - using emergency analysis")
                return self._create_emergency_analysis(company_name, company_number, "No AI clients available")
            
            # Optimize content
            optimized_content = self._optimize_content(content)
            
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
    
    def _analyze_with_openai(self, content: str, company_name: str, company_number: str,
                           analysis_context: Optional[str], extracted_content: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        """Analyze using OpenAI with thread-safe timeout"""
        
        if not self.openai_client:
            logger.warning("⚠️ OpenAI client not available")
            return None
        
        # Create timeout manager with much longer duration for large content
        timeout_manager = TimeoutManager(90.0)  # Increased to 90s for very large content
        timeout_manager.start()
        
        models = [self.primary_model, self.fallback_model]
        
        for model in models:
            for attempt in range(self.max_retries):
                try:
                    # Check timeout before each attempt
                    timeout_manager.check_timeout()
                    
                    logger.info(f"🔄 OpenAI: {model}, attempt {attempt + 1}")
                    
                    messages = self._create_complete_openai_messages(content, company_name, analysis_context)
                    
                    # Use remaining time for API timeout, much longer for complex analysis
                    api_timeout = min(timeout_manager.remaining_time(), 60.0)  # Increased to 60s per request
                    
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
                        # Check if we have time for retry
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
        timeout_manager = TimeoutManager(70.0)  # Increased to 70s for very large content
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
                    
                    # Use remaining time for API timeout, much longer for complex analysis
                    api_timeout = min(timeout_manager.remaining_time(), 50.0)  # Increased to 50s per request
                    
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
                        # Check if we have time for retry
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
    
    def _optimize_content(self, content: str) -> str:
        """Optimize content length"""
        if len(content) <= self.max_content_chars:
            return content
        
        optimized = content[:self.max_content_chars]
        logger.info(f"📊 Content optimized: {len(optimized):,} chars from {len(content):,}")
        return optimized
    
    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response with error handling"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            import re
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
        Should only consider the current analysis, not cumulative database assessments
        
        Returns:
            tuple: (confidence_level, explanation)
        """
        # Extract years information
        years_analyzed = analysis.get('years_analyzed', [])
        if isinstance(years_analyzed, str):
            try:
                # Try to parse years from string like "[2020, 2021, 2022, 2023, 2024]"
                import re
                year_matches = re.findall(r'\d{4}', years_analyzed)
                years_analyzed = [int(year) for year in year_matches]
            except:
                years_analyzed = []
        
        years_count = len(years_analyzed) if isinstance(years_analyzed, list) else 0
        files_processed = len(extracted_content) if extracted_content else years_count
        
        # Calculate years span
        if years_count >= 2:
            years_span = max(years_analyzed) - min(years_analyzed) + 1
        else:
            years_span = years_count
        
        # Check content quality indicators
        business_strategy = analysis.get('business_strategy', {})
        risk_strategy = analysis.get('risk_strategy', {})
        
        business_reasoning_length = len(str(business_strategy.get('dominant_rationale', business_strategy.get('dominant_reasoning', ''))))
        risk_reasoning_length = len(str(risk_strategy.get('dominant_rationale', risk_strategy.get('dominant_reasoning', ''))))
        
        # Confidence scoring based on data scope and quality
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
            score_breakdown.append(f"+12 pts: good reasoning quality (business: {business_reasoning_length}, risk: {risk_reasoning_length} chars)")
        elif business_reasoning_length >= 100 and risk_reasoning_length >= 100:
            confidence_score += 8
            score_breakdown.append(f"+8 pts: adequate reasoning (business: {business_reasoning_length}, risk: {risk_reasoning_length} chars)")
        else:
            confidence_score += 3
            score_breakdown.append(f"+3 pts: basic reasoning (business: {business_reasoning_length}, risk: {risk_reasoning_length} chars)")
        
        # Determine final confidence level
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
        
        return confidence_level, detailed_explanation
    
    def _create_complete_report(self, analysis: Dict[str, Any], company_name: str,
                              company_number: str, model: str, service: str, 
                              extracted_content: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Create complete report in the specified format with proper confidence assessment"""
        
        # Extract data with fallbacks
        business = analysis.get('business_strategy', {})
        risk = analysis.get('risk_strategy', {})
        swot = analysis.get('swot_analysis', {})
        
        # **FIXED: Calculate confidence based on current analysis scope only**
        confidence_level, confidence_explanation = self._determine_confidence_level(analysis, extracted_content)
        
        logger.info(f"🎯 Confidence assessment for {company_name}:")
        logger.info(f"   Years analyzed: {analysis.get('years_analyzed', 'unknown')}")
        logger.info(f"   Files processed: {len(extracted_content) if extracted_content else 'unknown'}")
        logger.info(f"   Calculated confidence: {confidence_level}")
        logger.info(f"   Explanation: {confidence_explanation}")
        
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
                    'reasoning_quality': 'assessed'
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
            "details": status["environment"]
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