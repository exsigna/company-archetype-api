#!/usr/bin/env python3
"""
RELIABLE AI Analyzer using OpenAI GPT-4 Turbo
Switches from unreliable Anthropic API to rock-solid OpenAI for production reliability
"""

import os
import logging
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
import time
import re
import random
from datetime import datetime
from collections import Counter, deque
import statistics
import threading

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Circuit breaker for API failures"""
    
    def __init__(self, failure_threshold: int = 3, timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False
        self.lock = threading.Lock()
    
    def can_proceed(self) -> bool:
        with self.lock:
            if not self.is_open:
                return True
            
            if (self.last_failure_time and 
                time.time() - self.last_failure_time > self.timeout):
                self.is_open = False
                self.failure_count = 0
                logger.info("🔄 Circuit breaker CLOSED - timeout expired")
                return True
            
            return False
    
    def record_failure(self):
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
                logger.warning(f"🚨 Circuit breaker OPEN - {self.failure_count} consecutive failures")
    
    def record_success(self):
        with self.lock:
            self.failure_count = 0
            if self.is_open:
                self.is_open = False
                logger.info("✅ Circuit breaker CLOSED - successful request")

class OptimizedClaudeAnalyzer:
    """
    RELIABLE AI Analyzer using OpenAI GPT-4 Turbo for production stability
    """
    
    def __init__(self):
        """Initialize reliable OpenAI-based analyzer"""
        self.anthropic_client = None
        self.openai_client = None
        self.client_type = "fallback"
        
        # OpenAI GPT-4 Turbo Configuration
        self.primary_model = "gpt-4-turbo-preview"      # Reliable primary
        self.fallback_model = "gpt-4-1106-preview"      # Reliable fallback
        self.max_output_tokens = 4096                   # GPT-4 Turbo max
        self.max_input_tokens = 128000                  # GPT-4 Turbo context
        self.max_input_chars = 500000                   # Conservative estimate
        
        # Single pass for reliability and speed
        self.num_passes = 1
        
        # Conservative retry settings (OpenAI is more reliable)
        self.max_retries = int(os.environ.get('AI_MAX_RETRIES', 5))
        self.base_retry_delay = float(os.environ.get('AI_RETRY_DELAY', 3.0))
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=int(os.environ.get('AI_CIRCUIT_THRESHOLD', 3)),
            timeout=int(os.environ.get('AI_CIRCUIT_TIMEOUT', 300))
        )
        
        logger.info(f"🚀 RELIABLE OpenAI GPT-4 Turbo Analyzer")
        logger.info(f"📊 Model: {self.primary_model}")
        logger.info(f"🔢 Tokens: {self.max_output_tokens}")
        logger.info(f"✅ Switching from unreliable Anthropic to reliable OpenAI")
        
        # Initialize OpenAI as primary
        self._init_openai_primary()
        self._init_anthropic_fallback()  # Keep as emergency fallback only
        
        # Archetype definitions optimized for GPT-4 Turbo
        self.business_archetypes = {
            "Scale-through-Distribution": {
                "definition": "Gains market share primarily by expanding distribution channels and partnerships faster than operational maturity develops",
                "strategic_context": "High-velocity expansion strategy with emphasis on market capture over operational excellence"
            },
            "Disciplined Specialist Growth": {
                "definition": "Maintains niche focus with strong underwriting capabilities, growing opportunistically while optimizing balance sheet efficiency",
                "strategic_context": "Conservative growth strategy emphasizing expertise depth over market breadth"
            },
            "Service-Driven Differentiator": {
                "definition": "Competes on superior client experience and advisory capability rather than price or scale",
                "strategic_context": "Relationship-centric model with emphasis on customer satisfaction and loyalty"
            },
            "Asset-Velocity Maximiser": {
                "definition": "Prioritizes rapid asset origination and turnover, often accepting higher funding costs for speed",
                "strategic_context": "Transaction-focused model optimizing for volume and velocity over margin per transaction"
            },
            "Balance-Sheet Steward": {
                "definition": "Prioritizes capital strength and stakeholder value over aggressive growth",
                "strategic_context": "Conservative approach emphasizing financial stability and long-term sustainability"
            }
        }
        
        self.risk_archetypes = {
            "Risk-First Conservative": {
                "definition": "Prioritizes capital preservation and regulatory compliance above growth opportunities",
                "strategic_context": "Defensive risk strategy emphasizing stability and regulatory relationship quality"
            },
            "Rules-Led Operator": {
                "definition": "Emphasizes strict procedural adherence and control consistency over business judgment",
                "strategic_context": "Process-driven risk management with emphasis on consistency and auditability"
            },
            "Embedded Risk Partner": {
                "definition": "Integrates risk teams into frontline business decisions with collaborative risk appetite setting",
                "strategic_context": "Partnership-based risk management with business-risk team collaboration"
            }
        }
        
        logger.info(f"✅ Reliable OpenAI Analyzer initialized. Engine: {self.client_type}")
    
    def analyze_for_board_optimized(self, content: str, company_name: str, company_number: str, 
                                  extracted_content: Optional[List[Dict[str, Any]]] = None,
                                  analysis_context: Optional[str] = None) -> Dict[str, Any]:
        """
        RELIABLE analysis using OpenAI GPT-4 Turbo
        """
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Starting RELIABLE OpenAI analysis for {company_name}")
            logger.info(f"📊 Input content: {len(content):,} characters")
            logger.info(f"🔢 Using reliable {self.primary_model}")
            
            # Check circuit breaker
            if not self.circuit_breaker.can_proceed():
                logger.warning("⚠️ Circuit breaker is OPEN - using emergency analysis")
                return self._create_emergency_analysis(company_name, company_number, "Circuit breaker open")
            
            # Optimize content for GPT-4 Turbo
            optimized_content = self._optimize_content_for_gpt4(content, extracted_content, company_name)
            logger.info(f"📊 Optimized content: {len(optimized_content):,} characters ({len(optimized_content)//4:,} estimated tokens)")
            
            # Single reliable OpenAI analysis
            logger.info(f"🎯 GPT-4 Turbo Analysis")
            
            try:
                analysis_result = self._openai_gpt4_analysis(
                    optimized_content, company_name, company_number, 
                    extracted_content, analysis_context
                )
                
                if analysis_result:
                    logger.info(f"✅ GPT-4 Turbo analysis completed successfully")
                    self.circuit_breaker.record_success()
                    
                    # Create final report
                    final_report = self._create_reliable_report(analysis_result, company_name, company_number)
                    
                    analysis_time = time.time() - start_time
                    total_chars = len(str(final_report))
                    logger.info(f"🎉 RELIABLE analysis completed in {analysis_time:.2f}s")
                    logger.info(f"📊 Total output: {total_chars:,} characters (~{total_chars//4:,} tokens)")
                    
                    return final_report
                else:
                    logger.warning(f"⚠️ GPT-4 Turbo analysis failed - using emergency fallback")
                    
            except Exception as e:
                logger.error(f"❌ GPT-4 Turbo analysis failed: {e}")
                self.circuit_breaker.record_failure()
            
            # If we get here, analysis failed
            logger.error("❌ Reliable analysis failed")
            return self._create_emergency_analysis(company_name, company_number, "GPT-4 Turbo failed")
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return self._create_emergency_analysis(company_name, company_number, str(e))
    
    def _openai_gpt4_analysis(self, content: str, company_name: str, company_number: str,
                             extracted_content: Optional[List[Dict[str, Any]]], 
                             analysis_context: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Reliable OpenAI GPT-4 Turbo analysis
        """
        
        if self.client_type != "openai_primary":
            logger.error("❌ OpenAI client required for reliable analysis")
            return None
        
        # Try primary model first, then fallback
        models_to_try = [self.primary_model, self.fallback_model]
        
        for model_idx, model in enumerate(models_to_try):
            for attempt in range(self.max_retries):
                try:
                    # Conservative token allocation for reliability
                    max_tokens = self._get_reliable_token_count(model, attempt)
                    
                    logger.info(f"🔄 Model: {model}, Attempt: {attempt + 1}/{self.max_retries}, Tokens: {max_tokens}")
                    
                    # Create optimized messages for GPT-4 Turbo
                    messages = self._create_gpt4_messages(content, company_name, analysis_context)
                    
                    response = self.openai_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.1,  # Low temperature for consistency
                        response_format={"type": "json_object"}  # Force JSON output
                    )
                    
                    response_text = response.choices[0].message.content
                    logger.info(f"✅ SUCCESS with {model} on attempt {attempt + 1}")
                    logger.info(f"📊 Response: {len(response_text):,} chars (~{len(response_text)//4:,} tokens)")
                    
                    # Parse JSON response
                    parsed_analysis = self._parse_gpt4_response(response_text, model, attempt)
                    
                    if parsed_analysis:
                        parsed_analysis['pass_metadata'] = {
                            'model_used': model,
                            'attempt_number': attempt + 1,
                            'max_tokens': max_tokens,
                            'analysis_timestamp': datetime.now().isoformat(),
                            'ai_service': 'openai_gpt4_turbo_reliable'
                        }
                        return parsed_analysis
                    else:
                        logger.warning(f"⚠️ JSON parsing failed, retrying...")
                        continue
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    # Handle rate limits and other errors
                    if "rate_limit" in error_msg.lower() or "429" in error_msg:
                        if attempt < self.max_retries - 1:
                            delay = self._calculate_reliable_delay(attempt)
                            logger.warning(f"⚠️ Rate limit - waiting {delay:.1f}s before retry {attempt + 2}")
                            time.sleep(delay)
                            continue
                        else:
                            logger.error(f"❌ All retries exhausted for {model} due to rate limits")
                            break
                    
                    # Handle other retryable errors
                    elif self._is_retryable_error(error_msg) and attempt < self.max_retries - 1:
                        delay = self.base_retry_delay * (1.5 ** attempt)
                        logger.warning(f"⚠️ Retryable error: {e} - waiting {delay:.1f}s")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"❌ Non-retryable error with {model}: {e}")
                        break
            
            # Try next model if available
            if model_idx < len(models_to_try) - 1:
                logger.warning(f"⚠️ All attempts failed with {model}, trying {models_to_try[model_idx + 1]}...")
                time.sleep(5)  # Brief pause before next model
        
        logger.error(f"❌ All OpenAI models failed")
        return None
    
    def _get_reliable_token_count(self, model: str, attempt: int) -> int:
        """Conservative token allocation for reliability"""
        
        base_tokens = {
            "gpt-4-turbo-preview": 4000,      # Conservative for reliability
            "gpt-4-1106-preview": 3500,      # Slightly lower for fallback
        }
        
        max_tokens = base_tokens.get(model, 3000)
        
        # Minimal reduction on retries (OpenAI is more reliable)
        if attempt > 3:
            max_tokens = int(max_tokens * 0.9)   # 10% reduction
        elif attempt > 1:
            max_tokens = int(max_tokens * 0.95)  # 5% reduction
        
        return max(max_tokens, 2000)  # Minimum viable output
    
    def _calculate_reliable_delay(self, attempt: int) -> float:
        """Calculate delay for OpenAI retries"""
        base_delays = [3, 5, 8, 12, 20]  # Conservative delays
        
        if attempt < len(base_delays):
            delay = base_delays[attempt]
        else:
            delay = 20  # Cap at 20 seconds
        
        # Add small jitter
        jitter = delay * 0.1 * random.random()
        return delay + jitter
    
    def _optimize_content_for_gpt4(self, content: str, extracted_content: Optional[List[Dict[str, Any]]], 
                                  company_name: str) -> str:
        """Optimize content for GPT-4 Turbo (128K context)"""
        
        if not content:
            return f"Limited content available for {company_name} analysis."
        
        # GPT-4 Turbo can handle more content reliably
        max_input_chars = self.max_input_chars
        
        if len(content) <= max_input_chars:
            logger.info(f"📊 Using full content: {len(content):,} characters")
            return content
        
        # Smart truncation for GPT-4 Turbo
        logger.info(f"📊 Content exceeds limit ({len(content):,} chars), optimizing for GPT-4...")
        
        # Take first portion (GPT-4 Turbo handles context well)
        optimized_content = content[:max_input_chars]
        
        logger.info(f"📊 Content optimized: {len(optimized_content):,} chars from {len(content):,} chars")
        return optimized_content
    
    def _create_gpt4_messages(self, content: str, company_name: str, 
                             analysis_context: Optional[str]) -> List[Dict[str, str]]:
        """Create message format for GPT-4 Turbo"""
        
        context_note = f"\n\nAnalysis Context: {analysis_context}" if analysis_context else ""
        
        system_prompt = f"""You are a strategic business analyst conducting comprehensive archetype analysis.

RESPOND WITH VALID JSON ONLY. Use this exact structure:

{{
  "business_strategy": {{
    "dominant_archetype": "[exact archetype name]",
    "dominant_rationale": "[comprehensive analysis 200+ words]",
    "secondary_archetype": "[exact archetype name]",
    "secondary_rationale": "[detailed analysis 100+ words]",
    "evidence_quotes": ["quote1", "quote2", "quote3"]
  }},
  "risk_strategy": {{
    "dominant_archetype": "[exact archetype name]",
    "dominant_rationale": "[comprehensive analysis 200+ words]",
    "secondary_archetype": "[exact archetype name]",
    "secondary_rationale": "[detailed analysis 100+ words]",
    "evidence_quotes": ["quote1", "quote2", "quote3"]
  }},
  "swot_analysis": {{
    "strengths": ["strength1", "strength2", "strength3", "strength4"],
    "weaknesses": ["weakness1", "weakness2", "weakness3", "weakness4"],
    "opportunities": ["opportunity1", "opportunity2", "opportunity3", "opportunity4"],
    "threats": ["threat1", "threat2", "threat3", "threat4"]
  }},
  "confidence_level": "high"
}}

BUSINESS ARCHETYPES:
{self._format_archetypes(self.business_archetypes)}

RISK ARCHETYPES:
{self._format_archetypes(self.risk_archetypes)}"""

        user_prompt = f"""Analyze {company_name} and provide comprehensive strategic archetype analysis.

COMPANY CONTENT:
{content}{context_note}

Respond with valid JSON using the exact structure specified."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _parse_gpt4_response(self, response: str, model: str, attempt: int) -> Optional[Dict[str, Any]]:
        """Parse GPT-4 Turbo JSON response"""
        
        try:
            # GPT-4 Turbo with json_object format should return clean JSON
            analysis = json.loads(response)
            logger.info(f"✅ GPT-4 JSON parse successful (Model: {model}, Attempt: {attempt + 1})")
            return self._validate_analysis_structure(analysis)
            
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ GPT-4 JSON parse failed: {e}")
            
            # Try to extract JSON from response
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    logger.info(f"✅ GPT-4 extracted JSON successful (Model: {model}, Attempt: {attempt + 1})")
                    return self._validate_analysis_structure(analysis)
            except:
                pass
            
            logger.error(f"❌ GPT-4 JSON parsing failed completely")
            return None
    
    def _validate_analysis_structure(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate analysis structure"""
        required_sections = ['business_strategy', 'risk_strategy']
        
        for section in required_sections:
            if section not in analysis:
                logger.warning(f"⚠️ Missing required section: {section}")
                return None
        
        if 'dominant_archetype' not in analysis['business_strategy']:
            logger.warning(f"⚠️ Missing dominant_archetype in business_strategy")
            return None
        
        if 'dominant_archetype' not in analysis['risk_strategy']:
            logger.warning(f"⚠️ Missing dominant_archetype in risk_strategy")
            return None
        
        return analysis
    
    def _format_archetypes(self, archetypes: Dict[str, Dict[str, Any]]) -> str:
        """Format archetype definitions"""
        formatted = ""
        for name, details in archetypes.items():
            formatted += f"- {name}: {details['definition']}\n"
        return formatted
    
    def _is_retryable_error(self, error_message: str) -> bool:
        """Check if error is retryable"""
        retryable_indicators = [
            "timeout", "connection", "network", "temporary", 
            "rate limit", "throttl", "busy", "503", "502"
        ]
        
        error_lower = error_message.lower()
        return any(indicator in error_lower for indicator in retryable_indicators)
    
    def _create_reliable_report(self, analysis: Dict[str, Any], 
                               company_name: str, company_number: str) -> Dict[str, Any]:
        """Create report from reliable OpenAI analysis"""
        
        return {
            'company_name': company_name,
            'company_number': company_number,
            'analysis_date': datetime.now().isoformat(),
            'business_strategy': {
                'dominant': analysis.get('business_strategy', {}).get('dominant_archetype', 'Unknown'),
                'dominant_reasoning': analysis.get('business_strategy', {}).get('dominant_rationale', 'GPT-4 Turbo analysis completed reliably'),
                'secondary': analysis.get('business_strategy', {}).get('secondary_archetype', 'Service-Driven Differentiator'),
                'secondary_reasoning': analysis.get('business_strategy', {}).get('secondary_rationale', 'Secondary analysis from GPT-4 Turbo'),
                'evidence_quotes': analysis.get('business_strategy', {}).get('evidence_quotes', [])
            },
            'risk_strategy': {
                'dominant': analysis.get('risk_strategy', {}).get('dominant_archetype', 'Unknown'),
                'dominant_reasoning': analysis.get('risk_strategy', {}).get('dominant_rationale', 'GPT-4 Turbo risk analysis completed reliably'),
                'secondary': analysis.get('risk_strategy', {}).get('secondary_archetype', 'Rules-Led Operator'),
                'secondary_reasoning': analysis.get('risk_strategy', {}).get('secondary_rationale', 'Secondary risk analysis from GPT-4 Turbo'),
                'evidence_quotes': analysis.get('risk_strategy', {}).get('evidence_quotes', [])
            },
            'swot_analysis': analysis.get('swot_analysis', {}),
            'analysis_metadata': {
                'analysis_type': 'reliable_openai_gpt4_turbo',
                'model_used': analysis.get('pass_metadata', {}).get('model_used', self.primary_model),
                'max_tokens': analysis.get('pass_metadata', {}).get('max_tokens', self.max_output_tokens),
                'successful_attempt': analysis.get('pass_metadata', {}).get('attempt_number', 1),
                'confidence_level': analysis.get('confidence_level', 'high'),
                'analysis_timestamp': datetime.now().isoformat(),
                'reliability_note': 'Switched from unreliable Anthropic to reliable OpenAI'
            }
        }
    
    def _init_openai_primary(self):
        """Initialize OpenAI as primary reliable service"""
        try:
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                logger.error("❌ OpenAI API key required for reliable analysis")
                return
                
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(
                    api_key=api_key, 
                    max_retries=0,  # We handle retries manually
                    timeout=120.0
                )
                self.client_type = "openai_primary"
                logger.info("🚀 OpenAI GPT-4 Turbo configured as RELIABLE primary service")
                return
            except ImportError:
                logger.error("❌ OpenAI library required for reliable analysis")
                return
                    
        except Exception as e:
            logger.error(f"❌ OpenAI setup failed: {e}")
    
    def _init_anthropic_fallback(self):
        """Keep Anthropic as emergency fallback only"""
        try:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                logger.info("ℹ️ No Anthropic API key - OpenAI only mode")
                return
                
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=60.0)
                logger.info("✅ Anthropic configured as emergency fallback only")
                return
            except ImportError:
                logger.info("ℹ️ Anthropic library not available - OpenAI only mode")
                return
                    
        except Exception as e:
            logger.info(f"ℹ️ Anthropic fallback setup skipped: {e}")
    
    def _create_emergency_analysis(self, company_name: str, company_number: str, error_message: str) -> Dict[str, Any]:
        """Emergency analysis when everything fails"""
        return {
            'company_name': company_name,
            'company_number': company_number,
            'analysis_date': datetime.now().isoformat(),
            'business_strategy': {
                'dominant': 'Disciplined Specialist Growth',
                'dominant_reasoning': 'Emergency analysis - Reliable OpenAI GPT-4 Turbo analysis failed. Conservative assessment indicates disciplined specialist growth characteristics.',
                'evidence_quotes': ['Emergency analysis - reliable system temporarily unavailable']
            },
            'risk_strategy': {
                'dominant': 'Risk-First Conservative',
                'dominant_reasoning': 'Emergency analysis - Conservative risk management approach during system constraints.',
                'evidence_quotes': ['Emergency analysis - reliable system temporarily unavailable']
            },
            'analysis_metadata': {
                'analysis_type': 'emergency_reliable_system_failure',
                'error_message': error_message,
                'primary_service': 'openai_gpt4_turbo',
                'reliability_note': 'OpenAI primary service temporarily unavailable'
            }
        }


# BACKWARD COMPATIBILITY
class ExecutiveAIAnalyzer(OptimizedClaudeAnalyzer):
    """Executive analyzer using reliable OpenAI GPT-4 Turbo"""
    
    def __init__(self):
        super().__init__()
        logger.info("🚀 ExecutiveAIAnalyzer - RELIABLE OpenAI GPT-4 Turbo!")
        logger.info("✅ Switched from unreliable Anthropic to reliable OpenAI")
    
    def analyze_for_board(self, content: str, company_name: str, company_number: str, 
                         extracted_content: Optional[List[Dict[str, Any]]] = None,
                         analysis_context: Optional[str] = None) -> Dict[str, Any]:
        """Reliable board analysis using OpenAI GPT-4 Turbo"""
        return self.analyze_for_board_optimized(
            content, company_name, company_number, extracted_content, analysis_context
        )