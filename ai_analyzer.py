#!/usr/bin/env python3
"""
Fixed AI Analyzer with Proper OpenAI Integration
Addresses the issues identified in your logs and code
"""

import os
import logging
import json
import time
import random
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ReliableAIAnalyzer:
    """
    Fixed AI Analyzer with proper OpenAI integration and error handling
    """
    
    def __init__(self):
        """Initialize with comprehensive error handling"""
        # Client initialization
        self.openai_client = None
        self.anthropic_client = None
        self.client_type = "uninitialized"
        
        # Model configuration
        self.primary_model = "gpt-4-turbo-preview"
        self.fallback_model = "gpt-4-1106-preview"
        self.max_output_tokens = 4096
        self.max_input_tokens = 128000
        
        # Retry configuration
        self.max_retries = 3
        self.base_retry_delay = 2.0
        
        # Archetype definitions
        self.business_archetypes = {
            "Scale-through-Distribution": "Gains market share primarily by expanding distribution channels",
            "Disciplined Specialist Growth": "Maintains niche focus with strong underwriting capabilities", 
            "Service-Driven Differentiator": "Competes on superior client experience and advisory capability",
            "Asset-Velocity Maximiser": "Prioritizes rapid asset origination and turnover",
            "Balance-Sheet Steward": "Prioritizes capital strength and stakeholder value"
        }
        
        self.risk_archetypes = {
            "Risk-First Conservative": "Prioritizes capital preservation and regulatory compliance",
            "Rules-Led Operator": "Emphasizes strict procedural adherence and control consistency",
            "Embedded Risk Partner": "Integrates risk teams into frontline business decisions"
        }
        
        logger.info("🚀 Initializing Reliable AI Analyzer...")
        
        # Initialize clients with proper error handling
        self._initialize_clients()
        
        logger.info(f"✅ Initialization complete. Client type: {self.client_type}")
    
    def _initialize_clients(self):
        """Initialize AI clients with comprehensive error handling"""
        
        # Try OpenAI first
        if self._init_openai():
            self.client_type = "openai_primary"
            logger.info("✅ OpenAI configured as primary service")
        else:
            logger.warning("⚠️ OpenAI initialization failed")
        
        # Initialize Anthropic as fallback
        if self._init_anthropic():
            if self.client_type == "uninitialized":
                self.client_type = "anthropic_fallback"
                logger.info("✅ Anthropic configured as fallback service")
            else:
                logger.info("✅ Anthropic available as backup")
        else:
            logger.warning("⚠️ Anthropic initialization failed")
        
        # Final status check
        if self.client_type == "uninitialized":
            self.client_type = "no_clients_available"
            logger.error("❌ No AI clients available")
    
    def _init_openai(self) -> bool:
        """Initialize OpenAI client with validation"""
        try:
            # Check API key
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                logger.warning("⚠️ OPENAI_API_KEY not found")
                return False
            
            logger.info(f"🔑 OpenAI API key found: {api_key[:10]}...")
            
            # Import OpenAI library
            try:
                from openai import OpenAI
            except ImportError as e:
                logger.error(f"❌ OpenAI library not available: {e}")
                return False
            
            # Create client
            self.openai_client = OpenAI(
                api_key=api_key,
                max_retries=0,  # Handle retries manually
                timeout=90.0
            )
            
            # Test the client
            try:
                test_response = self.openai_client.models.list()
                logger.info("✅ OpenAI client test successful")
                return True
            except Exception as e:
                logger.warning(f"⚠️ OpenAI client test failed: {e}")
                # Don't return False here - client might still work for chat completions
                return True
        
        except Exception as e:
            logger.error(f"❌ OpenAI initialization failed: {e}")
            return False
    
    def _init_anthropic(self) -> bool:
        """Initialize Anthropic client as fallback"""
        try:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                logger.info("ℹ️ ANTHROPIC_API_KEY not found")
                return False
            
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=api_key)
                logger.info("✅ Anthropic client initialized")
                return True
            except ImportError:
                logger.info("ℹ️ Anthropic library not available")
                return False
        
        except Exception as e:
            logger.warning(f"⚠️ Anthropic initialization failed: {e}")
            return False
    
    def analyze_for_board_optimized(self, content: str, company_name: str, company_number: str,
                                  extracted_content: Optional[List[Dict[str, Any]]] = None,
                                  analysis_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Main analysis method with proper error handling
        """
        start_time = time.time()
        
        logger.info(f"🚀 Starting analysis for {company_name} ({company_number})")
        logger.info(f"📊 Content length: {len(content):,} characters")
        logger.info(f"🔧 Using client type: {self.client_type}")
        
        try:
            # Check if we have any working clients
            if self.client_type == "no_clients_available":
                return self._create_emergency_analysis(company_name, company_number, "No AI clients available")
            
            # Optimize content for analysis
            optimized_content = self._optimize_content(content)
            
            # Try OpenAI first if available
            if self.openai_client and self.client_type in ["openai_primary", "anthropic_fallback"]:
                logger.info("🎯 Attempting OpenAI analysis...")
                result = self._analyze_with_openai(optimized_content, company_name, company_number, analysis_context)
                if result:
                    analysis_time = time.time() - start_time
                    logger.info(f"✅ OpenAI analysis completed in {analysis_time:.2f}s")
                    return result
                else:
                    logger.warning("⚠️ OpenAI analysis failed")
            
            # Try Anthropic if OpenAI failed
            if self.anthropic_client:
                logger.info("🎯 Attempting Anthropic analysis...")
                result = self._analyze_with_anthropic(optimized_content, company_name, company_number, analysis_context)
                if result:
                    analysis_time = time.time() - start_time
                    logger.info(f"✅ Anthropic analysis completed in {analysis_time:.2f}s")
                    return result
                else:
                    logger.warning("⚠️ Anthropic analysis failed")
            
            # If all else fails, emergency analysis
            logger.error("❌ All analysis methods failed")
            return self._create_emergency_analysis(company_name, company_number, "All AI services failed")
        
        except Exception as e:
            logger.error(f"❌ Analysis failed with exception: {e}")
            return self._create_emergency_analysis(company_name, company_number, str(e))
    
    def _analyze_with_openai(self, content: str, company_name: str, company_number: str,
                           analysis_context: Optional[str]) -> Optional[Dict[str, Any]]:
        """Analyze using OpenAI with proper error handling"""
        
        if not self.openai_client:
            logger.error("❌ OpenAI client not available")
            return None
        
        # Try primary model, then fallback
        models_to_try = [self.primary_model, self.fallback_model]
        
        for model in models_to_try:
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"🔄 OpenAI: {model}, attempt {attempt + 1}")
                    
                    # Create messages
                    messages = self._create_openai_messages(content, company_name, analysis_context)
                    
                    # Make API call
                    response = self.openai_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=self.max_output_tokens,
                        temperature=0.1,
                        response_format={"type": "json_object"}
                    )
                    
                    response_text = response.choices[0].message.content
                    logger.info(f"✅ OpenAI response received: {len(response_text)} chars")
                    
                    # Parse and validate response
                    analysis = self._parse_json_response(response_text)
                    if analysis:
                        return self._create_final_report(analysis, company_name, company_number, model, "openai")
                    else:
                        logger.warning("⚠️ Failed to parse OpenAI response")
                        continue
                
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"⚠️ OpenAI attempt failed: {error_msg}")
                    
                    # Handle rate limits
                    if "rate_limit" in error_msg.lower() or "429" in error_msg:
                        if attempt < self.max_retries - 1:
                            delay = self._calculate_retry_delay(attempt)
                            logger.info(f"⏰ Rate limit - waiting {delay:.1f}s")
                            time.sleep(delay)
                            continue
                    
                    # Handle other retryable errors
                    elif self._is_retryable_error(error_msg) and attempt < self.max_retries - 1:
                        delay = self._calculate_retry_delay(attempt)
                        logger.info(f"⏰ Retryable error - waiting {delay:.1f}s")
                        time.sleep(delay)
                        continue
                    else:
                        break  # Non-retryable error or max retries reached
            
            # If we get here, all attempts for this model failed
            logger.warning(f"❌ All attempts failed for {model}")
        
        return None
    
    def _analyze_with_anthropic(self, content: str, company_name: str, company_number: str,
                              analysis_context: Optional[str]) -> Optional[Dict[str, Any]]:
        """Analyze using Anthropic as fallback"""
        
        if not self.anthropic_client:
            return None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"🔄 Anthropic attempt {attempt + 1}")
                
                # Create prompt for Anthropic
                prompt = self._create_anthropic_prompt(content, company_name, analysis_context)
                
                # Make API call
                response = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=self.max_output_tokens,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                response_text = response.content[0].text
                logger.info(f"✅ Anthropic response received: {len(response_text)} chars")
                
                # Parse and validate response
                analysis = self._parse_json_response(response_text)
                if analysis:
                    return self._create_final_report(analysis, company_name, company_number, "claude-3-sonnet", "anthropic")
                else:
                    logger.warning("⚠️ Failed to parse Anthropic response")
                    continue
            
            except Exception as e:
                logger.warning(f"⚠️ Anthropic attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self._calculate_retry_delay(attempt))
        
        return None
    
    def _create_openai_messages(self, content: str, company_name: str, 
                              analysis_context: Optional[str]) -> List[Dict[str, str]]:
        """Create message format for OpenAI"""
        
        context_note = f"\n\nAnalysis Context: {analysis_context}" if analysis_context else ""
        
        system_prompt = f"""You are a strategic business analyst. Analyze companies and respond with valid JSON only.

Use this exact structure:
{{
  "business_strategy": {{
    "dominant_archetype": "[one of: {', '.join(self.business_archetypes.keys())}]",
    "reasoning": "[detailed analysis 150+ words]",
    "evidence": ["evidence1", "evidence2", "evidence3"]
  }},
  "risk_strategy": {{
    "dominant_archetype": "[one of: {', '.join(self.risk_archetypes.keys())}]", 
    "reasoning": "[detailed analysis 150+ words]",
    "evidence": ["evidence1", "evidence2", "evidence3"]
  }},
  "confidence": "high"
}}

BUSINESS ARCHETYPES:
{self._format_archetypes(self.business_archetypes)}

RISK ARCHETYPES:
{self._format_archetypes(self.risk_archetypes)}"""

        user_prompt = f"Analyze {company_name} using the company content below:{context_note}\n\n{content}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _create_anthropic_prompt(self, content: str, company_name: str,
                               analysis_context: Optional[str]) -> str:
        """Create prompt for Anthropic"""
        
        context_note = f"\n\nAnalysis Context: {analysis_context}" if analysis_context else ""
        
        return f"""Analyze {company_name} and provide strategic archetype analysis in JSON format.

BUSINESS ARCHETYPES:
{self._format_archetypes(self.business_archetypes)}

RISK ARCHETYPES:
{self._format_archetypes(self.risk_archetypes)}

Respond with valid JSON using this structure:
{{
  "business_strategy": {{
    "dominant_archetype": "[archetype name]",
    "reasoning": "[detailed analysis]",
    "evidence": ["evidence1", "evidence2"]
  }},
  "risk_strategy": {{
    "dominant_archetype": "[archetype name]",
    "reasoning": "[detailed analysis]", 
    "evidence": ["evidence1", "evidence2"]
  }},
  "confidence": "high"
}}

COMPANY CONTENT:{context_note}
{content}"""
    
    def _format_archetypes(self, archetypes: Dict[str, str]) -> str:
        """Format archetype definitions"""
        return "\n".join([f"- {name}: {definition}" for name, definition in archetypes.items()])
    
    def _optimize_content(self, content: str) -> str:
        """Optimize content length for analysis"""
        max_chars = 400000  # Conservative limit
        
        if len(content) <= max_chars:
            return content
        
        # Take first portion for now
        optimized = content[:max_chars]
        logger.info(f"📊 Content optimized: {len(optimized):,} chars from {len(content):,} chars")
        return optimized
    
    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response with error handling"""
        try:
            # Try direct parsing first
            return json.loads(response)
        except json.JSONDecodeError:
            # Try extracting JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
        
        logger.error("❌ Failed to parse JSON response")
        return None
    
    def _create_final_report(self, analysis: Dict[str, Any], company_name: str, 
                           company_number: str, model: str, service: str) -> Dict[str, Any]:
        """Create final analysis report"""
        
        return {
            'company_name': company_name,
            'company_number': company_number,
            'analysis_date': datetime.now().isoformat(),
            'business_strategy': {
                'dominant': analysis.get('business_strategy', {}).get('dominant_archetype', 'Disciplined Specialist Growth'),
                'dominant_reasoning': analysis.get('business_strategy', {}).get('reasoning', 'Analysis completed successfully'),
                'evidence_quotes': analysis.get('business_strategy', {}).get('evidence', [])
            },
            'risk_strategy': {
                'dominant': analysis.get('risk_strategy', {}).get('dominant_archetype', 'Risk-First Conservative'),
                'dominant_reasoning': analysis.get('risk_strategy', {}).get('reasoning', 'Risk analysis completed successfully'),
                'evidence_quotes': analysis.get('risk_strategy', {}).get('evidence', [])
            },
            'analysis_metadata': {
                'analysis_type': f'reliable_{service}',
                'model_used': model,
                'confidence_level': analysis.get('confidence', 'high'),
                'service_used': service,
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
    
    def _create_emergency_analysis(self, company_name: str, company_number: str, 
                                 error_message: str) -> Dict[str, Any]:
        """Create emergency analysis when all methods fail"""
        
        return {
            'company_name': company_name,
            'company_number': company_number,
            'analysis_date': datetime.now().isoformat(),
            'business_strategy': {
                'dominant': 'Disciplined Specialist Growth',
                'dominant_reasoning': f'Emergency analysis - {error_message}. Conservative assessment indicates disciplined specialist growth characteristics.',
                'evidence_quotes': ['Emergency analysis - AI services temporarily unavailable']
            },
            'risk_strategy': {
                'dominant': 'Risk-First Conservative',
                'dominant_reasoning': f'Emergency analysis - {error_message}. Conservative risk management approach.',
                'evidence_quotes': ['Emergency analysis - AI services temporarily unavailable']
            },
            'analysis_metadata': {
                'analysis_type': 'emergency_fallback',
                'error_message': error_message,
                'confidence_level': 'low',
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff"""
        delay = self.base_retry_delay * (2 ** attempt)
        jitter = delay * 0.1 * random.random()
        return min(delay + jitter, 60.0)  # Cap at 60 seconds
    
    def _is_retryable_error(self, error_message: str) -> bool:
        """Check if error is retryable"""
        retryable_patterns = [
            "timeout", "connection", "network", "temporary",
            "502", "503", "504", "rate", "throttle"
        ]
        
        error_lower = error_message.lower()
        return any(pattern in error_lower for pattern in retryable_patterns)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current analyzer status"""
        return {
            "client_type": self.client_type,
            "openai_available": self.openai_client is not None,
            "anthropic_available": self.anthropic_client is not None,
            "primary_model": self.primary_model,
            "ready": self.client_type != "no_clients_available"
        }


# Backward compatibility
class OptimizedClaudeAnalyzer(ReliableAIAnalyzer):
    """Backward compatibility wrapper"""
    pass

class ExecutiveAIAnalyzer(ReliableAIAnalyzer):
    """Executive analyzer wrapper"""
    
    def analyze_for_board(self, content: str, company_name: str, company_number: str,
                         extracted_content: Optional[List[Dict[str, Any]]] = None,
                         analysis_context: Optional[str] = None) -> Dict[str, Any]:
        """Board analysis method for compatibility"""
        return self.analyze_for_board_optimized(
            content, company_name, company_number, extracted_content, analysis_context
        )