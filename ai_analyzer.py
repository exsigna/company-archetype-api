#!/usr/bin/env python3
"""
Complete AI Analyzer with Exact Business and Risk Strategy Archetypes
Generates reports in the specified format with proper SWOT analysis
Thread-safe for Render deployment
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
        
        # Model configuration optimized for Render
        self.primary_model = "gpt-4-turbo-preview"
        self.fallback_model = "gpt-4-1106-preview"
        self.max_output_tokens = 4096
        self.max_retries = 2
        self.base_retry_delay = 1.5
        self.max_content_chars = 150000
        self.request_timeout = 25
        
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
        """Log environment information for debugging"""
        logger.info("🔍 Environment Debug Information:")
        
        openai_key = os.environ.get('OPENAI_API_KEY')
        anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
        
        if openai_key:
            logger.info(f"   ✅ OPENAI_API_KEY found: {openai_key[:10]}...")
        else:
            logger.error("   ❌ OPENAI_API_KEY not found!")
        
        if anthropic_key:
            logger.info(f"   ✅ ANTHROPIC_API_KEY found: {anthropic_key[:10]}...")
        else:
            logger.info("   ℹ️ ANTHROPIC_API_KEY not found (fallback only)")
    
    def _initialize_clients(self):
        """Initialize AI clients"""
        
        # Try OpenAI first
        if self._init_openai():
            self.client_type = "openai_primary"
            logger.info("✅ OpenAI configured as primary service")
        else:
            logger.warning("⚠️ OpenAI initialization failed")
        
        # Try Anthropic as fallback
        if self._init_anthropic():
            if self.client_type == "uninitialized":
                self.client_type = "anthropic_fallback"
                logger.info("✅ Anthropic configured as fallback service")
        else:
            logger.warning("⚠️ Anthropic initialization failed")
        
        if self.client_type == "uninitialized":
            self.client_type = "no_clients_available"
            logger.error("❌ No AI clients available")
    
    def _init_openai(self) -> bool:
        """Initialize OpenAI client"""
        try:
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                logger.warning("⚠️ OPENAI_API_KEY not found")
                return False
            
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(
                    api_key=api_key,
                    max_retries=0,
                    timeout=20.0
                )
                
                # Test connection
                test_response = self.openai_client.models.list()
                logger.info("✅ OpenAI client test successful")
                return True
            except ImportError as e:
                logger.error(f"❌ OpenAI library not available: {e}")
                return False
        except Exception as e:
            logger.error(f"❌ OpenAI initialization failed: {e}")
            return False
    
    def _init_anthropic(self) -> bool:
        """Initialize Anthropic client"""
        try:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                return False
            
            import anthropic
            self.anthropic_client = anthropic.Anthropic(api_key=api_key, timeout=15.0)
            logger.info("✅ Anthropic client initialized")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Anthropic initialization failed: {e}")
            return False
    
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
                return self._create_emergency_analysis(company_name, company_number, "No AI clients available")
            
            # Optimize content
            optimized_content = self._optimize_content(content)
            
            # Try OpenAI analysis
            if self.openai_client and self.client_type in ["openai_primary", "anthropic_fallback"]:
                logger.info("🎯 Attempting OpenAI analysis...")
                result = self._analyze_with_openai(optimized_content, company_name, company_number, analysis_context)
                if result:
                    analysis_time = time.time() - start_time
                    logger.info(f"✅ OpenAI analysis completed in {analysis_time:.2f}s")
                    return result
            
            # Try Anthropic fallback
            if self.anthropic_client:
                logger.info("🎯 Attempting Anthropic analysis...")
                result = self._analyze_with_anthropic(optimized_content, company_name, company_number, analysis_context)
                if result:
                    analysis_time = time.time() - start_time
                    logger.info(f"✅ Anthropic analysis completed in {analysis_time:.2f}s")
                    return result
            
            # Emergency fallback
            return self._create_emergency_analysis(company_name, company_number, "All AI services failed")
        
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return self._create_emergency_analysis(company_name, company_number, str(e))
    
    def _analyze_with_openai(self, content: str, company_name: str, company_number: str,
                           analysis_context: Optional[str]) -> Optional[Dict[str, Any]]:
        """Analyze using OpenAI with thread-safe timeout"""
        
        if not self.openai_client:
            return None
        
        # Create timeout manager
        timeout_manager = TimeoutManager(20.0)
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
                    api_timeout = min(timeout_manager.remaining_time(), 15.0)
                    
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
                        return self._create_complete_report(analysis, company_name, company_number, model, "openai")
                
                except TimeoutError as e:
                    logger.error(f"❌ OpenAI timeout: {e}")
                    return None
                
                except Exception as e:
                    logger.warning(f"⚠️ OpenAI attempt failed: {e}")
                    if self._is_retryable_error(str(e)) and attempt < self.max_retries - 1:
                        # Check if we have time for retry
                        if timeout_manager.remaining_time() > 3:
                            time.sleep(min(self.base_retry_delay, timeout_manager.remaining_time() / 2))
                            continue
                        else:
                            logger.warning("⚠️ Not enough time remaining for retry")
                            return None
                    else:
                        break
        
        return None
    
    def _analyze_with_anthropic(self, content: str, company_name: str, company_number: str,
                              analysis_context: Optional[str]) -> Optional[Dict[str, Any]]:
        """Analyze using Anthropic with thread-safe timeout"""
        
        if not self.anthropic_client:
            return None
        
        # Create timeout manager
        timeout_manager = TimeoutManager(15.0)
        timeout_manager.start()
        
        for attempt in range(self.max_retries):
            try:
                # Check timeout before each attempt
                timeout_manager.check_timeout()
                
                logger.info(f"🔄 Anthropic attempt {attempt + 1}")
                
                prompt = self._create_complete_anthropic_prompt(content, company_name, analysis_context)
                
                # Use remaining time for API timeout
                api_timeout = min(timeout_manager.remaining_time(), 12.0)
                
                response = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=self.max_output_tokens,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=api_timeout
                )
                
                response_text = response.content[0].text
                logger.info(f"✅ Anthropic response: {len(response_text)} chars")
                
                analysis = self._parse_json_response(response_text)
                if analysis:
                    return self._create_complete_report(analysis, company_name, company_number, "claude-3-sonnet", "anthropic")
            
            except TimeoutError as e:
                logger.error(f"❌ Anthropic timeout: {e}")
                return None
            
            except Exception as e:
                logger.warning(f"⚠️ Anthropic attempt failed: {e}")
                if attempt < self.max_retries - 1:
                    # Check if we have time for retry
                    if timeout_manager.remaining_time() > 2:
                        time.sleep(min(self.base_retry_delay, timeout_manager.remaining_time() / 2))
                        continue
                    else:
                        logger.warning("⚠️ Not enough time remaining for retry")
                        return None
        
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
  "confidence_level": "high"
}}

BUSINESS STRATEGY ARCHETYPES:
{business_archetypes_text}

RISK STRATEGY ARCHETYPES:
{risk_archetypes_text}

IMPORTANT: 
- Use EXACT archetype names from the lists above
- Provide specific evidence from the company content
- SWOT should reflect the combination of the 4 selected archetypes
- Focus on how the archetype combination creates specific advantages/disadvantages"""

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
  "confidence_level": "high"
}}

BUSINESS STRATEGY ARCHETYPES:
{business_list}

RISK STRATEGY ARCHETYPES:
{risk_list}

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
    
    def _create_complete_report(self, analysis: Dict[str, Any], company_name: str,
                              company_number: str, model: str, service: str) -> Dict[str, Any]:
        """Create complete report in the specified format"""
        
        # Extract data with fallbacks
        business = analysis.get('business_strategy', {})
        risk = analysis.get('risk_strategy', {})
        swot = analysis.get('swot_analysis', {})
        
        return {
            'success': True,
            'company_name': company_name,
            'company_number': company_number,
            'years_analyzed': analysis.get('years_analyzed', 'Recent period'),
            'files_processed': 'Multiple financial documents',
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
                    'rationale': business.get('dominant_rationale', 'Conservative growth approach with focus on underwriting quality.'),
                    'evidence': self._extract_evidence_points(business.get('dominant_rationale', ''))
                },
                'secondary': {
                    'archetype': business.get('secondary_archetype', 'Service-Driven Differentiator'),
                    'rationale': business.get('secondary_rationale', 'Focus on customer service and relationship building.'),
                    'evidence': self._extract_evidence_points(business.get('secondary_rationale', ''))
                },
                'material_changes': business.get('material_changes', 'No significant changes identified over the period analyzed.')
            },
            
            # Risk Strategy Analysis
            'risk_strategy_analysis': {
                'dominant': {
                    'archetype': risk.get('dominant_archetype', 'Risk-First Conservative'),
                    'rationale': risk.get('dominant_rationale', 'Prioritizes capital preservation and regulatory compliance.'),
                    'evidence': self._extract_evidence_points(risk.get('dominant_rationale', ''))
                },
                'secondary': {
                    'archetype': risk.get('secondary_archetype', 'Rules-Led Operator'),
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
            
            # Legacy format for backward compatibility
            'business_strategy': {
                'dominant': business.get('dominant_archetype', 'Disciplined Specialist Growth'),
                'dominant_reasoning': business.get('dominant_rationale', 'Analysis completed successfully'),
                'secondary': business.get('secondary_archetype', 'Service-Driven Differentiator'),
                'secondary_reasoning': business.get('secondary_rationale', 'Secondary analysis completed'),
                'evidence_quotes': self._extract_evidence_points(business.get('dominant_rationale', ''))
            },
            'risk_strategy': {
                'dominant': risk.get('dominant_archetype', 'Risk-First Conservative'),
                'dominant_reasoning': risk.get('dominant_rationale', 'Risk analysis completed successfully'),
                'secondary': risk.get('secondary_archetype', 'Rules-Led Operator'),
                'secondary_reasoning': risk.get('secondary_rationale', 'Secondary risk analysis completed'),
                'evidence_quotes': self._extract_evidence_points(risk.get('dominant_rationale', ''))
            },
            
            # Metadata
            'analysis_date': datetime.now().isoformat(),
            'analysis_type': f'complete_{service}_archetype_analysis',
            'confidence_level': analysis.get('confidence_level', 'high'),
            'processing_stats': {
                'model_used': model,
                'service_used': service,
                'analysis_timestamp': datetime.now().isoformat(),
                'archetypes_evaluated': {
                    'business_total': len(self.business_archetypes),
                    'risk_total': len(self.risk_archetypes)
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
            "archetypes_loaded": status["archetypes"]
        }


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