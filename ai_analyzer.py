#!/usr/bin/env python3
"""
AI Analyzer - Streamlined for Business and Risk Strategy Archetype Classification Only
"""

import logging
import json
import os
import re
from datetime import datetime
from typing import Dict, Any, Optional, List

from config import (
    DEFAULT_OPENAI_MODEL, DEFAULT_ANTHROPIC_MODEL, AI_MAX_TOKENS, AI_TEMPERATURE
)

logger = logging.getLogger(__name__)

class AIArchetypeAnalyzer:
    """AI-powered analyzer focused specifically on archetype classification"""
    
    def __init__(self):
        """Initialize the AI analyzer with proper API configuration"""
        self.client = None
        self.client_type = "fallback"
        self._setup_client()
        
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

    def _setup_client(self):
        """Setup the AI client with proper error handling"""
        try:
            # Check for OpenAI API key first
            openai_key = os.getenv('OPENAI_API_KEY')
            
            if openai_key and openai_key.strip() and not openai_key.startswith('your_'):
                try:
                    import openai
                    self.client = openai.OpenAI(api_key=openai_key)
                    self.client_type = "openai"
                    logger.info("✅ OpenAI client initialized successfully")
                    return
                except ImportError:
                    logger.warning("OpenAI library not installed")
                except Exception as e:
                    logger.warning(f"OpenAI setup failed: {e}")
            
            # Fallback to Anthropic
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            if anthropic_key and anthropic_key.strip() and not anthropic_key.startswith('your_'):
                try:
                    import anthropic
                    self.client = anthropic.Anthropic(api_key=anthropic_key)
                    self.client_type = "anthropic"
                    logger.info("✅ Anthropic client initialized successfully")
                    return
                except ImportError:
                    logger.warning("Anthropic library not installed")
                except Exception as e:
                    logger.warning(f"Anthropic setup failed: {e}")
            
            logger.warning("⚠️  No valid AI API keys found - will use fallback analysis")
            self.client = None
            self.client_type = "fallback"
            
        except Exception as e:
            logger.error(f"Error setting up AI client: {e}")
            self.client = None
            self.client_type = "fallback"

    def analyze_archetypes(self, content: str, company_name: str, company_number: str) -> Dict[str, Any]:
        """
        Analyze company archetypes focusing on dominant and secondary classifications
        
        Args:
            content: Combined content from annual reports
            company_name: Company name
            company_number: Company registration number
            
        Returns:
            Dictionary containing archetype analysis with dominant/secondary classifications
        """
        try:
            logger.info(f"🏛️ Starting archetype analysis for {company_name}")
            
            if self.client and self.client_type in ["openai", "anthropic"]:
                # AI-powered analysis
                business_analysis = self._classify_dominant_and_secondary_archetypes(
                    content, self.business_archetypes, "Business Strategy"
                )
                risk_analysis = self._classify_dominant_and_secondary_archetypes(
                    content, self.risk_archetypes, "Risk Strategy"
                )
                
                return self._create_success_result(
                    analysis_type="ai_archetype_classification",
                    company_name=company_name,
                    company_number=company_number,
                    business_strategy_archetypes=business_analysis,
                    risk_strategy_archetypes=risk_analysis,
                    model_used=f"{self.client_type}_{DEFAULT_OPENAI_MODEL if self.client_type == 'openai' else DEFAULT_ANTHROPIC_MODEL}"
                )
            else:
                logger.warning("🔄 Using fallback archetype analysis")
                return self._fallback_archetype_analysis(content, company_name, company_number)
                
        except Exception as e:
            logger.error(f"❌ Error in analyze_archetypes: {e}")
            return self._create_error_result(str(e))

    def _classify_dominant_and_secondary_archetypes(self, content: str, archetype_dict: Dict[str, str], label: str) -> Dict[str, Any]:
        """Classify and rank the top two matching archetypes with reasoning."""
        
        # Format archetype definitions for the prompt
        archetypes_text = "\n".join([f"- {name}: {definition}" for name, definition in archetype_dict.items()])
        
        prompt = f"""You are an expert analyst evaluating a UK financial services firm. Your task is to identify the dominant and secondary {label} Archetypes based on the annual report content below. Please use UK English spelling and terminology throughout your response.

Available {label} Archetypes:
{archetypes_text}

Instructions:
1. Analyse the content to understand the firm's strategic approach and priorities
2. Select the DOMINANT archetype that most strongly characterises the firm's {label.lower()}
3. Select a SECONDARY archetype that also applies but is less dominant (or "None" if no secondary archetype clearly applies)
4. Provide detailed reasoning based on specific evidence from the text
5. Focus on the period under review and any strategic changes mentioned
6. Use UK English spelling (e.g., 'analyse', 'realise', 'organisation', 'behaviour')

Output format (exactly as shown):
Dominant: <archetype_name>
Secondary: <archetype_name or "None">
Reasoning: <detailed explanation with specific evidence from the text>

TEXT TO ANALYSE:
{content[:8000]}"""

        try:
            if self.client_type == "openai":
                response = self.client.chat.completions.create(
                    model=DEFAULT_OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": f"You are an expert {label.lower()} analyst specialising in archetype classification for UK financial services firms. Always use UK English spelling and terminology in your responses."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=AI_TEMPERATURE
                )
                response_text = response.choices[0].message.content
                
            elif self.client_type == "anthropic":
                response = self.client.messages.create(
                    model=DEFAULT_ANTHROPIC_MODEL,
                    max_tokens=1000,
                    temperature=AI_TEMPERATURE,
                    system=f"You are an expert {label.lower()} analyst specialising in archetype classification for UK financial services firms. Always use UK English spelling and terminology in your responses.",
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.content[0].text
            
            return self._parse_archetype_response(response_text)
            
        except Exception as e:
            logger.error(f"❌ AI archetype classification failed for {label}: {e}")
            return self._fallback_single_archetype_analysis(content, archetype_dict, label)

    def _parse_archetype_response(self, response: str) -> Dict[str, str]:
        """Extracts dominant and secondary archetype classifications from LLM output."""
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

    def _fallback_archetype_analysis(self, content: str, company_name: str, company_number: str) -> Dict[str, Any]:
        """Enhanced fallback archetype analysis using pattern matching"""
        
        # Classify business archetypes
        business_analysis = self._fallback_single_archetype_analysis(
            content, self.business_archetypes, "Business Strategy"
        )
        
        # Classify risk archetypes
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
        """Fallback pattern-based archetype analysis for a single category"""
        content_lower = content.lower()
        archetype_scores = {}
        
        # Define keyword patterns for each archetype
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
        
        # Score each archetype based on keyword matches
        for archetype, patterns in keyword_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content_lower))
                score += matches
            archetype_scores[archetype] = score
        
        # Sort by score and select top archetypes
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

    def _extract_company_name(self, content: str, company_number: str) -> str:
        """Enhanced company name extraction"""
        patterns = [
            rf'({re.escape(company_number)}.*?)(?:\n|Limited|Ltd)',
            r'Together Personal Finance Limited',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Limited)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Ltd)',
            rf'Company\s+Name:?\s*([^\\n]+)',
            rf'({company_number})\s+([^\\n]+Limited)',
            r'Company.*?is\s+([A-Z][^.]*Limited)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content[:3000], re.IGNORECASE)
            if match:
                if len(match.groups()) >= 2:
                    name = match.group(2) if match.group(2) else match.group(1)
                else:
                    name = match.group(1) if '(' in pattern else match.group(0)
                
                name = re.sub(r'^\d+\s*', '', name.strip())
                name = re.sub(r'\s+', ' ', name)
                
                if len(name) > 5 and 'limited' in name.lower():
                    return name
        
        return f"Company {company_number}"

    def _create_success_result(self, analysis_type: str, company_name: str, 
                             company_number: str, business_strategy_archetypes: Dict[str, str],
                             risk_strategy_archetypes: Dict[str, str], model_used: str = None) -> Dict[str, Any]:
        """Create standardized success result for archetype analysis"""
        
        # Generate summary text
        analysis_text = self._generate_archetype_summary(
            company_name, company_number, business_strategy_archetypes, risk_strategy_archetypes
        )
        
        result = {
            "success": True,
            "analysis_type": analysis_type,
            "company_name": company_name,
            "company_number": company_number,
            "business_strategy_archetypes": business_strategy_archetypes,
            "risk_strategy_archetypes": risk_strategy_archetypes,
            "analysis_text": analysis_text,
            "timestamp": datetime.now().isoformat(),
        }
        
        if model_used:
            result["model_used"] = model_used
            
        return result

    def _generate_archetype_summary(self, company_name: str, company_number: str,
                                   business_archetypes: Dict[str, str], 
                                   risk_archetypes: Dict[str, str]) -> str:
        """Generate a summary of the archetype analysis"""
        
        summary = f"""# ARCHETYPE CLASSIFICATION: {company_name} ({company_number})

## BUSINESS STRATEGY ARCHETYPES

**Dominant Archetype**: {business_archetypes['dominant']}
- Definition: {self.business_archetypes.get(business_archetypes['dominant'], 'N/A')}

"""
        
        if business_archetypes['secondary']:
            summary += f"""**Secondary Archetype**: {business_archetypes['secondary']}
- Definition: {self.business_archetypes.get(business_archetypes['secondary'], 'N/A')}

"""
        
        summary += f"""**Analysis**: {business_archetypes['reasoning']}

## RISK STRATEGY ARCHETYPES

**Dominant Archetype**: {risk_archetypes['dominant']}
- Definition: {self.risk_archetypes.get(risk_archetypes['dominant'], 'N/A')}

"""
        
        if risk_archetypes['secondary']:
            summary += f"""**Secondary Archetype**: {risk_archetypes['secondary']}
- Definition: {self.risk_archetypes.get(risk_archetypes['secondary'], 'N/A')}

"""
        
        summary += f"""**Analysis**: {risk_archetypes['reasoning']}

## SUMMARY

{company_name} demonstrates characteristics primarily aligned with:
- **Business Strategy**: {business_archetypes['dominant']}
- **Risk Strategy**: {risk_archetypes['dominant']}

This archetype combination suggests a strategic approach focused on sustainable growth within defined risk parameters, consistent with regulatory expectations for UK financial services firms.

---
*Analysis based on annual report content for the period under review*
"""
        
        return summary

    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            "success": False,
            "error": error_msg,
            "analysis_type": "error",
            "timestamp": datetime.now().isoformat()
        }