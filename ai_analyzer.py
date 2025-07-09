#!/usr/bin/env python3
"""
AI Archetype Analyzer for Strategic Analysis
Uses OpenAI/Anthropic APIs for intelligent business and risk strategy classification
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional
import time

logger = logging.getLogger(__name__)

class AIArchetypeAnalyzer:
    """AI-powered archetype analysis for business and risk strategies"""
    
    def __init__(self):
        """Initialize AI analyzer with available providers"""
        self.client_type = "fallback"
        self.openai_client = None
        
        logger.info("🚀 AIArchetypeAnalyzer v2.0 starting...")
        
        # Try to initialize OpenAI
        self._init_openai()
        
        # Define archetypes - Finance-specific comprehensive definitions
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
        
        logger.info(f"✅ AIArchetypeAnalyzer v2.0 completed. Client type: {self.client_type}")
    
    def _init_openai(self):
        """Initialize OpenAI client - using v0.28.1 for stability"""
        try:
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                logger.warning("⚠️ No OpenAI API key found")
                return
                
            # Import OpenAI v0.28.x (stable version)
            import openai
            logger.info(f"OpenAI module version: {openai.__version__}")
            
            # Set API key for v0.28.x
            openai.api_key = api_key
            self.openai_client = openai
            self.client_type = "openai_v028"
            logger.info("✅ OpenAI v0.28.x client initialized successfully")
                    
        except Exception as e:
            logger.warning(f"OpenAI setup failed: {e}")
            logger.info("Falling back to pattern-based analysis")
    
    def _init_anthropic(self):
        """Anthropic not used - method removed"""
        pass
    
    def analyze_archetypes(self, content: str, company_name: str, company_number: str, 
                          extracted_content: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyze business and risk archetypes
        
        Args:
            content: Combined document content
            company_name: Company name
            company_number: Company number
            extracted_content: Individual file data
            
        Returns:
            Analysis results
        """
        try:
            if self.client_type == "openai_v028":
                return self._analyze_with_openai_v028(content, company_name, company_number, extracted_content)
            elif self.client_type in ["openai", "openai_legacy"]:
                return self._analyze_with_openai(content, company_name, company_number, extracted_content)
            else:
                return self._analyze_with_fallback(content, company_name, company_number, extracted_content)
                
        except Exception as e:
            logger.error(f"Error in archetype analysis: {e}")
            return self._analyze_with_fallback(content, company_name, company_number, extracted_content)
    
    def _analyze_with_openai(self, content: str, company_name: str, company_number: str, 
                           extracted_content: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze using OpenAI API (backup method for v1.x)"""
        try:
            # Sample content for analysis
            sample_content = content[:15000] if len(content) > 15000 else content
            
            prompt = self._create_analysis_prompt(sample_content, company_name)
            
            # Legacy OpenAI client
            response = self.openai_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert business strategy analyst specializing in financial services."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            analysis_text = response.choices[0].message.content
            
            return self._parse_ai_response(analysis_text, "openai", extracted_content)
            
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            logger.info("Falling back to pattern analysis")
            return self._analyze_with_fallback(content, company_name, company_number, extracted_content)

    def _analyze_with_openai_v028(self, content: str, company_name: str, company_number: str, 
                                extracted_content: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze using OpenAI API v0.28.x"""
        try:
            # Sample content for analysis
            sample_content = content[:15000] if len(content) > 15000 else content
            
            prompt = self._create_analysis_prompt(sample_content, company_name)
            
            # Use v0.28.x API format
            response = self.openai_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert business strategy analyst specializing in financial services."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            analysis_text = response.choices[0].message.content
            return self._parse_ai_response(analysis_text, "openai_v028", extracted_content)
            
        except Exception as e:
            logger.error(f"OpenAI v0.28.x analysis failed: {e}")
            logger.info("Falling back to pattern analysis")
            return self._analyze_with_fallback(content, company_name, company_number, extracted_content)
    
    def _analyze_with_anthropic(self, content: str, company_name: str, company_number: str,
                              extracted_content: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Anthropic analysis removed - not used"""
        return self._analyze_with_fallback(content, company_name, company_number, extracted_content)
    
    def _analyze_with_fallback(self, content: str, company_name: str, company_number: str,
                             extracted_content: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Fallback pattern-based analysis"""
        try:
            # Pattern-based keyword analysis
            content_lower = content.lower()
            
            # Business strategy analysis
            business_scores = {}
            for archetype, description in self.business_archetypes.items():
                score = self._calculate_pattern_score(content_lower, archetype.lower())
                business_scores[archetype] = score
            
            # Risk strategy analysis  
            risk_scores = {}
            for archetype, description in self.risk_archetypes.items():
                score = self._calculate_pattern_score(content_lower, archetype.lower())
                risk_scores[archetype] = score
            
            # Find dominant strategies
            business_dominant = max(business_scores, key=business_scores.get)
            risk_dominant = max(risk_scores, key=risk_scores.get)
            
            return {
                'business_strategy_archetypes': {
                    'dominant': business_dominant,
                    'secondary': self._get_secondary(business_scores, business_dominant),
                    'scores': business_scores,
                    'reasoning': f"Pattern analysis indicates {business_dominant} orientation based on keyword frequency and context.",
                    'definition': self.business_archetypes.get(business_dominant, "Definition not available"),
                    'secondary_definition': self.business_archetypes.get(self._get_secondary(business_scores, business_dominant), "Definition not available")
                },
                'risk_strategy_archetypes': {
                    'dominant': risk_dominant,
                    'secondary': self._get_secondary(risk_scores, risk_dominant),
                    'scores': risk_scores,
                    'reasoning': f"Analysis suggests {risk_dominant} risk management approach based on content patterns.",
                    'definition': self.risk_archetypes.get(risk_dominant, "Definition not available"),
                    'secondary_definition': self.risk_archetypes.get(self._get_secondary(risk_scores, risk_dominant), "Definition not available")
                },
                'analysis_type': 'fallback_pattern',
                'confidence_level': 'medium',
                'files_analyzed': len(extracted_content) if extracted_content else 1,
                'content_length': len(content),
                'archetype_definitions': {
                    'business_archetypes': self.business_archetypes,
                    'risk_archetypes': self.risk_archetypes
                }
            }
            
        except Exception as e:
            logger.error(f"Fallback analysis failed: {e}")
            return self._get_default_analysis()
    
    def _create_analysis_prompt(self, content: str, company_name: str) -> str:
        """Create analysis prompt for AI"""
        return f"""
Analyze the following company documents for {company_name} and classify their business and risk strategies.

Business Strategy Archetypes:
{json.dumps(self.business_archetypes, indent=2)}

Risk Strategy Archetypes:
{json.dumps(self.risk_archetypes, indent=2)}

Document Content:
{content}

Please provide:
1. Primary business strategy archetype and reasoning
2. Primary risk strategy archetype and reasoning
3. Secondary options for both
4. Confidence level (high/medium/low)

Format your response as JSON with keys: business_primary, business_reasoning, risk_primary, risk_reasoning, business_secondary, risk_secondary, confidence.
"""
    
    def _parse_ai_response(self, response_text: str, client_type: str, extracted_content: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Parse AI response into structured format"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                ai_data = json.loads(json_match.group())
                
                return {
                    'business_strategy_archetypes': {
                        'dominant': ai_data.get('business_primary', 'Disciplined Specialist Growth'),
                        'secondary': ai_data.get('business_secondary', 'Balance-Sheet Steward'),
                        'reasoning': ai_data.get('business_reasoning', 'AI-generated analysis'),
                        'definition': self.business_archetypes.get(ai_data.get('business_primary', 'Disciplined Specialist Growth'), "Definition not available"),
                        'secondary_definition': self.business_archetypes.get(ai_data.get('business_secondary', 'Balance-Sheet Steward'), "Definition not available")
                    },
                    'risk_strategy_archetypes': {
                        'dominant': ai_data.get('risk_primary', 'Risk-First Conservative'),
                        'secondary': ai_data.get('risk_secondary', 'Rules-Led Operator'), 
                        'reasoning': ai_data.get('risk_reasoning', 'AI-generated analysis'),
                        'definition': self.risk_archetypes.get(ai_data.get('risk_primary', 'Risk-First Conservative'), "Definition not available"),
                        'secondary_definition': self.risk_archetypes.get(ai_data.get('risk_secondary', 'Rules-Led Operator'), "Definition not available")
                    },
                    'analysis_type': f'ai_{client_type}',
                    'confidence_level': ai_data.get('confidence', 'medium'),
                    'files_analyzed': len(extracted_content) if extracted_content else 1,
                    'ai_raw_response': response_text,
                    'archetype_definitions': {
                        'business_archetypes': self.business_archetypes,
                        'risk_archetypes': self.risk_archetypes
                    }
                }
            else:
                # Fallback parsing
                return self._parse_text_response(response_text, client_type, extracted_content)
                
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return self._get_default_analysis()
    
    def _parse_text_response(self, response_text: str, client_type: str, extracted_content: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Parse non-JSON AI response"""
        # Simple text parsing for business and risk strategies
        business_archetype = "Growth"  # Default
        risk_archetype = "Balanced"    # Default
        
        # Look for mentioned archetypes
        response_lower = response_text.lower()
        
        for archetype in self.business_archetypes.keys():
            if archetype.lower() in response_lower:
                business_archetype = archetype
                break
        
        for archetype in self.risk_archetypes.keys():
            if archetype.lower() in response_lower:
                risk_archetype = archetype
                break
        
        return {
            'business_strategy_archetypes': {
                'dominant': business_archetype,
                'secondary': 'Innovation',
                'reasoning': f'AI analysis suggests {business_archetype} strategy based on document content.'
            },
            'risk_strategy_archetypes': {
                'dominant': risk_archetype,
                'secondary': 'Conservative',
                'reasoning': f'Risk management approach appears to be {risk_archetype} based on analysis.'
            },
            'analysis_type': f'ai_{client_type}_text',
            'confidence_level': 'medium',
            'files_analyzed': len(extracted_content) if extracted_content else 1,
            'ai_raw_response': response_text
        }
    
    def _calculate_pattern_score(self, content: str, archetype: str) -> float:
        """Calculate pattern-based score for archetype using finance-specific keywords"""
        
        # Finance-specific keyword mapping for business archetypes
        business_keywords = {
            'scale-through-distribution': ['distribution', 'channels', 'partners', 'network', 'partnership', 'broker', 'agent'],
            'land-grab platform': ['platform', 'market share', 'pricing', 'incentives', 'acquisition', 'growth', 'aggressive'],
            'asset-velocity maximiser': ['origination', 'turnover', 'velocity', 'bridging', 'invoice', 'quick', 'speed'],
            'yield-hunting': ['yield', 'margin', 'premium', 'credit', 'high-margin', 'pricing', 'risk premium'],
            'fee-extraction engine': ['fees', 'ancillary', 'cross-sell', 'add-on', 'packaged', 'service charges'],
            'disciplined specialist growth': ['specialist', 'niche', 'underwriting', 'opportunistic', 'balance sheet'],
            'expert niche leader': ['expertise', 'specialisation', 'niche', 'leader', 'specialist', 'focused'],
            'service-driven differentiator': ['service', 'experience', 'advice', 'client', 'customer', 'relationship'],
            'cost-leadership operator': ['cost', 'efficiency', 'lean', 'digital', 'automation', 'operational'],
            'tech-productivity accelerator': ['technology', 'automation', 'digital', 'productivity', 'innovation'],
            'product-innovation flywheel': ['innovation', 'product', 'development', 'new', 'features', 'launch'],
            'data-monetisation pioneer': ['data', 'analytics', 'insights', 'monetisation', 'technology'],
            'balance-sheet steward': ['capital', 'prudent', 'conservative', 'strength', 'stability'],
            'regulatory shelter occupant': ['regulatory', 'protection', 'compliance', 'mandate', 'licensed'],
            'regulator-mandated remediation': ['remediation', 'regulatory', 'compliance', 'improvement', 'requirements'],
            'wind-down / run-off': ['run-off', 'wind-down', 'closure', 'legacy', 'maturity'],
            'strategic withdrawal': ['withdrawal', 'divestment', 'exit', 'disposal', 'refocus'],
            'distressed-asset harvester': ['distressed', 'npl', 'acquisition', 'recovery', 'workout'],
            'counter-cyclical capitaliser': ['counter-cyclical', 'opportunistic', 'expansion', 'liquidity']
        }
        
        # Risk archetype keywords
        risk_keywords = {
            'risk-first conservative': ['conservative', 'prudent', 'capital preservation', 'compliance', 'resilience'],
            'rules-led operator': ['rules', 'procedures', 'compliance', 'controls', 'governance'],
            'resilience-focused architect': ['resilience', 'continuity', 'stress testing', 'scenario', 'planning'],
            'strategic risk-taker': ['risk appetite', 'growth', 'strategic', 'opportunity', 'calculated'],
            'control-lag follower': ['expansion', 'catch-up', 'scaling', 'development', 'control'],
            'reactive remediator': ['reactive', 'remediation', 'enforcement', 'findings', 'improvement'],
            'reputation-first shield': ['reputation', 'political', 'reputational', 'stakeholder', 'public'],
            'embedded risk partner': ['embedded', 'collaborative', 'partnership', 'integrated', 'frontline'],
            'quant-control enthusiast': ['quantitative', 'data', 'analytics', 'automation', 'predictive'],
            'tick-box minimalist': ['compliance', 'minimal', 'superficial', 'tick-box', 'procedural'],
            'mission-driven prudence': ['mission', 'stakeholder', 'community', 'social', 'purpose']
        }
        
        # Combine business and risk keywords
        all_keywords = {**business_keywords, **risk_keywords}
        
        # Normalize archetype name for lookup
        archetype_key = archetype.lower().replace(' ', '-').replace('/', '-')
        
        # Get keywords for this archetype
        archetype_keywords = all_keywords.get(archetype_key, [])
        
        # Calculate score based on keyword frequency
        score = 0
        for keyword in archetype_keywords:
            score += content.count(keyword.lower())
        
        # Add bonus for exact archetype name mentions
        archetype_words = archetype.lower().split()
        for word in archetype_words:
            if word in content and len(word) > 3:  # Skip short words
                score += 2
        
        # Normalize by content length
        normalized_score = score / max(len(content.split()), 1) * 1000
        
        return normalized_score
    
    def _get_secondary(self, scores: Dict[str, float], primary: str) -> str:
        """Get secondary archetype"""
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for archetype, score in sorted_scores:
            if archetype != primary:
                return archetype
        return list(scores.keys())[0]
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Get default analysis when all else fails"""
        return {
            'business_strategy_archetypes': {
                'dominant': 'Disciplined Specialist Growth',
                'secondary': 'Balance-Sheet Steward',
                'reasoning': 'Default analysis - insufficient data for detailed classification',
                'definition': self.business_archetypes.get('Disciplined Specialist Growth', "Niche focus with strong underwriting edge; grows opportunistically while recycling balance-sheet"),
                'secondary_definition': self.business_archetypes.get('Balance-Sheet Steward', "Low-risk appetite, prioritises capital strength and membership value")
            },
            'risk_strategy_archetypes': {
                'dominant': 'Risk-First Conservative',
                'secondary': 'Rules-Led Operator',
                'reasoning': 'Default risk assessment - conservative approach assumed',
                'definition': self.risk_archetypes.get('Risk-First Conservative', "Prioritises capital preservation and regulatory compliance; growth is secondary to resilience"),
                'secondary_definition': self.risk_archetypes.get('Rules-Led Operator', "Strict adherence to rules and checklists; prioritises control consistency over judgment or speed")
            },
            'analysis_type': 'default',
            'confidence_level': 'low',
            'files_analyzed': 0,
            'archetype_definitions': {
                'business_archetypes': self.business_archetypes,
                'risk_archetypes': self.risk_archetypes
            }
        }

if __name__ == "__main__":
    # Test the AI analyzer
    print("Testing AI Archetype Analyzer...")
    
    analyzer = AIArchetypeAnalyzer()
    
    # Test with sample content
    sample_content = """
    The company is focused on growth and expansion into new markets.
    We are investing heavily in innovation and technology development.
    Risk management is balanced with growth objectives.
    """
    
    result = analyzer.analyze_archetypes(sample_content, "Test Company", "12345678")
    
    print("Analysis Results:")
    print(json.dumps(result, indent=2))
    
    print("AI Analyzer test completed.")