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
        self.anthropic_client = None
        
        logger.info("🚀 AIArchetypeAnalyzer v2.0 starting...")
        
        # Try to initialize OpenAI
        self._init_openai()
        
        # Try to initialize Anthropic as backup
        if self.client_type == "fallback":
            self._init_anthropic()
        
        # Define archetypes
        self.business_archetypes = {
            "Growth": "Expansion-focused, market share acquisition, scaling operations",
            "Innovation": "R&D investment, new product development, technology advancement", 
            "Efficiency": "Cost optimization, process improvement, operational excellence",
            "Customer-Centric": "Customer satisfaction, service quality, relationship building",
            "Diversification": "Market expansion, product portfolio broadening, risk spreading",
            "Conservative": "Steady operations, incremental growth, stability focus"
        }
        
        self.risk_archetypes = {
            "Conservative": "Risk-averse, compliance-focused, stability prioritized",
            "Balanced": "Moderate risk tolerance, diversified approach, measured decisions",
            "Aggressive": "High risk tolerance, growth-oriented, bold strategic moves",
            "Adaptive": "Flexible risk management, responsive to market changes",
            "Compliance-Focused": "Regulatory adherence, governance emphasis, structured approach"
        }
        
        logger.info(f"✅ AIArchetypeAnalyzer v2.0 completed. Client type: {self.client_type}")
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            api_key = os.environ.get('OPENAI_API_KEY')
            if api_key:
                from openai import OpenAI
                # Remove any problematic parameters
                self.openai_client = OpenAI(
                    api_key=api_key,
                    timeout=30.0
                )
                self.client_type = "openai"
                logger.info("✅ OpenAI client initialized")
            else:
                logger.warning("⚠️ No OpenAI API key found")
        except Exception as e:
            logger.warning(f"OpenAI setup failed: {e}")
            logger.info("Falling back to pattern-based analysis")
    
    def _init_anthropic(self):
        """Initialize Anthropic client"""
        try:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if api_key:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=api_key)
                self.client_type = "anthropic"
                logger.info("✅ Anthropic client initialized")
            else:
                logger.warning("⚠️ No Anthropic API key found")
        except Exception as e:
            logger.warning(f"Anthropic setup failed: {e}")
            logger.warning("Using fallback pattern analysis")
    
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
            if self.client_type == "openai":
                return self._analyze_with_openai(content, company_name, company_number, extracted_content)
            elif self.client_type == "anthropic":
                return self._analyze_with_anthropic(content, company_name, company_number, extracted_content)
            else:
                return self._analyze_with_fallback(content, company_name, company_number, extracted_content)
                
        except Exception as e:
            logger.error(f"Error in archetype analysis: {e}")
            return self._analyze_with_fallback(content, company_name, company_number, extracted_content)
    
    def _analyze_with_openai(self, content: str, company_name: str, company_number: str, 
                           extracted_content: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze using OpenAI API"""
        try:
            # Sample content for analysis
            sample_content = content[:15000] if len(content) > 15000 else content
            
            prompt = self._create_analysis_prompt(sample_content, company_name)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert business strategy analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            analysis_text = response.choices[0].message.content
            return self._parse_ai_response(analysis_text, "openai", extracted_content)
            
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return self._analyze_with_fallback(content, company_name, company_number, extracted_content)
    
    def _analyze_with_anthropic(self, content: str, company_name: str, company_number: str,
                              extracted_content: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze using Anthropic API"""
        try:
            # Sample content for analysis
            sample_content = content[:15000] if len(content) > 15000 else content
            
            prompt = self._create_analysis_prompt(sample_content, company_name)
            
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            analysis_text = response.content[0].text
            return self._parse_ai_response(analysis_text, "anthropic", extracted_content)
            
        except Exception as e:
            logger.error(f"Anthropic analysis failed: {e}")
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
                    'reasoning': f"Pattern analysis indicates {business_dominant.lower()} orientation based on keyword frequency and context."
                },
                'risk_strategy_archetypes': {
                    'dominant': risk_dominant,
                    'secondary': self._get_secondary(risk_scores, risk_dominant),
                    'scores': risk_scores,
                    'reasoning': f"Analysis suggests {risk_dominant.lower()} risk management approach based on content patterns."
                },
                'analysis_type': 'fallback_pattern',
                'confidence_level': 'medium',
                'files_analyzed': len(extracted_content) if extracted_content else 1,
                'content_length': len(content)
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
                        'dominant': ai_data.get('business_primary', 'Growth'),
                        'secondary': ai_data.get('business_secondary', 'Innovation'),
                        'reasoning': ai_data.get('business_reasoning', 'AI-generated analysis')
                    },
                    'risk_strategy_archetypes': {
                        'dominant': ai_data.get('risk_primary', 'Balanced'),
                        'secondary': ai_data.get('risk_secondary', 'Conservative'), 
                        'reasoning': ai_data.get('risk_reasoning', 'AI-generated analysis')
                    },
                    'analysis_type': f'ai_{client_type}',
                    'confidence_level': ai_data.get('confidence', 'medium'),
                    'files_analyzed': len(extracted_content) if extracted_content else 1,
                    'ai_raw_response': response_text
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
        """Calculate pattern-based score for archetype"""
        keywords = {
            'growth': ['grow', 'expand', 'increase', 'scale', 'acquisition', 'market share'],
            'innovation': ['innovate', 'technology', 'research', 'development', 'new product'],
            'efficiency': ['efficiency', 'cost', 'optimize', 'streamline', 'productivity'],
            'customer': ['customer', 'service', 'satisfaction', 'relationship', 'client'],
            'diversification': ['diversify', 'expand', 'portfolio', 'market', 'opportunity'],
            'conservative': ['stable', 'steady', 'maintain', 'preserve', 'reliable'],
            'balanced': ['balance', 'moderate', 'prudent', 'measured', 'diversified'],
            'aggressive': ['aggressive', 'bold', 'ambitious', 'rapid', 'accelerate'],
            'adaptive': ['adapt', 'flexible', 'responsive', 'agile', 'dynamic'],
            'compliance': ['compliance', 'regulatory', 'governance', 'policy', 'standards']
        }
        
        archetype_keywords = keywords.get(archetype, [])
        score = sum(content.count(keyword) for keyword in archetype_keywords)
        
        # Normalize by content length
        return score / max(len(content.split()), 1) * 1000
    
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
                'dominant': 'Growth',
                'secondary': 'Innovation',
                'reasoning': 'Default analysis - insufficient data for detailed classification'
            },
            'risk_strategy_archetypes': {
                'dominant': 'Balanced',
                'secondary': 'Conservative',
                'reasoning': 'Default risk assessment - balanced approach assumed'
            },
            'analysis_type': 'default',
            'confidence_level': 'low',
            'files_analyzed': 0
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