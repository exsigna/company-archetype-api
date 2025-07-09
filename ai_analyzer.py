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
        """Fallback pattern-based analysis with robust error handling"""
        try:
            # Pattern-based keyword analysis
            content_lower = str(content).lower() if content else ""
            
            # Business strategy analysis
            business_scores = {}
            for archetype, description in self.business_archetypes.items():
                try:
                    score = self._calculate_pattern_score(content_lower, str(archetype))
                    business_scores[str(archetype)] = float(score)
                except Exception as e:
                    logger.debug(f"Error scoring {archetype}: {e}")
                    business_scores[str(archetype)] = 0.0
            
            # Risk strategy analysis  
            risk_scores = {}
            for archetype, description in self.risk_archetypes.items():
                try:
                    score = self._calculate_pattern_score(content_lower, str(archetype))
                    risk_scores[str(archetype)] = float(score)
                except Exception as e:
                    logger.debug(f"Error scoring {archetype}: {e}")
                    risk_scores[str(archetype)] = 0.0
            
            # Find dominant strategies safely
            try:
                business_dominant = max(business_scores, key=business_scores.get) if business_scores else "Disciplined Specialist Growth"
                business_secondary = self._get_secondary(business_scores, business_dominant)
            except Exception as e:
                logger.error(f"Error finding business dominant: {e}")
                business_dominant = "Disciplined Specialist Growth"
                business_secondary = "Balance-Sheet Steward"
            
            try:
                risk_dominant = max(risk_scores, key=risk_scores.get) if risk_scores else "Risk-First Conservative"
                risk_secondary = self._get_secondary(risk_scores, risk_dominant)
            except Exception as e:
                logger.error(f"Error finding risk dominant: {e}")
                risk_dominant = "Risk-First Conservative"
                risk_secondary = "Rules-Led Operator"
            
            # Ensure all values are strings and valid
            business_dominant = str(business_dominant)
            business_secondary = str(business_secondary)
            risk_dominant = str(risk_dominant)
            risk_secondary = str(risk_secondary)
            
            # Validate archetypes exist
            if business_dominant not in self.business_archetypes:
                business_dominant = "Disciplined Specialist Growth"
            if business_secondary not in self.business_archetypes:
                business_secondary = "Balance-Sheet Steward"
            if risk_dominant not in self.risk_archetypes:
                risk_dominant = "Risk-First Conservative"
            if risk_secondary not in self.risk_archetypes:
                risk_secondary = "Rules-Led Operator"

            return {
                'business_strategy_archetypes': {
                    'dominant': business_dominant,
                    'secondary': business_secondary,
                    'scores': business_scores,
                    'reasoning': f"Pattern analysis indicates {business_dominant} orientation based on comprehensive keyword frequency analysis and document context review.",
                    'evidence': [f"Keyword analysis score: {business_scores.get(business_dominant, 0):.2f}", f"Secondary indicators support {business_secondary}"],
                    'definition': self.business_archetypes[business_dominant],
                    'secondary_definition': self.business_archetypes[business_secondary],
                    'comprehensive_analysis': self._format_business_analysis(business_dominant, f"Pattern analysis identifies {business_dominant} through systematic evaluation of strategic indicators in the company's documentation.", [f"Primary archetype scoring: {business_scores.get(business_dominant, 0):.2f}"])
                },
                'risk_strategy_archetypes': {
                    'dominant': risk_dominant,
                    'secondary': risk_secondary,
                    'scores': risk_scores,
                    'reasoning': f"Comprehensive analysis suggests {risk_dominant} risk management approach based on detailed content pattern evaluation and regulatory focus indicators.",
                    'evidence': [f"Risk keyword analysis score: {risk_scores.get(risk_dominant, 0):.2f}", f"Secondary risk indicators support {risk_secondary}"],
                    'definition': self.risk_archetypes[risk_dominant],
                    'secondary_definition': self.risk_archetypes[risk_secondary],
                    'comprehensive_analysis': self._format_risk_analysis(risk_dominant, f"Pattern analysis identifies {risk_dominant} through systematic evaluation of risk management indicators in the company's documentation.", [f"Primary archetype scoring: {risk_scores.get(risk_dominant, 0):.2f}"])
                },
                'analysis_type': 'fallback_pattern_comprehensive',
                'confidence_level': 'medium',
                'files_analyzed': len(extracted_content) if extracted_content else 1,
                'content_length': len(content) if content else 0,
                'supporting_quotes': [],
                'archetype_definitions': {
                    'business_archetypes': self.business_archetypes,
                    'risk_archetypes': self.risk_archetypes
                }
            }
            
        except Exception as e:
            logger.error(f"Fallback analysis failed: {e}")
            return self._get_default_analysis()
    
    def _create_analysis_prompt(self, content: str, company_name: str) -> str:
        """Create comprehensive analysis prompt for AI"""
        return f"""
You are an expert financial services strategy analyst. Analyze the following company documents for {company_name} and provide a comprehensive strategic archetype classification.

BUSINESS STRATEGY ARCHETYPES:
{json.dumps(self.business_archetypes, indent=2)}

RISK STRATEGY ARCHETYPES:
{json.dumps(self.risk_archetypes, indent=2)}

DOCUMENT CONTENT:
{content}

ANALYSIS REQUIREMENTS:

1. **Business Strategy Classification:**
   - Identify the PRIMARY business strategy archetype that best fits {company_name}
   - Identify a SECONDARY business strategy archetype 
   - Provide DETAILED EVIDENCE from the documents supporting each classification
   - Explain how the company's activities, focus areas, and strategic direction align with the chosen archetypes
   - Reference specific examples from the content (growth initiatives, market focus, operational approaches, etc.)

2. **Risk Strategy Classification:**
   - Identify the PRIMARY risk strategy archetype that best fits {company_name}
   - Identify a SECONDARY risk strategy archetype
   - Provide DETAILED EVIDENCE from the documents supporting each classification
   - Explain how the company's risk management approach, regulatory stance, and control frameworks align with the chosen archetypes
   - Reference specific examples from the content (risk policies, compliance approaches, capital management, etc.)

3. **Evidence-Based Reasoning:**
   - Quote specific phrases or sections from the documents that support your classifications
   - Explain how the company's stated vision, mission, and strategic objectives align with the archetypes
   - Identify key performance indicators, business metrics, or strategic initiatives that demonstrate the archetype behaviors
   - Consider the company's market positioning, customer focus, and competitive approach

4. **Comprehensive Output Format:**
   Provide detailed analysis in this structure:

**BUSINESS STRATEGY**
**[Primary Archetype Name]**
[2-3 sentences explaining what this archetype represents and why it fits {company_name}. Include specific evidence from documents such as strategic initiatives, market focus, growth approach, operational model, etc. Reference specific quotes or data points where possible.]

**RISK STRATEGY** 
**[Primary Archetype Name]**
[2-3 sentences explaining what this archetype represents and why it fits {company_name}. Include specific evidence from documents such as risk policies, compliance approach, capital management, regulatory engagement, control frameworks, etc. Reference specific quotes or data points where possible.]

**SUPPORTING EVIDENCE:**
- Primary Business Evidence: [Specific quotes/examples from documents]
- Secondary Business Evidence: [Additional supporting evidence]
- Primary Risk Evidence: [Specific quotes/examples from documents] 
- Secondary Risk Evidence: [Additional supporting evidence]

**CONFIDENCE ASSESSMENT:**
[High/Medium/Low] confidence based on clarity and consistency of evidence in the documents.

Format your response as JSON with keys: business_primary, business_secondary, business_reasoning, business_evidence, risk_primary, risk_secondary, risk_reasoning, risk_evidence, confidence, supporting_quotes.
"""
    
    def _parse_ai_response(self, response_text: str, client_type: str, extracted_content: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Parse comprehensive AI response into structured format"""
        try:
            # Clean the response text first
            cleaned_response = str(response_text).strip()
            
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_response, re.DOTALL)
            
            if json_match:
                try:
                    ai_data = json.loads(json_match.group())
                    
                    # Ensure ai_data is a dictionary
                    if not isinstance(ai_data, dict):
                        logger.warning("AI response is not a dictionary, using fallback")
                        return self._parse_text_response(cleaned_response, client_type, extracted_content)
                    
                    # Extract comprehensive analysis data
                    business_primary = str(ai_data.get('business_primary', 'Disciplined Specialist Growth'))
                    business_secondary = str(ai_data.get('business_secondary', 'Balance-Sheet Steward'))
                    risk_primary = str(ai_data.get('risk_primary', 'Risk-First Conservative'))
                    risk_secondary = str(ai_data.get('risk_secondary', 'Rules-Led Operator'))
                    
                    # Validate archetypes exist
                    if business_primary not in self.business_archetypes:
                        business_primary = 'Disciplined Specialist Growth'
                    if business_secondary not in self.business_archetypes:
                        business_secondary = 'Balance-Sheet Steward'
                    if risk_primary not in self.risk_archetypes:
                        risk_primary = 'Risk-First Conservative'
                    if risk_secondary not in self.risk_archetypes:
                        risk_secondary = 'Rules-Led Operator'
                    
                    # Extract comprehensive reasoning and evidence
                    business_reasoning = str(ai_data.get('business_reasoning', ''))
                    business_evidence = ai_data.get('business_evidence', [])
                    risk_reasoning = str(ai_data.get('risk_reasoning', ''))
                    risk_evidence = ai_data.get('risk_evidence', [])
                    supporting_quotes = ai_data.get('supporting_quotes', [])
                    
                    # Ensure evidence is in list format
                    if isinstance(business_evidence, str):
                        business_evidence = [business_evidence]
                    if isinstance(risk_evidence, str):
                        risk_evidence = [risk_evidence]
                    if isinstance(supporting_quotes, str):
                        supporting_quotes = [supporting_quotes]
                    
                    return {
                        'business_strategy_archetypes': {
                            'dominant': business_primary,
                            'secondary': business_secondary,
                            'reasoning': business_reasoning or f"AI analysis identifies {business_primary} as the primary business strategy based on comprehensive document review.",
                            'evidence': business_evidence,
                            'definition': self.business_archetypes[business_primary],
                            'secondary_definition': self.business_archetypes[business_secondary],
                            'comprehensive_analysis': self._format_business_analysis(business_primary, business_reasoning, business_evidence)
                        },
                        'risk_strategy_archetypes': {
                            'dominant': risk_primary,
                            'secondary': risk_secondary, 
                            'reasoning': risk_reasoning or f"AI analysis identifies {risk_primary} as the primary risk strategy based on comprehensive document review.",
                            'evidence': risk_evidence,
                            'definition': self.risk_archetypes[risk_primary],
                            'secondary_definition': self.risk_archetypes[risk_secondary],
                            'comprehensive_analysis': self._format_risk_analysis(risk_primary, risk_reasoning, risk_evidence)
                        },
                        'analysis_type': f'ai_{client_type}_comprehensive',
                        'confidence_level': str(ai_data.get('confidence', 'medium')),
                        'files_analyzed': len(extracted_content) if extracted_content else 1,
                        'supporting_quotes': supporting_quotes,
                        'ai_raw_response': cleaned_response[:300] + '...' if len(cleaned_response) > 300 else cleaned_response,
                        'archetype_definitions': {
                            'business_archetypes': self.business_archetypes,
                            'risk_archetypes': self.risk_archetypes
                        }
                    }
                    
                except json.JSONDecodeError as je:
                    logger.warning(f"JSON decode error: {je}, using text parsing")
                    return self._parse_text_response(cleaned_response, client_type, extracted_content)
            else:
                # No JSON found, use enhanced text parsing
                logger.info("No JSON structure found in AI response, using enhanced text parsing")
                return self._parse_text_response_comprehensive(cleaned_response, client_type, extracted_content)
                
        except Exception as e:
            logger.error(f"Error parsing comprehensive AI response: {e}")
            logger.info("Using fallback text parsing")
            return self._parse_text_response_comprehensive(response_text, client_type, extracted_content)
    
    def _format_business_analysis(self, archetype: str, reasoning: str, evidence: List[str]) -> str:
        """Format comprehensive business strategy analysis"""
        archetype_def = self.business_archetypes.get(archetype, "")
        
        formatted_analysis = f"**{archetype}**\n\n"
        
        if reasoning and len(reasoning) > 50:
            formatted_analysis += reasoning
        else:
            # Generate comprehensive reasoning if not provided
            formatted_analysis += f"The company demonstrates a {archetype} approach, characterized by {archetype_def.lower()}. "
            
            if evidence:
                formatted_analysis += "This is evidenced by "
                formatted_analysis += ", ".join(evidence[:3])  # Top 3 pieces of evidence
                formatted_analysis += "."
            else:
                formatted_analysis += "This strategic orientation is reflected in the company's operational focus and market positioning as described in the analyzed documents."
        
        return formatted_analysis
    
    def _format_risk_analysis(self, archetype: str, reasoning: str, evidence: List[str]) -> str:
        """Format comprehensive risk strategy analysis"""
        archetype_def = self.risk_archetypes.get(archetype, "")
        
        formatted_analysis = f"**{archetype}**\n\n"
        
        if reasoning and len(reasoning) > 50:
            formatted_analysis += reasoning
        else:
            # Generate comprehensive reasoning if not provided
            formatted_analysis += f"The company adopts a {archetype} risk management approach, which {archetype_def.lower()}. "
            
            if evidence:
                formatted_analysis += "This approach is demonstrated through "
                formatted_analysis += ", ".join(evidence[:3])  # Top 3 pieces of evidence
                formatted_analysis += "."
            else:
                formatted_analysis += "This risk strategy is evident in the company's governance frameworks and regulatory compliance approaches as detailed in the analyzed documents."
        
        return formatted_analysis

    def _parse_text_response_comprehensive(self, response_text: str, client_type: str, extracted_content: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Enhanced text parsing for comprehensive analysis"""
        try:
            # Analyze the full response text for archetype mentions and context
            response_lower = response_text.lower()
            
            # Find business archetype with context
            business_archetype = "Disciplined Specialist Growth"
            business_context = ""
            
            for archetype in self.business_archetypes.keys():
                if archetype.lower() in response_lower:
                    business_archetype = archetype
                    # Extract surrounding context (up to 200 characters around the mention)
                    import re
                    pattern = rf'.{{0,100}}{re.escape(archetype.lower())}.{{0,100}}'
                    match = re.search(pattern, response_lower, re.IGNORECASE)
                    if match:
                        business_context = match.group()
                    break
            
            # Find risk archetype with context
            risk_archetype = "Risk-First Conservative"
            risk_context = ""
            
            for archetype in self.risk_archetypes.keys():
                if archetype.lower() in response_lower:
                    risk_archetype = archetype
                    # Extract surrounding context
                    import re
                    pattern = rf'.{{0,100}}{re.escape(archetype.lower())}.{{0,100}}'
                    match = re.search(pattern, response_lower, re.IGNORECASE)
                    if match:
                        risk_context = match.group()
                    break
            
            # Extract key phrases that might be evidence
            evidence_phrases = self._extract_evidence_phrases(response_text)
            
            return {
                'business_strategy_archetypes': {
                    'dominant': business_archetype,
                    'secondary': 'Balance-Sheet Steward',
                    'reasoning': f'Analysis identifies {business_archetype} based on comprehensive document review. {business_context}',
                    'evidence': evidence_phrases.get('business', []),
                    'definition': self.business_archetypes.get(business_archetype, "Definition not available"),
                    'secondary_definition': self.business_archetypes.get('Balance-Sheet Steward', "Definition not available"),
                    'comprehensive_analysis': self._format_business_analysis(business_archetype, business_context, evidence_phrases.get('business', []))
                },
                'risk_strategy_archetypes': {
                    'dominant': risk_archetype,
                    'secondary': 'Rules-Led Operator',
                    'reasoning': f'Risk management approach identified as {risk_archetype} based on comprehensive document review. {risk_context}',
                    'evidence': evidence_phrases.get('risk', []),
                    'definition': self.risk_archetypes.get(risk_archetype, "Definition not available"),
                    'secondary_definition': self.risk_archetypes.get('Rules-Led Operator', "Definition not available"),
                    'comprehensive_analysis': self._format_risk_analysis(risk_archetype, risk_context, evidence_phrases.get('risk', []))
                },
                'analysis_type': f'ai_{client_type}_text_comprehensive',
                'confidence_level': 'medium',
                'files_analyzed': len(extracted_content) if extracted_content else 1,
                'supporting_quotes': evidence_phrases.get('quotes', []),
                'ai_raw_response': response_text[:300] + '...' if len(response_text) > 300 else response_text,
                'archetype_definitions': {
                    'business_archetypes': self.business_archetypes,
                    'risk_archetypes': self.risk_archetypes
                }
            }
            
        except Exception as e:
            logger.error(f"Enhanced text parsing failed: {e}")
            return self._get_default_analysis()

    def _extract_evidence_phrases(self, text: str) -> Dict[str, List[str]]:
        """Extract potential evidence phrases from AI response"""
        import re
        
        evidence = {
            'business': [],
            'risk': [],
            'quotes': []
        }
        
        try:
            # Look for quoted text
            quotes = re.findall(r'"([^"]*)"', text)
            evidence['quotes'] = [quote.strip() for quote in quotes if len(quote.strip()) > 10][:5]
            
            # Look for business-related evidence keywords
            business_keywords = ['growth', 'strategy', 'market', 'customer', 'product', 'innovation', 'efficiency', 'expansion']
            for keyword in business_keywords:
                pattern = rf'[^.]*{keyword}[^.]*\.'
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches[:2]:  # Limit to 2 per keyword
                    if len(match.strip()) > 20:
                        evidence['business'].append(match.strip())
            
            # Look for risk-related evidence keywords
            risk_keywords = ['risk', 'compliance', 'regulatory', 'capital', 'governance', 'control', 'resilience']
            for keyword in risk_keywords:
                pattern = rf'[^.]*{keyword}[^.]*\.'
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches[:2]:  # Limit to 2 per keyword
                    if len(match.strip()) > 20:
                        evidence['risk'].append(match.strip())
            
        except Exception as e:
            logger.debug(f"Error extracting evidence phrases: {e}")
        
        return evidence
    
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
        """Get secondary archetype - fixed to handle edge cases"""
        try:
            if not scores or not isinstance(scores, dict):
                return list(self.business_archetypes.keys())[1] if primary != list(self.business_archetypes.keys())[0] else list(self.business_archetypes.keys())[0]
            
            # Convert any non-string keys to strings
            clean_scores = {}
            for key, value in scores.items():
                clean_key = str(key) if key is not None else "Unknown"
                clean_value = float(value) if isinstance(value, (int, float)) else 0.0
                clean_scores[clean_key] = clean_value
            
            # Sort by score, descending
            sorted_scores = sorted(clean_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Find first archetype that's not the primary
            for archetype, score in sorted_scores:
                if str(archetype) != str(primary):
                    return str(archetype)
            
            # Fallback - return first available archetype
            all_archetypes = list(self.business_archetypes.keys()) + list(self.risk_archetypes.keys())
            for archetype in all_archetypes:
                if str(archetype) != str(primary):
                    return str(archetype)
            
            # Ultimate fallback
            return "Balance-Sheet Steward" if primary != "Balance-Sheet Steward" else "Disciplined Specialist Growth"
            
        except Exception as e:
            logger.error(f"Error in _get_secondary: {e}")
            # Safe fallback
            return "Balance-Sheet Steward" if str(primary) != "Balance-Sheet Steward" else "Disciplined Specialist Growth"
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Get comprehensive default analysis when all else fails"""
        return {
            'business_strategy_archetypes': {
                'dominant': 'Disciplined Specialist Growth',
                'secondary': 'Balance-Sheet Steward',
                'reasoning': 'Default analysis applied due to insufficient data for detailed classification. This represents a conservative assessment based on typical financial services industry patterns.',
                'evidence': ['Limited document content available for comprehensive analysis', 'Default classification based on conservative industry standards'],
                'definition': self.business_archetypes.get('Disciplined Specialist Growth', "Niche focus with strong underwriting edge; grows opportunistically while recycling balance-sheet"),
                'secondary_definition': self.business_archetypes.get('Balance-Sheet Steward', "Low-risk appetite, prioritises capital strength and membership value"),
                'comprehensive_analysis': "**Disciplined Specialist Growth**\n\nDefault classification indicates a conservative approach focused on niche lending with strong underwriting capabilities. This represents a prudent business model that emphasizes sustainable growth while maintaining balance sheet discipline."
            },
            'risk_strategy_archetypes': {
                'dominant': 'Risk-First Conservative',
                'secondary': 'Rules-Led Operator',
                'reasoning': 'Default risk assessment assumes conservative approach prioritizing regulatory compliance and capital preservation. This reflects standard industry prudent risk management practices.',
                'evidence': ['Conservative risk appetite assumed based on industry standards', 'Regulatory compliance focus typical for financial services'],
                'definition': self.risk_archetypes.get('Risk-First Conservative', "Prioritises capital preservation and regulatory compliance; growth is secondary to resilience"),
                'secondary_definition': self.risk_archetypes.get('Rules-Led Operator', "Strict adherence to rules and checklists; prioritises control consistency over judgment or speed"),
                'comprehensive_analysis': "**Risk-First Conservative**\n\nDefault classification emphasizes capital preservation and regulatory compliance as primary risk management objectives. This approach prioritizes resilience and stability over aggressive growth strategies, reflecting prudent risk governance typical of established financial services institutions."
            },
            'analysis_type': 'default_comprehensive',
            'confidence_level': 'low',
            'files_analyzed': 0,
            'supporting_quotes': [],
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