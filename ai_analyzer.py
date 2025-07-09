#!/usr/bin/env python3
"""
Enhanced AI Archetype Analyzer for Board-Level Strategic Analysis
Delivers McKinsey/BCG-grade strategic insights for executive decision making
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
import time
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class ExecutiveAIAnalyzer:
    """
    Executive-grade AI analyzer delivering board-ready strategic insights
    Focused on actionable intelligence and strategic implications
    """
    
    def __init__(self):
        """Initialize with enterprise-grade configuration"""
        self.client_type = "fallback"
        self.openai_client = None
        
        logger.info("🏛️ Executive AI Analyzer v3.0 - Board-Grade Analysis Engine")
        
        # Initialize AI providers
        self._init_openai()
        
        # Executive-focused archetype definitions with strategic context
        self.business_archetypes = {
            "Scale-through-Distribution": {
                "definition": "Gains market share primarily by expanding distribution channels and partnerships faster than operational maturity develops",
                "strategic_context": "High-velocity expansion strategy with emphasis on market capture over operational excellence",
                "board_implications": "Requires significant investment in channel management and partner oversight capabilities",
                "key_metrics": ["Channel ROI", "Partner acquisition rate", "Market penetration speed"]
            },
            "Land-Grab Platform": {
                "definition": "Uses aggressive below-market pricing or incentives to rapidly build large multi-sided platforms",
                "strategic_context": "Market dominance strategy accepting short-term losses for long-term platform value",
                "board_implications": "High cash burn with delayed profitability; requires patient capital and clear path to monetization",
                "key_metrics": ["Platform adoption rate", "Network effects strength", "Unit economics trajectory"]
            },
            "Asset-Velocity Maximiser": {
                "definition": "Prioritizes rapid asset origination and turnover, often accepting higher funding costs for speed",
                "strategic_context": "Transaction-focused model optimizing for volume and velocity over margin per transaction",
                "board_implications": "Requires sophisticated risk management and operational scaling capabilities",
                "key_metrics": ["Asset turnover ratio", "Origination speed", "Capital velocity"]
            },
            "Yield-Hunting": {
                "definition": "Focuses on high-margin segments and prices aggressively for risk premium",
                "strategic_context": "Premium pricing strategy targeting underserved or higher-risk market segments",
                "board_implications": "Requires exceptional credit assessment and pricing capabilities; cyclical earnings exposure",
                "key_metrics": ["Risk-adjusted returns", "Credit loss rates", "Pricing power"]
            },
            "Fee-Extraction Engine": {
                "definition": "Derives majority of profits from ancillary fees, add-ons, and cross-selling rather than core products",
                "strategic_context": "Revenue diversification through service monetization and customer lifecycle optimization",
                "board_implications": "Regulatory scrutiny risk; requires transparent customer communication and value delivery",
                "key_metrics": ["Fee income ratio", "Cross-sell success rate", "Customer lifetime value"]
            },
            "Disciplined Specialist Growth": {
                "definition": "Maintains niche focus with strong underwriting capabilities, growing opportunistically while optimizing balance sheet efficiency",
                "strategic_context": "Conservative growth strategy emphasizing expertise depth over market breadth",
                "board_implications": "Sustainable competitive advantage through specialization; limited addressable market size",
                "key_metrics": ["ROE consistency", "Market share in niche", "Credit quality metrics"]
            },
            "Expert Niche Leader": {
                "definition": "Develops deep expertise in micro-segments with modest but highly stable growth",
                "strategic_context": "Expertise-based competitive moat with premium pricing power in specialized markets",
                "board_implications": "Resilient to competition but limited scalability; succession planning critical",
                "key_metrics": ["Expert reputation measures", "Client retention rates", "Premium pricing sustainability"]
            },
            "Service-Driven Differentiator": {
                "definition": "Competes on superior client experience and advisory capability rather than price or scale",
                "strategic_context": "Relationship-centric model with emphasis on customer satisfaction and loyalty",
                "board_implications": "Higher cost base offset by premium pricing and lower churn; talent-dependent",
                "key_metrics": ["Net Promoter Score", "Client retention", "Service premium captured"]
            },
            "Cost-Leadership Operator": {
                "definition": "Achieves competitive advantage through lean operations, digital automation, and zero-based cost management",
                "strategic_context": "Efficiency-driven strategy enabling competitive pricing while maintaining margins",
                "board_implications": "Requires continuous operational improvement and technology investment",
                "key_metrics": ["Cost-to-income ratio", "Digital adoption rates", "Process automation levels"]
            },
            "Tech-Productivity Accelerator": {
                "definition": "Leverages heavy automation and AI to compress unit costs and redeploy human capital to higher-value activities",
                "strategic_context": "Technology-first approach to operational leverage and competitive differentiation",
                "board_implications": "Significant upfront technology investment; organizational change management needs",
                "key_metrics": ["Technology ROI", "Productivity per employee", "Automation penetration"]
            },
            "Product-Innovation Flywheel": {
                "definition": "Maintains competitive advantage through continuous launch of novel product variants and features",
                "strategic_context": "Innovation-driven growth with emphasis on first-mover advantages and market disruption",
                "board_implications": "High R&D investment; requires agile development capabilities and risk tolerance",
                "key_metrics": ["Innovation pipeline value", "Time-to-market", "New product revenue contribution"]
            },
            "Data-Monetisation Pioneer": {
                "definition": "Converts proprietary data assets into revenue streams through analytics and insights platforms",
                "strategic_context": "Data-as-a-service strategy leveraging information advantages for competitive differentiation",
                "board_implications": "Requires data governance capabilities and privacy compliance; regulatory considerations",
                "key_metrics": ["Data revenue streams", "Analytics adoption", "Data asset valuation"]
            },
            "Balance-Sheet Steward": {
                "definition": "Prioritizes capital strength and stakeholder value over aggressive growth",
                "strategic_context": "Conservative approach emphasizing financial stability and long-term sustainability",
                "board_implications": "Lower growth but higher resilience; member/stakeholder value focus over shareholder returns",
                "key_metrics": ["Capital ratios", "Dividend sustainability", "Financial stability ratings"]
            },
            "Regulatory Shelter Occupant": {
                "definition": "Leverages regulatory protections or franchise advantages to defend market position",
                "strategic_context": "Protected market strategy with emphasis on regulatory compliance and relationship management",
                "board_implications": "Regulatory dependency risk; limited competitive pressure but potential for disruption",
                "key_metrics": ["Regulatory relationship quality", "Market share stability", "Franchise value"]
            },
            "Regulator-Mandated Remediation": {
                "definition": "Operating under regulatory constraints with resources focused on compliance and historical issue resolution",
                "strategic_context": "Turnaround situation with regulatory oversight limiting strategic options",
                "board_implications": "Limited growth opportunities; significant compliance costs; reputational recovery needed",
                "key_metrics": ["Remediation progress", "Regulatory milestone completion", "Compliance cost trends"]
            },
            "Wind-down / Run-off": {
                "definition": "Managing existing portfolio to maturity or sale with minimal new business origination",
                "strategic_context": "Portfolio optimization strategy focused on value extraction from legacy assets",
                "board_implications": "Limited strategic options; focus on cash generation and stakeholder protection",
                "key_metrics": ["Portfolio runoff rate", "Legacy value realization", "Wind-down costs"]
            },
            "Strategic Withdrawal": {
                "definition": "Actively divesting business lines or geographies to refocus on core franchise strengths",
                "strategic_context": "Portfolio rationalization strategy to improve focus and resource allocation",
                "board_implications": "One-time costs and potential value destruction; requires clear strategic vision for remaining business",
                "key_metrics": ["Divestiture values realized", "Core business performance", "Strategic focus measures"]
            },
            "Distressed-Asset Harvester": {
                "definition": "Acquires undervalued or distressed assets during market downturns for future value realization",
                "strategic_context": "Counter-cyclical investment strategy requiring specialized workout capabilities",
                "board_implications": "Requires patient capital and specialized expertise; timing and market cycle dependency",
                "key_metrics": ["Asset acquisition returns", "Workout success rates", "Market timing effectiveness"]
            },
            "Counter-Cyclical Capitaliser": {
                "definition": "Expands lending and investment precisely when competitors retreat, using superior liquidity position",
                "strategic_context": "Opportunistic growth strategy leveraging market dislocations for competitive advantage",
                "board_implications": "Requires strong balance sheet and contrarian investment philosophy; market timing risk",
                "key_metrics": ["Counter-cyclical deployment effectiveness", "Market share gains", "Liquidity strength"]
            }
        }
        
        self.risk_archetypes = {
            "Risk-First Conservative": {
                "definition": "Prioritizes capital preservation and regulatory compliance above growth opportunities",
                "strategic_context": "Defensive risk strategy emphasizing stability and regulatory relationship quality",
                "board_implications": "Lower returns but higher predictability; strong regulatory standing; limited growth flexibility",
                "key_metrics": ["Capital buffer levels", "Regulatory rating scores", "Risk-adjusted returns"]
            },
            "Rules-Led Operator": {
                "definition": "Emphasizes strict procedural adherence and control consistency over business judgment",
                "strategic_context": "Process-driven risk management with emphasis on consistency and auditability",
                "board_implications": "Reduced operational risk but potential for missed opportunities; requires strong process governance",
                "key_metrics": ["Process compliance rates", "Control effectiveness scores", "Operational risk incidents"]
            },
            "Resilience-Focused Architect": {
                "definition": "Designs operations for crisis endurance through comprehensive stress testing and scenario planning",
                "strategic_context": "Future-proofing strategy with emphasis on operational continuity and shock absorption",
                "board_implications": "Higher operational costs but superior crisis performance; competitive advantage in stressed markets",
                "key_metrics": ["Stress test performance", "Business continuity effectiveness", "Crisis response capabilities"]
            },
            "Strategic Risk-Taker": {
                "definition": "Accepts elevated risk exposure to unlock growth opportunities, using sophisticated risk management to offset exposure",
                "strategic_context": "Calculated risk strategy balancing growth ambition with risk management sophistication",
                "board_implications": "Higher potential returns with increased volatility; requires sophisticated risk infrastructure",
                "key_metrics": ["Risk-adjusted returns", "Risk management sophistication", "Growth from risk-taking"]
            },
            "Control-Lag Follower": {
                "definition": "Expands products or markets ahead of control maturity, managing risks reactively",
                "strategic_context": "Growth-first approach with risk management following business expansion",
                "board_implications": "Higher growth potential but elevated operational and compliance risks; requires rapid control development",
                "key_metrics": ["Growth versus control maturity gap", "Post-expansion risk incidents", "Control catch-up effectiveness"]
            },
            "Reactive Remediator": {
                "definition": "Risk strategy shaped by external events, regulatory findings, or audit discoveries",
                "strategic_context": "Event-driven risk management with limited proactive strategic planning",
                "board_implications": "Higher compliance costs and regulatory scrutiny; unpredictable risk management effectiveness",
                "key_metrics": ["Regulatory finding trends", "Remediation completion rates", "Proactive versus reactive risk actions"]
            },
            "Reputation-First Shield": {
                "definition": "Actively avoids reputational or political risks, sometimes at the expense of commercial logic",
                "strategic_context": "Stakeholder perception management prioritized over pure financial optimization",
                "board_implications": "Reduced reputational risk but potential opportunity costs; strong stakeholder relationships",
                "key_metrics": ["Reputational risk incidents", "Stakeholder satisfaction scores", "Political relationship quality"]
            },
            "Embedded Risk Partner": {
                "definition": "Integrates risk teams into frontline business decisions with collaborative risk appetite setting",
                "strategic_context": "Partnership-based risk management with business-risk team collaboration",
                "board_implications": "Better risk-return optimization but requires cultural alignment and strong risk talent",
                "key_metrics": ["Business-risk collaboration effectiveness", "Risk-adjusted decision quality", "Risk culture measures"]
            },
            "Quant-Control Enthusiast": {
                "definition": "Leverages advanced analytics, automation, and predictive modeling as primary risk management tools",
                "strategic_context": "Technology-driven risk management with emphasis on data-driven decision making",
                "board_implications": "Superior risk insights but technology dependency; requires significant analytical investment",
                "key_metrics": ["Predictive model accuracy", "Risk analytics ROI", "Technology-driven risk reduction"]
            },
            "Tick-Box Minimalist": {
                "definition": "Maintains superficial control structures primarily for regulatory compliance optics",
                "strategic_context": "Compliance-focused approach with limited genuine risk management intent",
                "board_implications": "Regulatory compliance but limited risk management effectiveness; potential for unexpected losses",
                "key_metrics": ["Regulatory compliance scores", "Risk management effectiveness gaps", "Unexpected loss frequency"]
            },
            "Mission-Driven Prudence": {
                "definition": "Anchors risk appetite in stakeholder protection and long-term social license considerations",
                "strategic_context": "Purpose-driven risk management balancing commercial objectives with stakeholder welfare",
                "board_implications": "Strong stakeholder trust but potential commercial constraints; long-term sustainability focus",
                "key_metrics": ["Stakeholder trust measures", "Social impact metrics", "Long-term value creation"]
            }
        }
        
        logger.info(f"✅ Executive AI Analyzer v3.0 initialized. Analysis engine: {self.client_type}")
    
    def _init_openai(self):
        """Initialize OpenAI with enhanced error handling"""
        try:
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                logger.warning("⚠️ OpenAI API key not found - using enhanced fallback analysis")
                return
                
            import openai
            logger.info(f"OpenAI version: {openai.__version__}")
            
            # Configure for v0.28.x
            openai.api_key = api_key
            self.openai_client = openai
            self.client_type = "openai_executive"
            logger.info("✅ OpenAI configured for executive-grade analysis")
                    
        except Exception as e:
            logger.warning(f"OpenAI setup failed: {e}")
            logger.info("Enhanced fallback analysis will be used")
    
    def analyze_for_board(self, content: str, company_name: str, company_number: str, 
                         extracted_content: Optional[List[Dict[str, Any]]] = None,
                         analysis_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Executive-grade archetype analysis for board presentation
        
        Args:
            content: Document content for analysis
            company_name: Company name
            company_number: Company registration number
            extracted_content: Detailed file extraction data
            analysis_context: Additional context (e.g., "annual strategic review", "acquisition due diligence")
            
        Returns:
            Board-ready strategic analysis with actionable insights
        """
        start_time = time.time()
        
        try:
            logger.info(f"🎯 Starting executive analysis for {company_name}")
            
            if self.client_type == "openai_executive":
                analysis = self._executive_ai_analysis(content, company_name, company_number, extracted_content, analysis_context)
            else:
                analysis = self._executive_fallback_analysis(content, company_name, company_number, extracted_content, analysis_context)
            
            # Enhance with executive summary and strategic implications
            enhanced_analysis = self._create_board_ready_output(analysis, company_name, analysis_context)
            
            analysis_time = time.time() - start_time
            logger.info(f"✅ Executive analysis completed in {analysis_time:.2f}s")
            
            return enhanced_analysis
            
        except Exception as e:
            logger.error(f"Executive analysis failed: {e}")
            return self._create_emergency_board_analysis(company_name, company_number, str(e))
    
    def _executive_ai_analysis(self, content: str, company_name: str, company_number: str,
                              extracted_content: Optional[List[Dict[str, Any]]], 
                              analysis_context: Optional[str]) -> Dict[str, Any]:
        """AI-powered executive analysis with board-grade insights"""
        try:
            # Prepare content for analysis (focused extraction)
            analysis_content = self._prepare_executive_content(content, company_name)
            
            # Create executive-grade prompt
            prompt = self._create_executive_prompt(analysis_content, company_name, analysis_context)
            
            # Execute AI analysis with retry logic
            response = self._execute_ai_analysis(prompt)
            
            # Parse and validate response
            parsed_analysis = self._parse_executive_response(response, extracted_content)
            
            return parsed_analysis
            
        except Exception as e:
            logger.error(f"AI executive analysis failed: {e}")
            return self._executive_fallback_analysis(content, company_name, company_number, extracted_content, analysis_context)
    
    def _create_executive_prompt(self, content: str, company_name: str, analysis_context: Optional[str]) -> str:
        """Create board-grade analysis prompt for AI"""
        context_note = f"\n\nANALYSIS CONTEXT: {analysis_context}" if analysis_context else ""
        
        return f"""
You are a McKinsey/BCG Principal conducting a strategic analysis of {company_name} for board presentation.

CRITICAL REQUIREMENTS:
1. Extract SPECIFIC EVIDENCE from documents - direct quotes that boards will find credible
2. Focus on STRATEGIC IMPLICATIONS - what this means for competitive position and future performance
3. Identify KEY RISKS AND OPPORTUNITIES - actionable insights for strategic decision making
4. Provide QUANTIFIABLE METRICS where possible - boards need measurable indicators
5. Consider REGULATORY AND MARKET CONTEXT - external factors affecting strategy

BUSINESS STRATEGY ARCHETYPES:
{self._format_archetypes_for_prompt(self.business_archetypes)}

RISK STRATEGY ARCHETYPES:
{self._format_archetypes_for_prompt(self.risk_archetypes)}

COMPANY DOCUMENTS FOR ANALYSIS:
{content}{context_note}

REQUIRED OUTPUT FORMAT (JSON):
{{
  "executive_summary": "2-3 sentence strategic assessment for board consumption",
  "business_strategy": {{
    "primary_archetype": "[exact archetype name]",
    "secondary_archetype": "[exact archetype name]",
    "strategic_rationale": "Board-level explanation with specific evidence from documents",
    "competitive_implications": "What this strategy means for market position and competitive advantage",
    "evidence_quotes": ["Direct quote 1", "Direct quote 2", "Direct quote 3"],
    "key_metrics": ["Metric 1", "Metric 2", "Metric 3"],
    "strategic_risks": ["Risk 1", "Risk 2"],
    "strategic_opportunities": ["Opportunity 1", "Opportunity 2"]
  }},
  "risk_strategy": {{
    "primary_archetype": "[exact archetype name]", 
    "secondary_archetype": "[exact archetype name]",
    "risk_rationale": "Board-level explanation with specific evidence from documents",
    "governance_implications": "What this approach means for risk governance and oversight",
    "evidence_quotes": ["Direct quote 1", "Direct quote 2"],
    "key_risk_metrics": ["Risk metric 1", "Risk metric 2"],
    "risk_concerns": ["Concern 1", "Concern 2"],
    "risk_strengths": ["Strength 1", "Strength 2"]
  }},
  "strategic_recommendations": [
    "Actionable recommendation 1 for board consideration",
    "Actionable recommendation 2 for board consideration"
  ],
  "confidence_assessment": "high/medium/low",
  "critical_information_gaps": ["Gap 1", "Gap 2"]
}}

ANALYSIS STANDARDS:
- Use consultant-grade language appropriate for board presentations
- Focus on strategic materiality - what matters most for company direction
- Provide actionable insights that inform decision making
- Support all assessments with specific documentary evidence
- Consider both current position and strategic trajectory
"""
    
    def _format_archetypes_for_prompt(self, archetypes: Dict[str, Dict[str, Any]]) -> str:
        """Format archetype definitions for AI prompt"""
        formatted = ""
        for name, details in archetypes.items():
            formatted += f"\n- {name}: {details['definition']}\n  Strategic Context: {details['strategic_context']}\n"
        return formatted
    
    def _prepare_executive_content(self, content: str, company_name: str) -> str:
        """Prepare content for executive analysis with key section extraction"""
        if not content:
            return f"Limited content available for {company_name} analysis."
        
        # Extract key sections for strategic analysis
        strategic_sections = self._extract_strategic_sections(content)
        
        # Limit content while preserving most strategic value
        if len(content) > 20000:
            # Prioritize strategic content
            priority_content = "\n\n".join([
                strategic_sections.get('strategy', '')[:3000],
                strategic_sections.get('business_model', '')[:2000], 
                strategic_sections.get('risk_management', '')[:2000],
                strategic_sections.get('financial_highlights', '')[:1500],
                strategic_sections.get('governance', '')[:1500]
            ])
            return priority_content if priority_content.strip() else content[:15000]
        
        return content
    
    def _extract_strategic_sections(self, content: str) -> Dict[str, str]:
        """Extract key strategic sections from content"""
        sections = {
            'strategy': '',
            'business_model': '',
            'risk_management': '',
            'financial_highlights': '',
            'governance': ''
        }
        
        content_lower = content.lower()
        
        # Strategy indicators
        strategy_keywords = ['strategy', 'strategic', 'vision', 'mission', 'objective', 'goal', 'plan']
        business_keywords = ['business model', 'revenue', 'market', 'customer', 'product', 'service']
        risk_keywords = ['risk', 'compliance', 'regulatory', 'governance', 'control']
        financial_keywords = ['financial', 'profit', 'revenue', 'capital', 'asset', 'income']
        
        # Extract relevant paragraphs
        paragraphs = content.split('\n\n')
        
        for paragraph in paragraphs:
            para_lower = paragraph.lower()
            
            if any(keyword in para_lower for keyword in strategy_keywords):
                sections['strategy'] += paragraph + '\n\n'
            elif any(keyword in para_lower for keyword in business_keywords):
                sections['business_model'] += paragraph + '\n\n'
            elif any(keyword in para_lower for keyword in risk_keywords):
                sections['risk_management'] += paragraph + '\n\n'
            elif any(keyword in para_lower for keyword in financial_keywords):
                sections['financial_highlights'] += paragraph + '\n\n'
        
        return sections
    
    def _execute_ai_analysis(self, prompt: str, max_retries: int = 3) -> str:
        """Execute AI analysis with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.openai_client.ChatCompletion.create(
                    model="gpt-4",  # Use GPT-4 for executive analysis
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a senior strategy consultant delivering board-grade analysis. Focus on strategic implications and actionable insights."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,  # Lower temperature for consistent executive output
                    max_tokens=2000,
                    top_p=0.9
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"AI analysis attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)  # Brief pause before retry
        
        raise Exception("AI analysis failed after all retries")
    
    def _parse_executive_response(self, response_text: str, extracted_content: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Parse AI response into executive analysis structure"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                try:
                    ai_data = json.loads(json_match.group())
                    
                    # Validate and structure executive analysis
                    return self._structure_executive_analysis(ai_data, extracted_content, response_text)
                    
                except json.JSONDecodeError:
                    logger.warning("JSON parsing failed, using enhanced text parsing")
                    return self._parse_executive_text(response_text, extracted_content)
            else:
                return self._parse_executive_text(response_text, extracted_content)
                
        except Exception as e:
            logger.error(f"Executive response parsing failed: {e}")
            raise e
    
    def _structure_executive_analysis(self, ai_data: Dict[str, Any], 
                                    extracted_content: Optional[List[Dict[str, Any]]],
                                    raw_response: str) -> Dict[str, Any]:
        """Structure AI data into executive analysis format"""
        
        # Extract business strategy analysis
        business_data = ai_data.get('business_strategy', {})
        business_primary = business_data.get('primary_archetype', 'Disciplined Specialist Growth')
        business_secondary = business_data.get('secondary_archetype', 'Balance-Sheet Steward')
        
        # Extract risk strategy analysis  
        risk_data = ai_data.get('risk_strategy', {})
        risk_primary = risk_data.get('primary_archetype', 'Risk-First Conservative')
        risk_secondary = risk_data.get('secondary_archetype', 'Rules-Led Operator')
        
        # Validate archetypes
        business_primary = self._validate_archetype(business_primary, self.business_archetypes)
        business_secondary = self._validate_archetype(business_secondary, self.business_archetypes)
        risk_primary = self._validate_archetype(risk_primary, self.risk_archetypes)
        risk_secondary = self._validate_archetype(risk_secondary, self.risk_archetypes)
        
        return {
            'executive_summary': ai_data.get('executive_summary', 'Strategic analysis completed with archetype classification'),
            'business_strategy_analysis': {
                'dominant_archetype': business_primary,
                'secondary_archetype': business_secondary,
                'strategic_rationale': business_data.get('strategic_rationale', ''),
                'competitive_implications': business_data.get('competitive_implications', ''),
                'evidence_quotes': business_data.get('evidence_quotes', []),
                'key_metrics': business_data.get('key_metrics', []),
                'strategic_risks': business_data.get('strategic_risks', []),
                'strategic_opportunities': business_data.get('strategic_opportunities', []),
                'archetype_definition': self.business_archetypes[business_primary]['definition'],
                'strategic_context': self.business_archetypes[business_primary]['strategic_context'],
                'board_implications': self.business_archetypes[business_primary]['board_implications']
            },
            'risk_strategy_analysis': {
                'dominant_archetype': risk_primary,
                'secondary_archetype': risk_secondary,
                'risk_rationale': risk_data.get('risk_rationale', ''),
                'governance_implications': risk_data.get('governance_implications', ''),
                'evidence_quotes': risk_data.get('evidence_quotes', []),
                'key_risk_metrics': risk_data.get('key_risk_metrics', []),
                'risk_concerns': risk_data.get('risk_concerns', []),
                'risk_strengths': risk_data.get('risk_strengths', []),
                'archetype_definition': self.risk_archetypes[risk_primary]['definition'],
                'strategic_context': self.risk_archetypes[risk_primary]['strategic_context'],
                'board_implications': self.risk_archetypes[risk_primary]['board_implications']
            },
            'strategic_recommendations': ai_data.get('strategic_recommendations', []),
            'analysis_metadata': {
                'analysis_type': 'ai_executive_comprehensive',
                'confidence_level': ai_data.get('confidence_assessment', 'medium'),
                'files_analyzed': len(extracted_content) if extracted_content else 1,
                'critical_information_gaps': ai_data.get('critical_information_gaps', []),
                'analysis_timestamp': datetime.now().isoformat(),
                'raw_ai_response': raw_response[:500] + '...' if len(raw_response) > 500 else raw_response
            }
        }
    
    def _validate_archetype(self, archetype: str, archetype_dict: Dict[str, Dict[str, Any]]) -> str:
        """Validate archetype exists, return default if not"""
        if archetype in archetype_dict:
            return archetype
        
        # Try case-insensitive match
        for key in archetype_dict.keys():
            if key.lower() == archetype.lower():
                return key
        
        # Return default
        return list(archetype_dict.keys())[0]
    
    def _parse_executive_text(self, response_text: str, extracted_content: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Enhanced text parsing for executive insights when JSON fails"""
        
        # Extract key insights from unstructured response
        insights = self._extract_executive_insights(response_text)
        
        # Determine archetypes from text
        business_archetype = self._identify_archetype_from_text(response_text, self.business_archetypes)
        risk_archetype = self._identify_archetype_from_text(response_text, self.risk_archetypes)
        
        return {
            'executive_summary': insights.get('summary', 'Strategic analysis completed based on document review'),
            'business_strategy_analysis': {
                'dominant_archetype': business_archetype,
                'secondary_archetype': 'Balance-Sheet Steward',
                'strategic_rationale': insights.get('business_rationale', f'Analysis indicates {business_archetype} strategic orientation'),
                'competitive_implications': insights.get('competitive_context', 'Strategic positioning requires further analysis'),
                'evidence_quotes': insights.get('quotes', [])[:3],
                'key_metrics': self.business_archetypes[business_archetype]['key_metrics'],
                'strategic_risks': insights.get('risks', [])[:2],
                'strategic_opportunities': insights.get('opportunities', [])[:2],
                'archetype_definition': self.business_archetypes[business_archetype]['definition'],
                'strategic_context': self.business_archetypes[business_archetype]['strategic_context'],
                'board_implications': self.business_archetypes[business_archetype]['board_implications']
            },
            'risk_strategy_analysis': {
                'dominant_archetype': risk_archetype,
                'secondary_archetype': 'Rules-Led Operator',
                'risk_rationale': insights.get('risk_rationale', f'Risk management approach aligns with {risk_archetype}'),
                'governance_implications': insights.get('governance_context', 'Risk governance framework requires board oversight'),
                'evidence_quotes': insights.get('risk_quotes', [])[:2],
                'key_risk_metrics': self.risk_archetypes[risk_archetype]['key_metrics'],
                'risk_concerns': insights.get('risk_concerns', ['Regulatory compliance', 'Operational risk']),
                'risk_strengths': insights.get('risk_strengths', ['Established controls', 'Compliance focus']),
                'archetype_definition': self.risk_archetypes[risk_archetype]['definition'],
                'strategic_context': self.risk_archetypes[risk_archetype]['strategic_context'],
                'board_implications': self.risk_archetypes[risk_archetype]['board_implications']
            },
            'strategic_recommendations': insights.get('recommendations', ['Conduct detailed strategic review', 'Enhance risk monitoring capabilities']),
            'analysis_metadata': {
                'analysis_type': 'ai_executive_text_parsed',
                'confidence_level': 'medium',
                'files_analyzed': len(extracted_content) if extracted_content else 1,
                'critical_information_gaps': ['Detailed financial metrics needed', 'Strategic plan documentation required'],
                'analysis_timestamp': datetime.now().isoformat(),
                'raw_ai_response': response_text[:500] + '...' if len(response_text) > 500 else response_text
            }
        }
    
    def _extract_executive_insights(self, text: str) -> Dict[str, Any]:
        """Extract executive insights from unstructured AI response"""
        insights = {
            'summary': '',
            'business_rationale': '',
            'competitive_context': '',
            'risk_rationale': '',
            'governance_context': '',
            'quotes': [],
            'risks': [],
            'opportunities': [],
            'recommendations': []
        }
        
        # Extract quoted content
        quotes = re.findall(r'"([^"]*)"', text)
        insights['quotes'] = [q.strip() for q in quotes if len(q.strip()) > 15]
        
        # Extract key sentences
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            
            if any(word in sentence_lower for word in ['strategy', 'strategic', 'competitive']):
                if not insights['business_rationale']:
                    insights['business_rationale'] = sentence.strip()
            elif any(word in sentence_lower for word in ['risk', 'governance', 'compliance']):
                if not insights['risk_rationale']:
                    insights['risk_rationale'] = sentence.strip()
            elif any(word in sentence_lower for word in ['recommend', 'should', 'suggest']):
                insights['recommendations'].append(sentence.strip())
            elif any(word in sentence_lower for word in ['opportunity', 'potential', 'growth']):
                insights['opportunities'].append(sentence.strip())
            elif any(word in sentence_lower for word in ['risk', 'concern', 'challenge']):
                insights['risks'].append(sentence.strip())
        
        # Create executive summary from first substantive paragraph
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            if len(para.strip()) > 100 and not para.strip().startswith('{'):
                insights['summary'] = para.strip()[:200] + '...' if len(para) > 200 else para.strip()
                break
        
        return insights
    
    def _identify_archetype_from_text(self, text: str, archetypes: Dict[str, Dict[str, Any]]) -> str:
        """Identify archetype from text analysis"""
        text_lower = text.lower()
        scores = {}
        
        for archetype_name, archetype_data in archetypes.items():
            score = 0
            
            # Check for direct mentions
            if archetype_name.lower() in text_lower:
                score += 10
            
            # Check definition keywords
            definition_words = archetype_data['definition'].lower().split()
            for word in definition_words:
                if len(word) > 4 and word in text_lower:
                    score += 1
            
            # Check strategic context keywords
            context_words = archetype_data['strategic_context'].lower().split()
            for word in context_words:
                if len(word) > 4 and word in text_lower:
                    score += 1
            
            scores[archetype_name] = score
        
        # Return highest scoring archetype
        if scores:
            return max(scores, key=scores.get)
        
        return list(archetypes.keys())[0]  # Default fallback
    
    def _executive_fallback_analysis(self, content: str, company_name: str, company_number: str,
                                   extracted_content: Optional[List[Dict[str, Any]]],
                                   analysis_context: Optional[str]) -> Dict[str, Any]:
        """Executive-grade fallback analysis using enhanced pattern recognition"""
        
        logger.info("Using enhanced executive fallback analysis")
        
        # Perform sophisticated content analysis
        content_insights = self._analyze_content_patterns(content, company_name)
        
        # Determine archetypes using executive logic
        business_archetype = self._determine_business_archetype_executive(content_insights)
        risk_archetype = self._determine_risk_archetype_executive(content_insights)
        
        # Create executive-grade analysis
        return {
            'executive_summary': f'{company_name} demonstrates {business_archetype} strategic orientation with {risk_archetype} risk management approach, based on comprehensive document analysis.',
            'business_strategy_analysis': {
                'dominant_archetype': business_archetype,
                'secondary_archetype': self._get_secondary_archetype(business_archetype, self.business_archetypes),
                'strategic_rationale': content_insights['business_rationale'],
                'competitive_implications': content_insights['competitive_implications'],
                'evidence_quotes': content_insights['evidence_quotes'][:3],
                'key_metrics': self.business_archetypes[business_archetype]['key_metrics'],
                'strategic_risks': content_insights['strategic_risks'],
                'strategic_opportunities': content_insights['strategic_opportunities'],
                'archetype_definition': self.business_archetypes[business_archetype]['definition'],
                'strategic_context': self.business_archetypes[business_archetype]['strategic_context'],
                'board_implications': self.business_archetypes[business_archetype]['board_implications']
            },
            'risk_strategy_analysis': {
                'dominant_archetype': risk_archetype,
                'secondary_archetype': self._get_secondary_archetype(risk_archetype, self.risk_archetypes),
                'risk_rationale': content_insights['risk_rationale'],
                'governance_implications': content_insights['governance_implications'],
                'evidence_quotes': content_insights['risk_evidence_quotes'][:2],
                'key_risk_metrics': self.risk_archetypes[risk_archetype]['key_metrics'],
                'risk_concerns': content_insights['risk_concerns'],
                'risk_strengths': content_insights['risk_strengths'],
                'archetype_definition': self.risk_archetypes[risk_archetype]['definition'],
                'strategic_context': self.risk_archetypes[risk_archetype]['strategic_context'],
                'board_implications': self.risk_archetypes[risk_archetype]['board_implications']
            },
            'strategic_recommendations': content_insights['recommendations'],
            'analysis_metadata': {
                'analysis_type': 'executive_fallback_comprehensive',
                'confidence_level': content_insights['confidence_level'],
                'files_analyzed': len(extracted_content) if extracted_content else 1,
                'critical_information_gaps': content_insights['information_gaps'],
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_context': analysis_context
            }
        }
    
    def _analyze_content_patterns(self, content: str, company_name: str) -> Dict[str, Any]:
        """Advanced content pattern analysis for executive insights"""
        
        if not content:
            return self._get_minimal_content_insights(company_name)
        
        content_lower = content.lower()
        
        # Extract strategic indicators
        strategic_indicators = self._extract_strategic_indicators(content, content_lower)
        
        # Extract risk indicators
        risk_indicators = self._extract_risk_indicators(content, content_lower)
        
        # Extract evidence quotes
        evidence_quotes = self._extract_evidence_quotes(content)
        
        # Determine business insights
        business_insights = self._derive_business_insights(strategic_indicators, content_lower)
        
        # Determine risk insights
        risk_insights = self._derive_risk_insights(risk_indicators, content_lower)
        
        # Create comprehensive insights
        return {
            'business_rationale': business_insights['rationale'],
            'competitive_implications': business_insights['competitive_context'],
            'strategic_risks': business_insights['risks'],
            'strategic_opportunities': business_insights['opportunities'],
            'risk_rationale': risk_insights['rationale'],
            'governance_implications': risk_insights['governance_context'],
            'risk_concerns': risk_insights['concerns'],
            'risk_strengths': risk_insights['strengths'],
            'evidence_quotes': evidence_quotes['business'],
            'risk_evidence_quotes': evidence_quotes['risk'],
            'recommendations': self._generate_strategic_recommendations(strategic_indicators, risk_indicators),
            'confidence_level': self._assess_confidence_level(content, strategic_indicators, risk_indicators),
            'information_gaps': self._identify_information_gaps(strategic_indicators, risk_indicators)
        }
    
    def _extract_strategic_indicators(self, content: str, content_lower: str) -> Dict[str, Any]:
        """Extract strategic business indicators from content"""
        indicators = {
            'growth_mentions': 0,
            'innovation_mentions': 0,
            'market_focus': [],
            'product_focus': [],
            'efficiency_focus': 0,
            'specialist_focus': 0,
            'scale_focus': 0,
            'service_focus': 0
        }
        
        # Count strategic keywords
        growth_keywords = ['growth', 'expansion', 'develop', 'increase', 'scale']
        innovation_keywords = ['innovation', 'technology', 'digital', 'new product', 'development']
        efficiency_keywords = ['efficiency', 'cost', 'productivity', 'lean', 'optimization']
        specialist_keywords = ['specialist', 'niche', 'expertise', 'focused', 'specialized']
        scale_keywords = ['distribution', 'network', 'platform', 'channel', 'partnership']
        service_keywords = ['service', 'customer', 'client', 'experience', 'relationship']
        
        for keyword in growth_keywords:
            indicators['growth_mentions'] += content_lower.count(keyword)
        
        for keyword in innovation_keywords:
            indicators['innovation_mentions'] += content_lower.count(keyword)
            
        for keyword in efficiency_keywords:
            indicators['efficiency_focus'] += content_lower.count(keyword)
            
        for keyword in specialist_keywords:
            indicators['specialist_focus'] += content_lower.count(keyword)
            
        for keyword in scale_keywords:
            indicators['scale_focus'] += content_lower.count(keyword)
            
        for keyword in service_keywords:
            indicators['service_focus'] += content_lower.count(keyword)
        
        # Extract market and product mentions
        market_indicators = ['mortgage', 'lending', 'credit', 'finance', 'banking', 'insurance']
        for indicator in market_indicators:
            if indicator in content_lower:
                indicators['market_focus'].append(indicator)
        
        return indicators
    
    def _extract_risk_indicators(self, content: str, content_lower: str) -> Dict[str, Any]:
        """Extract risk management indicators from content"""
        indicators = {
            'compliance_focus': 0,
            'regulatory_mentions': 0,
            'capital_focus': 0,
            'control_mentions': 0,
            'resilience_focus': 0,
            'risk_appetite_indicators': [],
            'governance_strength': 0
        }
        
        # Count risk keywords
        compliance_keywords = ['compliance', 'regulatory', 'regulation', 'fca', 'pra']
        capital_keywords = ['capital', 'reserves', 'buffer', 'strength', 'adequacy']
        control_keywords = ['control', 'governance', 'oversight', 'monitoring', 'framework']
        resilience_keywords = ['resilience', 'stress', 'scenario', 'continuity', 'recovery']
        
        for keyword in compliance_keywords:
            indicators['compliance_focus'] += content_lower.count(keyword)
            
        for keyword in capital_keywords:
            indicators['capital_focus'] += content_lower.count(keyword)
            
        for keyword in control_keywords:
            indicators['control_mentions'] += content_lower.count(keyword)
            
        for keyword in resilience_keywords:
            indicators['resilience_focus'] += content_lower.count(keyword)
        
        # Assess governance strength
        governance_indicators = ['board', 'director', 'committee', 'policy', 'procedure']
        for indicator in governance_indicators:
            indicators['governance_strength'] += content_lower.count(indicator)
        
        return indicators
    
    def _extract_evidence_quotes(self, content: str) -> Dict[str, List[str]]:
        """Extract relevant quotes as evidence"""
        quotes = {
            'business': [],
            'risk': []
        }
        
        # Split into sentences and filter for strategic content
        sentences = re.split(r'[.!?]+', content)
        
        business_keywords = ['strategy', 'vision', 'mission', 'objective', 'market', 'growth', 'product']
        risk_keywords = ['risk', 'compliance', 'governance', 'capital', 'regulatory', 'control']
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 30:  # Meaningful length
                sentence_lower = sentence.lower()
                
                if any(keyword in sentence_lower for keyword in business_keywords):
                    quotes['business'].append(sentence)
                elif any(keyword in sentence_lower for keyword in risk_keywords):
                    quotes['risk'].append(sentence)
        
        # Return top quotes by relevance
        return {
            'business': quotes['business'][:5],
            'risk': quotes['risk'][:5]
        }
    
    def _derive_business_insights(self, indicators: Dict[str, Any], content_lower: str) -> Dict[str, Any]:
        """Derive business strategy insights from indicators"""
        
        # Determine dominant strategic orientation
        if indicators['specialist_focus'] > 5:
            rationale = f"Strong specialist focus indicated by {indicators['specialist_focus']} mentions of specialization themes and niche market positioning."
            competitive_context = "Differentiation through expertise depth rather than scale, creating defensive competitive moats."
            
        elif indicators['scale_focus'] > 3:
            rationale = f"Scale-oriented strategy evidenced by {indicators['scale_focus']} references to distribution, platforms, and network expansion."
            competitive_context = "Market share capture through channel expansion and partnership leverage."
            
        elif indicators['service_focus'] > 4:
            rationale = f"Service differentiation strategy with {indicators['service_focus']} mentions of customer experience and relationship focus."
            competitive_context = "Premium positioning through superior client experience rather than price competition."
            
        elif indicators['innovation_mentions'] > 3:
            rationale = f"Innovation-driven approach with {indicators['innovation_mentions']} technology and development references."
            competitive_context = "Competitive advantage through product innovation and technological capabilities."
            
        else:
            rationale = "Conservative business approach focused on operational stability and controlled growth."
            competitive_context = "Emphasis on sustainable operations and market position defense."
        
        # Identify strategic risks and opportunities
        risks = []
        opportunities = []
        
        if indicators['growth_mentions'] > 5:
            risks.append("Execution risk from ambitious growth targets")
            opportunities.append("Market expansion potential if growth initiatives succeed")
        
        if indicators['specialist_focus'] > 3:
            risks.append("Limited addressable market size constraining scale")
            opportunities.append("Premium pricing potential in specialist segments")
        
        if len(indicators['market_focus']) > 2:
            opportunities.append("Diversified market exposure reducing concentration risk")
        
        return {
            'rationale': rationale,
            'competitive_context': competitive_context,
            'risks': risks or ["Market competitive pressure", "Regulatory change impact"],
            'opportunities': opportunities or ["Market position consolidation", "Operational efficiency gains"]
        }
    
    def _derive_risk_insights(self, indicators: Dict[str, Any], content_lower: str) -> Dict[str, Any]:
        """Derive risk strategy insights from indicators"""
        
        # Determine risk management orientation
        if indicators['compliance_focus'] > 8:
            rationale = f"Compliance-first risk approach with {indicators['compliance_focus']} regulatory references indicating conservative stance."
            governance_context = "Strong regulatory relationship prioritized over commercial optimization."
            
        elif indicators['resilience_focus'] > 3:
            rationale = f"Resilience-focused risk management with {indicators['resilience_focus']} mentions of stress testing and scenario planning."
            governance_context = "Operational continuity and crisis preparedness emphasized in governance framework."
            
        elif indicators['capital_focus'] > 5:
            rationale = f"Capital-centric risk approach with {indicators['capital_focus']} strength and adequacy references."
            governance_context = "Board oversight focused on capital preservation and financial stability."
            
        else:
            rationale = "Balanced risk management approach integrating compliance, operational, and strategic considerations."
            governance_context = "Risk governance framework balancing growth enablement with protective oversight."
        
        # Assess risk strengths and concerns
        strengths = []
        concerns = []
        
        if indicators['governance_strength'] > 10:
            strengths.append("Well-established governance framework with comprehensive board oversight")
        
        if indicators['compliance_focus'] > 5:
            strengths.append("Strong regulatory compliance culture and proactive engagement")
        
        if indicators['control_mentions'] > 5:
            strengths.append("Comprehensive control environment with robust monitoring capabilities")
        
        if indicators['compliance_focus'] < 3:
            concerns.append("Limited evidence of comprehensive regulatory compliance framework")
        
        if indicators['governance_strength'] < 5:
            concerns.append("Governance structure may require strengthening for effective oversight")
        
        return {
            'rationale': rationale,
            'governance_context': governance_context,
            'strengths': strengths or ["Established risk management processes", "Regulatory compliance focus"],
            'concerns': concerns or ["Evolving regulatory landscape challenges", "Operational risk management complexity"]
        }
    
    def _determine_business_archetype_executive(self, insights: Dict[str, Any]) -> str:
        """Determine business archetype using executive-grade logic"""
        
        # Extract key indicators
        rationale = insights['business_rationale'].lower()
        
        # Business archetype mapping based on strategic indicators
        if 'specialist' in rationale and 'focus' in rationale:
            if 'niche' in rationale or 'expertise' in rationale:
                return 'Expert Niche Leader'
            else:
                return 'Disciplined Specialist Growth'
        
        elif 'service' in rationale and ('experience' in rationale or 'relationship' in rationale):
            return 'Service-Driven Differentiator'
        
        elif 'scale' in rationale and ('distribution' in rationale or 'network' in rationale):
            return 'Scale-through-Distribution'
        
        elif 'innovation' in rationale and ('technology' in rationale or 'development' in rationale):
            return 'Product-Innovation Flywheel'
        
        elif 'efficiency' in rationale or 'cost' in rationale:
            return 'Cost-Leadership Operator'
        
        elif 'conservative' in rationale or 'stability' in rationale:
            return 'Balance-Sheet Steward'
        
        else:
            # Default to most common financial services archetype
            return 'Disciplined Specialist Growth'
    
    def _determine_risk_archetype_executive(self, insights: Dict[str, Any]) -> str:
        """Determine risk archetype using executive-grade logic"""
        
        rationale = insights['risk_rationale'].lower()
        
        # Risk archetype mapping
        if 'compliance' in rationale and ('conservative' in rationale or 'regulatory' in rationale):
            return 'Risk-First Conservative'
        
        elif 'resilience' in rationale and ('stress' in rationale or 'scenario' in rationale):
            return 'Resilience-Focused Architect'
        
        elif 'capital' in rationale and ('preservation' in rationale or 'strength' in rationale):
            return 'Risk-First Conservative'
        
        elif 'balanced' in rationale and ('growth' in rationale or 'strategic' in rationale):
            return 'Strategic Risk-Taker'
        
        elif 'governance' in rationale and 'framework' in rationale:
            return 'Embedded Risk Partner'
        
        else:
            # Default conservative approach for financial services
            return 'Risk-First Conservative'
    
    def _get_secondary_archetype(self, primary: str, archetype_dict: Dict[str, Dict[str, Any]]) -> str:
        """Get appropriate secondary archetype"""
        
        # Business archetype secondary mapping
        if archetype_dict == self.business_archetypes:
            secondary_mapping = {
                'Disciplined Specialist Growth': 'Balance-Sheet Steward',
                'Expert Niche Leader': 'Service-Driven Differentiator',
                'Service-Driven Differentiator': 'Expert Niche Leader',
                'Scale-through-Distribution': 'Asset-Velocity Maximiser',
                'Product-Innovation Flywheel': 'Tech-Productivity Accelerator',
                'Cost-Leadership Operator': 'Tech-Productivity Accelerator',
                'Balance-Sheet Steward': 'Disciplined Specialist Growth'
            }
        else:
            # Risk archetype secondary mapping
            secondary_mapping = {
                'Risk-First Conservative': 'Rules-Led Operator',
                'Resilience-Focused Architect': 'Strategic Risk-Taker',
                'Strategic Risk-Taker': 'Embedded Risk Partner',
                'Embedded Risk Partner': 'Quant-Control Enthusiast',
                'Rules-Led Operator': 'Risk-First Conservative'
            }
        
        return secondary_mapping.get(primary, list(archetype_dict.keys())[1])
    
    def _generate_strategic_recommendations(self, strategic_indicators: Dict[str, Any], 
                                         risk_indicators: Dict[str, Any]) -> List[str]:
        """Generate executive-level strategic recommendations"""
        recommendations = []
        
        # Business strategy recommendations
        if strategic_indicators['growth_mentions'] > 5:
            recommendations.append("Establish clear growth governance framework to manage execution risk while capturing market opportunities")
        
        if strategic_indicators['specialist_focus'] > 3:
            recommendations.append("Leverage specialist expertise for premium pricing while exploring adjacent market expansion opportunities")
        
        if strategic_indicators['innovation_mentions'] > 3:
            recommendations.append("Develop innovation portfolio management to balance breakthrough potential with execution certainty")
        
        # Risk strategy recommendations  
        if risk_indicators['compliance_focus'] > 8:
            recommendations.append("Balance regulatory excellence with commercial agility to avoid over-conservative constraint on growth")
        
        if risk_indicators['governance_strength'] < 5:
            recommendations.append("Strengthen board risk oversight capabilities and risk management framework sophistication")
        
        # General strategic recommendations
        if not recommendations:
            recommendations = [
                "Conduct comprehensive strategic review to clarify competitive positioning and growth trajectory",
                "Enhance risk management capabilities to support strategic objectives while maintaining prudent oversight"
            ]
        
        return recommendations[:4]  # Limit to 4 key recommendations
    
    def _assess_confidence_level(self, content: str, strategic_indicators: Dict[str, Any], 
                               risk_indicators: Dict[str, Any]) -> str:
        """Assess confidence level in analysis"""
        
        if not content or len(content) < 1000:
            return 'low'
        
        # Count total strategic indicators
        strategic_score = sum([
            strategic_indicators.get('growth_mentions', 0),
            strategic_indicators.get('innovation_mentions', 0),
            strategic_indicators.get('specialist_focus', 0),
            len(strategic_indicators.get('market_focus', []))
        ])
        
        risk_score = sum([
            risk_indicators.get('compliance_focus', 0),
            risk_indicators.get('governance_strength', 0),
            risk_indicators.get('control_mentions', 0)
        ])
        
        total_score = strategic_score + risk_score
        
        if total_score > 20:
            return 'high'
        elif total_score > 10:
            return 'medium'
        else:
            return 'low'
    
    def _identify_information_gaps(self, strategic_indicators: Dict[str, Any], 
                                 risk_indicators: Dict[str, Any]) -> List[str]:
        """Identify critical information gaps for board consideration"""
        gaps = []
        
        if strategic_indicators.get('growth_mentions', 0) < 2:
            gaps.append("Limited strategic growth planning documentation")
        
        if len(strategic_indicators.get('market_focus', [])) < 2:
            gaps.append("Market positioning and competitive strategy clarity needed")
        
        if risk_indicators.get('governance_strength', 0) < 5:
            gaps.append("Risk governance framework documentation requires enhancement")
        
        if risk_indicators.get('compliance_focus', 0) < 3:
            gaps.append("Regulatory compliance strategy and capabilities assessment needed")
        
        if not gaps:
            gaps = ["Quantitative performance metrics for strategic and risk objectives"]
        
        return gaps[:3]  # Top 3 gaps
    
    def _get_minimal_content_insights(self, company_name: str) -> Dict[str, Any]:
        """Provide minimal insights when content is insufficient"""
        return {
            'business_rationale': f'Limited documentation available for {company_name} strategic analysis',
            'competitive_implications': 'Competitive positioning assessment requires additional strategic documentation',
            'strategic_risks': ['Insufficient strategic planning documentation', 'Limited competitive intelligence'],
            'strategic_opportunities': ['Strategic planning enhancement opportunity', 'Documentation improvement potential'],
            'risk_rationale': 'Risk management assessment constrained by limited available documentation',
            'governance_implications': 'Governance framework assessment requires additional policy and procedure documentation',
            'risk_concerns': ['Documentation gaps in risk management', 'Limited governance visibility'],
            'risk_strengths': ['Opportunity for comprehensive risk framework development'],
            'evidence_quotes': [],
            'risk_evidence_quotes': [],
            'recommendations': [
                'Conduct comprehensive strategic planning documentation review',
                'Enhance risk management framework documentation and governance visibility'
            ],
            'confidence_level': 'low',
            'information_gaps': [
                'Strategic planning documentation',
                'Risk management policies and procedures',
                'Competitive positioning analysis'
            ]
        }
    
    def _create_board_ready_output(self, analysis: Dict[str, Any], company_name: str, 
                                 analysis_context: Optional[str]) -> Dict[str, Any]:
        """Transform analysis into board-ready format with executive summary and actionable insights"""
        
        # Extract key components
        business_analysis = analysis.get('business_strategy_analysis', {})
        risk_analysis = analysis.get('risk_strategy_analysis', {})
        
        # Create executive dashboard
        executive_dashboard = {
            'strategic_archetype_summary': {
                'business_strategy': business_analysis.get('dominant_archetype', 'Not determined'),
                'risk_strategy': risk_analysis.get('dominant_archetype', 'Not determined'),
                'strategic_alignment': self._assess_strategy_risk_alignment(
                    business_analysis.get('dominant_archetype'), 
                    risk_analysis.get('dominant_archetype')
                )
            },
            'key_strategic_insights': self._create_key_insights(business_analysis, risk_analysis),
            'board_action_items': self._create_board_action_items(analysis),
            'strategic_risk_heatmap': self._create_risk_heatmap(business_analysis, risk_analysis)
        }
        
        # Enhanced analysis with board context
        enhanced_analysis = analysis.copy()
        enhanced_analysis['executive_dashboard'] = executive_dashboard
        enhanced_analysis['board_presentation_summary'] = self._create_board_presentation_summary(
            analysis, company_name, analysis_context
        )
        
        return enhanced_analysis
    
    def _assess_strategy_risk_alignment(self, business_archetype: Optional[str], 
                                      risk_archetype: Optional[str]) -> Dict[str, Any]:
        """Assess alignment between business and risk strategies"""
        
        if not business_archetype or not risk_archetype:
            return {
                'alignment_level': 'Unknown',
                'alignment_commentary': 'Insufficient data for alignment assessment'
            }
        
        # Define alignment matrix
        high_alignment_pairs = [
            ('Disciplined Specialist Growth', 'Risk-First Conservative'),
            ('Balance-Sheet Steward', 'Risk-First Conservative'),
            ('Expert Niche Leader', 'Mission-Driven Prudence'),
            ('Service-Driven Differentiator', 'Reputation-First Shield'),
            ('Product-Innovation Flywheel', 'Strategic Risk-Taker'),
            ('Scale-through-Distribution', 'Control-Lag Follower')
        ]
        
        moderate_alignment_pairs = [
            ('Disciplined Specialist Growth', 'Rules-Led Operator'),
            ('Cost-Leadership Operator', 'Risk-First Conservative'),
            ('Tech-Productivity Accelerator', 'Quant-Control Enthusiast')
        ]
        
        if (business_archetype, risk_archetype) in high_alignment_pairs:
            return {
                'alignment_level': 'High',
                'alignment_commentary': f'{business_archetype} strategy is well-supported by {risk_archetype} risk approach, creating coherent strategic framework'
            }
        elif (business_archetype, risk_archetype) in moderate_alignment_pairs:
            return {
                'alignment_level': 'Moderate', 
                'alignment_commentary': f'{business_archetype} strategy is generally compatible with {risk_archetype} risk approach, with opportunities for optimization'
            }
        else:
            return {
                'alignment_level': 'Requires Review',
                'alignment_commentary': f'Potential misalignment between {business_archetype} strategy and {risk_archetype} risk approach warrants board discussion'
            }
    
    def _create_key_insights(self, business_analysis: Dict[str, Any], 
                           risk_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create key strategic insights for board consumption"""
        insights = []
        
        # Business strategy insight
        business_archetype = business_analysis.get('dominant_archetype', '')
        if business_archetype:
            business_context = self.business_archetypes.get(business_archetype, {}).get('strategic_context', '')
            insights.append({
                'category': 'Business Strategy',
                'insight': f'Company operates as {business_archetype}',
                'implication': business_context,
                'board_significance': 'Defines competitive positioning and growth trajectory'
            })
        
        # Risk strategy insight
        risk_archetype = risk_analysis.get('dominant_archetype', '')
        if risk_archetype:
            risk_context = self.risk_archetypes.get(risk_archetype, {}).get('strategic_context', '')
            insights.append({
                'category': 'Risk Strategy',
                'insight': f'Risk management follows {risk_archetype} approach',
                'implication': risk_context,
                'board_significance': 'Determines risk appetite and governance requirements'
            })
        
        # Competitive positioning insight
        competitive_implications = business_analysis.get('competitive_implications', '')
        if competitive_implications:
            insights.append({
                'category': 'Competitive Position',
                'insight': 'Strategic positioning analysis',
                'implication': competitive_implications,
                'board_significance': 'Informs market strategy and investment priorities'
            })
        
        return insights[:3]  # Top 3 insights for board focus
    
    def _create_board_action_items(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create specific action items for board consideration"""
        action_items = []
        
        recommendations = analysis.get('strategic_recommendations', [])
        confidence_level = analysis.get('analysis_metadata', {}).get('confidence_level', 'medium')
        information_gaps = analysis.get('analysis_metadata', {}).get('critical_information_gaps', [])
        
        # Strategic action items from recommendations
        for i, recommendation in enumerate(recommendations[:2]):
            action_items.append({
                'priority': 'High' if i == 0 else 'Medium',
                'action': recommendation,
                'owner': 'Board/Executive Team',
                'timeline': '3-6 months',
                'success_criteria': 'Implementation plan developed and progress metrics established'
            })
        
        # Information gap action items
        if information_gaps:
            action_items.append({
                'priority': 'Medium',
                'action': f'Address critical information gaps: {", ".join(information_gaps[:2])}',
                'owner': 'Management Team',
                'timeline': '1-3 months',
                'success_criteria': 'Complete documentation and analysis available for board review'
            })
        
        # Confidence-based action items
        if confidence_level == 'low':
            action_items.append({
                'priority': 'High',
                'action': 'Conduct comprehensive strategic review with external advisory support',
                'owner': 'Board Chair/CEO',
                'timeline': '2-4 months',
                'success_criteria': 'Detailed strategic assessment and roadmap approved by board'
            })
        
        return action_items[:4]  # Maximum 4 action items
    
    def _create_risk_heatmap(self, business_analysis: Dict[str, Any], 
                           risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create strategic risk heatmap for board oversight"""
        
        risk_heatmap = {
            'high_priority_risks': [],
            'medium_priority_risks': [],
            'risk_mitigation_strengths': []
        }
        
        # Extract risks from analysis
        strategic_risks = business_analysis.get('strategic_risks', [])
        risk_concerns = risk_analysis.get('risk_concerns', [])
        risk_strengths = risk_analysis.get('risk_strengths', [])
        
        # Categorize risks by priority
        all_risks = strategic_risks + risk_concerns
        
        # High priority risks (first 2)
        risk_heatmap['high_priority_risks'] = all_risks[:2] if all_risks else [
            'Strategic planning visibility requires enhancement',
            'Risk framework documentation needs strengthening'
        ]
        
        # Medium priority risks (next 2)
        risk_heatmap['medium_priority_risks'] = all_risks[2:4] if len(all_risks) > 2 else [
            'Competitive positioning monitoring',
            'Regulatory landscape evolution'
        ]
        
        # Risk strengths
        risk_heatmap['risk_mitigation_strengths'] = risk_strengths[:3] if risk_strengths else [
            'Established governance framework',
            'Regulatory compliance focus',
            'Operational stability'
        ]
        
        return risk_heatmap
    
    def _create_board_presentation_summary(self, analysis: Dict[str, Any], company_name: str,
                                         analysis_context: Optional[str]) -> Dict[str, Any]:
        """Create executive summary suitable for board presentation"""
        
        business_analysis = analysis.get('business_strategy_analysis', {})
        risk_analysis = analysis.get('risk_strategy_analysis', {})
        executive_summary = analysis.get('executive_summary', '')
        
        context_intro = f" for {analysis_context}" if analysis_context else ""
        
        return {
            'presentation_title': f'Strategic Archetype Analysis: {company_name}{context_intro}',
            'executive_summary': executive_summary,
            'key_findings': {
                'business_strategy_classification': {
                    'primary': business_analysis.get('dominant_archetype', 'Not determined'),
                    'definition': business_analysis.get('archetype_definition', ''),
                    'board_implications': business_analysis.get('board_implications', '')
                },
                'risk_strategy_classification': {
                    'primary': risk_analysis.get('dominant_archetype', 'Not determined'),
                    'definition': risk_analysis.get('archetype_definition', ''),
                    'board_implications': risk_analysis.get('board_implications', '')
                }
            },
            'strategic_recommendations_summary': {
                'immediate_actions': analysis.get('strategic_recommendations', [])[:2],
                'strategic_considerations': analysis.get('strategic_recommendations', [])[2:4] if len(analysis.get('strategic_recommendations', [])) > 2 else []
            },
            'next_steps': {
                'board_discussion_points': [
                    'Review and validate strategic archetype classifications',
                    'Assess alignment between business and risk strategies',
                    'Approve recommended strategic initiatives and timeline'
                ],
                'management_actions': [
                    'Develop detailed implementation plans for approved recommendations',
                    'Address identified information gaps and documentation requirements',
                    'Establish progress monitoring and reporting framework'
                ]
            },
            'appendix_data': {
                'analysis_confidence': analysis.get('analysis_metadata', {}).get('confidence_level', 'medium'),
                'data_sources': f"{analysis.get('analysis_metadata', {}).get('files_analyzed', 1)} documents analyzed",
                'analysis_date': analysis.get('analysis_metadata', {}).get('analysis_timestamp', ''),
                'methodology': analysis.get('analysis_metadata', {}).get('analysis_type', 'comprehensive_strategic_analysis')
            }
        }
    
    def _create_emergency_board_analysis(self, company_name: str, company_number: str, 
                                       error_message: str) -> Dict[str, Any]:
        """Create emergency analysis when all processing fails"""
        return {
            'executive_summary': f'Strategic analysis for {company_name} encountered processing constraints. Emergency board-grade assessment provided based on standard financial services frameworks.',
            'business_strategy_analysis': {
                'dominant_archetype': 'Disciplined Specialist Growth',
                'secondary_archetype': 'Balance-Sheet Steward',
                'strategic_rationale': 'Conservative assessment applied due to processing limitations. Disciplined Specialist Growth represents prudent default for financial services strategic positioning.',
                'competitive_implications': 'Requires detailed strategic review to determine actual competitive positioning and market strategy.',
                'evidence_quotes': [],
                'key_metrics': ['ROE consistency', 'Market share in niche', 'Credit quality metrics'],
                'strategic_risks': ['Limited strategic visibility', 'Competitive positioning uncertainty'],
                'strategic_opportunities': ['Strategic planning enhancement', 'Market positioning clarification'],
                'archetype_definition': self.business_archetypes['Disciplined Specialist Growth']['definition'],
                'strategic_context': self.business_archetypes['Disciplined Specialist Growth']['strategic_context'],
                'board_implications': self.business_archetypes['Disciplined Specialist Growth']['board_implications']
            },
            'risk_strategy_analysis': {
                'dominant_archetype': 'Risk-First Conservative',
                'secondary_archetype': 'Rules-Led Operator',
                'risk_rationale': 'Conservative risk assessment applied as prudent default. Risk-First Conservative approach assumed for financial services regulatory context.',
                'governance_implications': 'Board oversight required for comprehensive risk strategy assessment and framework validation.',
                'evidence_quotes': [],
                'key_risk_metrics': ['Capital buffer levels', 'Regulatory rating scores', 'Risk-adjusted returns'],
                'risk_concerns': ['Risk framework visibility limited', 'Governance assessment requires enhancement'],
                'risk_strengths': ['Conservative approach assumed', 'Regulatory compliance focus maintained'],
                'archetype_definition': self.risk_archetypes['Risk-First Conservative']['definition'],
                'strategic_context': self.risk_archetypes['Risk-First Conservative']['strategic_context'],
                'board_implications': self.risk_archetypes['Risk-First Conservative']['board_implications']
            },
            'strategic_recommendations': [
                'Immediate: Conduct comprehensive strategic documentation review and gap analysis',
                'Priority: Engage external strategic advisory support for detailed archetype assessment',
                'Strategic: Develop comprehensive strategic planning framework with clear archetype alignment'
            ],
            'executive_dashboard': {
                'strategic_archetype_summary': {
                    'business_strategy': 'Disciplined Specialist Growth (Default Assessment)',
                    'risk_strategy': 'Risk-First Conservative (Default Assessment)',
                    'strategic_alignment': {
                        'alignment_level': 'Requires Validation',
                        'alignment_commentary': 'Default assessment requires board validation through comprehensive strategic review'
                    }
                },
                'key_strategic_insights': [
                    {
                        'category': 'Analysis Status',
                        'insight': 'Processing limitations encountered',
                        'implication': 'Comprehensive strategic review required for accurate archetype assessment',
                        'board_significance': 'Board should commission detailed strategic analysis with enhanced documentation'
                    }
                ],
                'board_action_items': [
                    {
                        'priority': 'Immediate',
                        'action': 'Commission comprehensive strategic analysis with external advisory support',
                        'owner': 'Board Chair/CEO',
                        'timeline': '1-2 months',
                        'success_criteria': 'Detailed strategic assessment completed with validated archetype classifications'
                    }
                ]
            },
            'board_presentation_summary': {
                'presentation_title': f'Emergency Strategic Assessment: {company_name}',
                'executive_summary': 'Processing constraints required emergency assessment. Immediate strategic review recommended.',
                'key_findings': {
                    'analysis_status': 'Emergency assessment applied due to processing limitations',
                    'immediate_requirement': 'Comprehensive strategic review with enhanced documentation'
                },
                'next_steps': {
                    'immediate_board_action': 'Approve comprehensive strategic analysis engagement',
                    'timeline': '1-2 months for complete assessment'
                }
            },
            'analysis_metadata': {
                'analysis_type': 'emergency_board_assessment',
                'confidence_level': 'low',
                'critical_information_gaps': ['Complete strategic documentation', 'Detailed financial analysis', 'Comprehensive risk assessment'],
                'analysis_timestamp': datetime.now().isoformat(),
                'processing_note': f'Emergency assessment due to: {error_message}'
            }
        }

# Usage example and testing framework
if __name__ == "__main__":
    print("🏛️ Executive AI Analyzer v3.0 - Board-Grade Analysis Engine")
    print("=" * 60)
    
    analyzer = ExecutiveAIAnalyzer()
    
    # Test with sample financial services content
    sample_content = """
    Together Personal Finance Limited is a specialist mortgage lender focused on providing 
    secured loans, consumer buy-to-let, and bridging finance. The company's vision is aligned 
    to be the most valued lending company in the UK, focusing on sustainable and controlled 
    growth within the specialist lending market.
    
    The company's strategy includes delivering good customer outcomes, enhancing customer 
    experience, and focusing on supporting customer needs. The company maintains strong 
    regulatory relationships and emphasizes compliance with FCA and PRA requirements.
    
    Risk management is centered on capital preservation and regulatory compliance, with 
    comprehensive stress testing and scenario planning capabilities. The board maintains 
    active oversight of risk appetite and strategic direction.
    """
    
    print("Testing board-grade analysis...")
    result = analyzer.analyze_for_board(
        content=sample_content,
        company_name="Together Personal Finance Limited",
        company_number="02613335",
        analysis_context="Annual Strategic Review"
    )
    
    print("\n📊 EXECUTIVE DASHBOARD")
    print("=" * 40)
    dashboard = result.get('executive_dashboard', {})
    
    summary = dashboard.get('strategic_archetype_summary', {})
    print(f"Business Strategy: {summary.get('business_strategy', 'Not determined')}")
    print(f"Risk Strategy: {summary.get('risk_strategy', 'Not determined')}")
    
    alignment = summary.get('strategic_alignment', {})
    print(f"Strategy-Risk Alignment: {alignment.get('alignment_level', 'Unknown')}")
    
    print(f"\n📋 BOARD ACTION ITEMS")
    print("=" * 30)
    action_items = dashboard.get('board_action_items', [])
    for i, item in enumerate(action_items[:2], 1):
        print(f"{i}. [{item.get('priority', 'Medium')}] {item.get('action', 'No action specified')}")
    
    print(f"\n🎯 KEY INSIGHTS")
    print("=" * 20)
    insights = dashboard.get('key_strategic_insights', [])
    for insight in insights[:2]:
        print(f"• {insight.get('category', 'General')}: {insight.get('insight', 'No insight')}")
    
    print(f"\n✅ Executive AI Analyzer v3.0 test completed")
    print(f"Analysis Type: {result.get('analysis_metadata', {}).get('analysis_type', 'unknown')}")
    print(f"Confidence Level: {result.get('analysis_metadata', {}).get('confidence_level', 'unknown')}")
    
    print("\nReady for board-grade strategic analysis! 🏛️")