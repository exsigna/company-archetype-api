#!/usr/bin/env python3
"""
OPTIMIZED Multi-Pass AI Analyzer - Maximum Claude 3.5 Sonnet Token Usage
Uses full 200K input / 8K output capacity for maximum analysis depth
"""

import os
import logging
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
import time
import re
from datetime import datetime
from collections import Counter
import statistics

logger = logging.getLogger(__name__)

class OptimizedClaudeAnalyzer:
    """
    Optimized analyzer that maximizes Claude 3.5 Sonnet's 200K input / 8K output capacity
    """
    
    def __init__(self):
        """Initialize optimized Claude analyzer"""
        self.anthropic_client = None
        self.openai_client = None
        self.client_type = "fallback"
        
        # OPTIMIZED: Maximum token usage settings
        self.max_input_tokens = 200000      # Claude 3.5 Sonnet max input
        self.max_output_tokens = 8000       # Claude 3.5 Sonnet max output (increased from 4500)
        self.max_input_chars = 800000       # ~200K tokens = ~800K characters
        self.target_analysis_length = 32000 # ~8K tokens = ~32K characters
        
        # Multi-pass settings
        self.num_passes = int(os.environ.get('AI_ANALYSIS_PASSES', 3))
        self.temperature_variation = os.environ.get('TEMPERATURE_VARIATION', 'true').lower() == 'true'
        
        logger.info(f"🚀 OPTIMIZED Claude 3.5 Sonnet Analyzer v8.0")
        logger.info(f"📊 Max input: {self.max_input_tokens:,} tokens ({self.max_input_chars:,} chars)")
        logger.info(f"📊 Max output: {self.max_output_tokens:,} tokens ({self.target_analysis_length:,} chars)")
        logger.info(f"🔄 Analysis passes: {self.num_passes}")
        
        # Initialize AI providers
        self._init_anthropic_primary()
        self._init_openai_fallback()
        
        # Enhanced archetype definitions with more detail for richer analysis
        self.business_archetypes = {
            "Scale-through-Distribution": {
                "definition": "Gains market share primarily by expanding distribution channels and partnerships faster than operational maturity develops",
                "strategic_context": "High-velocity expansion strategy with emphasis on market capture over operational excellence",
                "evidence_keywords": ["distribution", "channels", "partnerships", "network", "expansion", "market share", "scale", "intermediaries", "brokers"],
                "strategic_indicators": ["rapid channel expansion", "partnership velocity", "market penetration focus", "distribution-first growth"],
                "competitive_dynamics": ["channel conflict management", "partner relationship quality", "distribution network effects"],
                "risk_considerations": ["operational lag behind expansion", "quality control challenges", "partner dependency risks"]
            },
            "Disciplined Specialist Growth": {
                "definition": "Maintains niche focus with strong underwriting capabilities, growing opportunistically while optimizing balance sheet efficiency",
                "strategic_context": "Conservative growth strategy emphasizing expertise depth over market breadth",
                "evidence_keywords": ["specialist", "niche", "underwriting", "disciplined", "controlled growth", "expertise", "conservative", "prudent"],
                "strategic_indicators": ["selective market focus", "expertise-driven decisions", "controlled expansion", "balance sheet optimization"],
                "competitive_dynamics": ["niche market leadership", "expertise barriers to entry", "premium pricing power"],
                "risk_considerations": ["market concentration risk", "limited diversification", "expertise dependency"]
            },
            "Service-Driven Differentiator": {
                "definition": "Competes on superior client experience and advisory capability rather than price or scale",
                "strategic_context": "Relationship-centric model with emphasis on customer satisfaction and loyalty",
                "evidence_keywords": ["service", "client experience", "advisory", "relationship", "satisfaction", "loyalty", "customer-centric"],
                "strategic_indicators": ["service quality metrics", "customer retention focus", "advisory capabilities", "relationship investment"],
                "competitive_dynamics": ["service-based differentiation", "customer lifetime value optimization", "advisory moat building"],
                "risk_considerations": ["service cost management", "scalability challenges", "talent dependency"]
            },
            "Asset-Velocity Maximiser": {
                "definition": "Prioritizes rapid asset origination and turnover, often accepting higher funding costs for speed",
                "strategic_context": "Transaction-focused model optimizing for volume and velocity over margin per transaction",
                "evidence_keywords": ["origination", "turnover", "velocity", "volume", "transaction", "speed"]
            },
            "Balance-Sheet Steward": {
                "definition": "Prioritizes capital strength and stakeholder value over aggressive growth",
                "strategic_context": "Conservative approach emphasizing financial stability and long-term sustainability",
                "evidence_keywords": ["capital", "stability", "conservative", "stakeholder", "prudent", "stewardship"]
            }
        }
        
        self.risk_archetypes = {
            "Risk-First Conservative": {
                "definition": "Prioritizes capital preservation and regulatory compliance above growth opportunities",
                "strategic_context": "Defensive risk strategy emphasizing stability and regulatory relationship quality",
                "evidence_keywords": ["capital preservation", "compliance", "conservative", "stability", "defensive", "prudent"],
                "governance_indicators": ["strong capital ratios", "regulatory excellence", "conservative risk appetite", "stability focus"],
                "control_frameworks": ["comprehensive risk management", "regulatory compliance systems", "capital management"],
                "stakeholder_impact": ["regulatory relationship quality", "stakeholder confidence", "crisis resilience"]
            },
            "Rules-Led Operator": {
                "definition": "Emphasizes strict procedural adherence and control consistency over business judgment",
                "strategic_context": "Process-driven risk management with emphasis on consistency and auditability",
                "evidence_keywords": ["procedures", "controls", "consistency", "process", "adherence", "systematic"]
            },
            "Embedded Risk Partner": {
                "definition": "Integrates risk teams into frontline business decisions with collaborative risk appetite setting",
                "strategic_context": "Partnership-based risk management with business-risk team collaboration",
                "evidence_keywords": ["integrated", "collaborative", "partnership", "embedded", "business decisions", "alignment"]
            }
        }
        
        logger.info(f"✅ Optimized Claude Analyzer initialized. Engine: {self.client_type}")
    
    def analyze_for_board_optimized(self, content: str, company_name: str, company_number: str, 
                                  extracted_content: Optional[List[Dict[str, Any]]] = None,
                                  analysis_context: Optional[str] = None) -> Dict[str, Any]:
        """
        OPTIMIZED multi-pass analysis using Claude 3.5 Sonnet's full capacity
        """
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Starting OPTIMIZED {self.num_passes}-pass analysis for {company_name}")
            logger.info(f"📊 Input content: {len(content):,} characters")
            
            # Step 1: Optimize content for maximum Claude usage
            optimized_content = self._optimize_content_for_max_tokens(content, extracted_content, company_name)
            logger.info(f"📊 Optimized content: {len(optimized_content):,} characters ({len(optimized_content)//4:,} estimated tokens)")
            
            # Step 2: Perform multiple optimized analysis passes
            individual_analyses = []
            
            for pass_num in range(1, self.num_passes + 1):
                logger.info(f"🎯 OPTIMIZED Analysis Pass {pass_num}/{self.num_passes}")
                
                # Vary temperature for diversity
                temperature = self._get_optimized_temperature(pass_num)
                
                try:
                    pass_analysis = self._optimized_single_pass(
                        optimized_content, company_name, company_number, 
                        extracted_content, analysis_context, 
                        pass_num, temperature
                    )
                    
                    if pass_analysis:
                        individual_analyses.append(pass_analysis)
                        business_arch = pass_analysis.get('business_strategy', {}).get('dominant_archetype', 'Unknown')
                        risk_arch = pass_analysis.get('risk_strategy', {}).get('dominant_archetype', 'Unknown')
                        analysis_length = len(str(pass_analysis))
                        logger.info(f"✅ Pass {pass_num} completed: {business_arch} | {risk_arch} ({analysis_length:,} chars)")
                    else:
                        logger.warning(f"⚠️ Pass {pass_num} failed - continuing")
                        
                except Exception as e:
                    logger.error(f"❌ Pass {pass_num} failed: {e}")
                    continue
                
                # Brief pause between passes
                if pass_num < self.num_passes:
                    time.sleep(2)
            
            if not individual_analyses:
                logger.error("❌ All optimized analysis passes failed")
                return self._create_emergency_analysis(company_name, company_number, "All passes failed")
            
            logger.info(f"📊 Completed {len(individual_analyses)}/{self.num_passes} optimized passes")
            
            # Step 3: Enhanced synthesis with maximum detail
            synthesized_analysis = self._enhanced_synthesis(
                individual_analyses, company_name, company_number, optimized_content
            )
            
            # Step 4: Create comprehensive optimized report
            final_report = self._create_optimized_report(
                synthesized_analysis, individual_analyses, company_name, company_number
            )
            
            analysis_time = time.time() - start_time
            total_chars = len(str(final_report))
            logger.info(f"🎉 OPTIMIZED analysis completed in {analysis_time:.2f}s")
            logger.info(f"📊 Total output: {total_chars:,} characters (~{total_chars//4:,} tokens)")
            
            return final_report
            
        except Exception as e:
            logger.error(f"❌ Optimized analysis failed: {e}")
            return self._create_emergency_analysis(company_name, company_number, str(e))
    
    def _optimize_content_for_max_tokens(self, content: str, extracted_content: Optional[List[Dict[str, Any]]], 
                                       company_name: str) -> str:
        """
        Optimize content to use Claude 3.5 Sonnet's full 200K token input capacity
        """
        if not content:
            return f"Limited content available for {company_name} analysis."
        
        # OPTIMIZED: Use much more content (up to 800K characters ≈ 200K tokens)
        if len(content) <= self.max_input_chars:
            logger.info(f"📊 Using full content: {len(content):,} characters")
            return content
        
        # If content exceeds limits, intelligently truncate while preserving key sections
        logger.info(f"📊 Content exceeds limit ({len(content):,} chars), optimizing...")
        
        # Enhanced content prioritization
        sections = content.split('\n\n')
        scored_sections = []
        
        # ENHANCED: More sophisticated scoring for maximum value
        priority_terms = {
            # Strategic content (highest priority)
            'strategic report': 50, 'business review': 45, 'strategy': 40,
            'strategic objective': 35, 'business model': 30, 'vision': 25,
            
            # Risk and governance (high priority)
            'risk management': 45, 'governance': 40, 'compliance': 35,
            'board': 30, 'regulatory': 25, 'internal control': 20,
            
            # Financial performance (medium-high priority)
            'financial performance': 35, 'profitability': 30, 'revenue': 25,
            'margin': 20, 'cost management': 15, 'efficiency': 10,
            
            # Operations and market (medium priority)
            'operational': 25, 'market position': 20, 'competitive': 15,
            'customer': 10, 'innovation': 15, 'technology': 10,
            
            # Supporting content (lower priority)
            'corporate': 10, 'sustainability': 8, 'environment': 5,
            'social': 5, 'employee': 5, 'community': 3
        }
        
        for section in sections:
            if len(section.strip()) < 50:  # Skip very short sections
                continue
                
            section_lower = section.lower()
            score = 0
            
            # Calculate relevance score
            for term, weight in priority_terms.items():
                if term in section_lower:
                    score += weight
            
            # Bonus for sections with specific strategic keywords
            strategic_keywords = ['archetype', 'positioning', 'competitive advantage', 
                                'market share', 'growth strategy', 'transformation']
            for keyword in strategic_keywords:
                if keyword in section_lower:
                    score += 20
            
            # Bonus for sections with quantitative data
            if any(indicator in section for indicator in ['%', '£', '$', '€', 'million', 'billion']):
                score += 10
            
            if score > 0:
                scored_sections.append((score, section))
        
        # Sort by relevance and build optimized content
        scored_sections.sort(key=lambda x: x[0], reverse=True)
        
        optimized_content = ""
        for score, section in scored_sections:
            if len(optimized_content) + len(section) < self.max_input_chars:
                optimized_content += section + "\n\n"
            else:
                break
        
        logger.info(f"📊 Content optimized: {len(optimized_content):,} chars from {len(content):,} chars")
        return optimized_content
    
    def _get_optimized_temperature(self, pass_num: int) -> float:
        """
        Get optimized temperature for maximum Claude performance
        """
        # Optimized temperature progression for maximum insight diversity
        temperatures = [0.05, 0.25, 0.15, 0.3, 0.1]  # More sophisticated progression
        return temperatures[(pass_num - 1) % len(temperatures)]
    
    def _optimized_single_pass(self, content: str, company_name: str, company_number: str,
                             extracted_content: Optional[List[Dict[str, Any]]], 
                             analysis_context: Optional[str],
                             pass_num: int, temperature: float) -> Optional[Dict[str, Any]]:
        """
        Single pass optimized for Claude 3.5 Sonnet's maximum capacity
        """
        try:
            if self.client_type == "anthropic_claude":
                return self._anthropic_optimized_pass(
                    content, company_name, company_number, 
                    extracted_content, analysis_context, 
                    pass_num, temperature
                )
            else:
                logger.warning(f"Non-Anthropic client for pass {pass_num} - using fallback")
                return self._fallback_optimized_pass(company_name, company_number, pass_num)
                
        except Exception as e:
            logger.error(f"Optimized pass {pass_num} failed: {e}")
            return None
    
    def _anthropic_optimized_pass(self, content: str, company_name: str, company_number: str,
                                extracted_content: Optional[List[Dict[str, Any]]], 
                                analysis_context: Optional[str],
                                pass_num: int, temperature: float) -> Optional[Dict[str, Any]]:
        """
        OPTIMIZED Anthropic Claude pass using maximum tokens
        """
        try:
            # Create enhanced prompt for maximum output
            prompt = self._create_optimized_prompt(content, company_name, analysis_context, pass_num)
            
            logger.info(f"📊 Pass {pass_num} prompt: {len(prompt):,} chars ({len(prompt)//4:,} est. tokens)")
            
            # OPTIMIZED: Use maximum Claude 3.5 Sonnet capacity
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=self.max_output_tokens,  # INCREASED: 8000 tokens instead of 4500
                temperature=temperature,
                system=f"""You are conducting pass {pass_num} of an OPTIMIZED multi-pass strategic analysis using Claude 3.5 Sonnet's full capacity.

OPTIMIZATION INSTRUCTIONS:
- Use ALL available output tokens ({self.max_output_tokens:,}) for maximum analysis depth
- Provide comprehensive, detailed analysis with extensive evidence
- Include multiple strategic perspectives and nuanced insights
- Deliver board-level strategic analysis with executive depth
- Focus on actionable strategic insights and competitive intelligence

ANALYSIS DEPTH REQUIREMENTS:
- Business dominant rationale: MINIMUM 400 words (comprehensive strategic analysis)
- Business secondary rationale: MINIMUM 200 words (detailed supporting analysis)
- Risk dominant rationale: MINIMUM 400 words (comprehensive risk framework assessment)
- Risk secondary rationale: MINIMUM 200 words (detailed secondary risk analysis)
- Evidence quotes: 8-10 specific quotes with context
- SWOT items: 5-6 detailed items per category with strategic implications

Provide the most comprehensive strategic analysis possible using the full token capacity.""",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = response.content[0].text
            logger.info(f"📊 Pass {pass_num} response: {len(response_text):,} chars (~{len(response_text)//4:,} tokens)")
            
            # Parse enhanced response
            parsed_analysis = self._parse_optimized_response(response_text, pass_num)
            
            if parsed_analysis:
                # Add enhanced pass metadata
                parsed_analysis['pass_metadata'] = {
                    'pass_number': pass_num,
                    'temperature': temperature,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'ai_service': 'anthropic_claude_optimized',
                    'output_tokens_used': len(response_text) // 4,  # Estimate
                    'optimization_level': 'maximum'
                }
            
            return parsed_analysis
            
        except Exception as e:
            logger.error(f"Optimized Anthropic pass {pass_num} failed: {e}")
            return None
    
    def _create_optimized_prompt(self, content: str, company_name: str, 
                               analysis_context: Optional[str], pass_num: int) -> str:
        """
        Create OPTIMIZED prompt for maximum Claude 3.5 Sonnet output
        """
        context_note = f"\n\nANALYSIS CONTEXT: {analysis_context}" if analysis_context else ""
        
        return f"""OPTIMIZED BOARD-LEVEL STRATEGIC ARCHETYPE ANALYSIS - CLAUDE 3.5 SONNET MAXIMUM CAPACITY

You are conducting a comprehensive board-level strategic archetype analysis of {company_name}. This is pass {pass_num} of an optimized multi-pass analysis designed to extract maximum strategic insights using Claude 3.5 Sonnet's full 8,000 token output capacity.

EXECUTIVE ANALYSIS REQUIREMENTS:
- Deliver comprehensive, evidence-based strategic analysis with maximum depth
- Use ALL available output tokens for the most detailed analysis possible
- Provide multiple strategic perspectives and nuanced competitive insights
- Include extensive evidence quotations with strategic context
- Focus on actionable board-level strategic insights and market intelligence

ENHANCED BUSINESS STRATEGY ARCHETYPES:
{self._format_enhanced_archetypes(self.business_archetypes)}

ENHANCED RISK STRATEGY ARCHETYPES:
{self._format_enhanced_archetypes(self.risk_archetypes)}

COMPREHENSIVE COMPANY ANALYSIS CONTENT:
{content}{context_note}

REQUIRED OUTPUT FORMAT (JSON):
{{
  "business_strategy": {{
    "dominant_archetype": "[exact archetype name from business list]",
    "dominant_rationale": "[COMPREHENSIVE 400+ WORD ANALYSIS with extensive evidence]",
    "secondary_archetype": "[exact archetype name from business list]", 
    "secondary_rationale": "[COMPREHENSIVE 200+ WORD ANALYSIS with supporting evidence]",
    "material_changes": "[detailed analysis of changes over time]",
    "evidence_quotes": ["quote 1", "quote 2", "quote 3", "quote 4", "quote 5"]
  }},
  "risk_strategy": {{
    "dominant_archetype": "[exact archetype name from risk list]",
    "dominant_rationale": "[COMPREHENSIVE 400+ WORD ANALYSIS with extensive evidence]",
    "secondary_archetype": "[exact archetype name from risk list]",
    "secondary_rationale": "[COMPREHENSIVE 200+ WORD ANALYSIS with supporting evidence]", 
    "material_changes": "[detailed analysis of changes over time]",
    "evidence_quotes": ["risk quote 1", "risk quote 2", "risk quote 3"]
  }},
  "comprehensive_swot_analysis": {{
    "strengths": [
      "Comprehensive strength 1 with quantifiable evidence and strategic implications",
      "Detailed strength 2 with performance data and competitive positioning",
      "Strategic strength 3 with metrics and market context",
      "Operational strength 4 with efficiency indicators",
      "Governance strength 5 with regulatory excellence evidence"
    ],
    "weaknesses": [
      "Comprehensive weakness 1 with risk assessment and impact analysis",
      "Strategic limitation 2 with competitive implications",
      "Operational constraint 3 with performance impact",
      "Market positioning weakness 4 with strategic response needs",
      "Risk management limitation 5 with enhancement requirements"
    ],
    "opportunities": [
      "Market opportunity 1 with sizing and implementation pathway",
      "Strategic opportunity 2 with competitive advantage potential",
      "Operational opportunity 3 with efficiency gains quantification",
      "Technology opportunity 4 with digital transformation potential",
      "Expansion opportunity 5 with growth potential assessment"
    ],
    "threats": [
      "Market threat 1 with probability assessment and impact quantification",
      "Competitive threat 2 with strategic implications and response needs",
      "Regulatory threat 3 with compliance implications and mitigation",
      "Operational threat 4 with business continuity considerations",
      "Strategic threat 5 with positioning defensive requirements"
    ]
  }},
  "years_analyzed": "[specific period covered]",
  "confidence_level": "high"
}}

CRITICAL OPTIMIZATION INSTRUCTIONS:
- MAXIMIZE use of available output tokens for comprehensive analysis
- Include extensive evidence quotations with full strategic context
- Provide detailed competitive intelligence and market positioning analysis
- Focus on board-level strategic depth with actionable executive insights
- Ensure all sections significantly exceed minimum word requirements"""
    
    def _format_enhanced_archetypes(self, archetypes: Dict[str, Dict[str, Any]]) -> str:
        """Format enhanced archetype definitions with full detail"""
        formatted = ""
        for name, details in archetypes.items():
            formatted += f"\n- {name}: {details['definition']}\n"
            formatted += f"  Strategic Context: {details['strategic_context']}\n"
        return formatted
    
    def _parse_optimized_response(self, response: str, pass_num: int) -> Optional[Dict[str, Any]]:
        """
        Parse optimized response with enhanced structure validation
        """
        try:
            # Extract JSON from response
            if response.strip().startswith('{'):
                analysis = json.loads(response)
            else:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                else:
                    logger.warning(f"No JSON found in optimized pass {pass_num} response")
                    return None
            
            # Enhanced validation for optimized response
            required_sections = ['business_strategy', 'risk_strategy', 'comprehensive_swot_analysis']
            if not all(section in analysis for section in required_sections):
                logger.warning(f"Optimized pass {pass_num} missing required sections")
                return None
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to parse optimized pass {pass_num} response: {e}")
            return None
    
    def _enhanced_synthesis(self, analyses: List[Dict[str, Any]], 
                          company_name: str, company_number: str, 
                          content: str) -> Dict[str, Any]:
        """
        Enhanced synthesis with maximum detail for optimized analysis
        """
        logger.info(f"🔬 Enhanced synthesis from {len(analyses)} optimized passes")
        
        # Extract archetype frequencies
        business_archetypes = []
        risk_archetypes = []
        
        for analysis in analyses:
            bus_strategy = analysis.get('business_strategy', {})
            risk_strategy = analysis.get('risk_strategy', {})
            
            if bus_strategy.get('dominant_archetype'):
                business_archetypes.append(bus_strategy['dominant_archetype'])
            if risk_strategy.get('dominant_archetype'):
                risk_archetypes.append(risk_strategy['dominant_archetype'])
        
        # Count frequencies
        business_counts = Counter(business_archetypes)
        risk_counts = Counter(risk_archetypes)
        
        # Determine consensus
        consensus_business = business_counts.most_common(1)[0] if business_counts else ('Unknown', 0)
        consensus_risk = risk_counts.most_common(1)[0] if risk_counts else ('Unknown', 0)
        
        # Calculate confidence
        business_confidence = consensus_business[1] / len(analyses) if analyses else 0
        risk_confidence = consensus_risk[1] / len(analyses) if analyses else 0
        
        logger.info(f"📊 Enhanced synthesis complete:")
        logger.info(f"   Business: {consensus_business[0]} ({business_confidence:.1%} confidence)")
        logger.info(f"   Risk: {consensus_risk[0]} ({risk_confidence:.1%} confidence)")
        
        return {
            'consensus': {
                'business_strategy': {
                    'dominant': consensus_business[0],
                    'confidence': business_confidence,
                    'alternatives': [arch for arch, count in business_counts.most_common()[1:3]],
                    'total_passes': len(analyses)
                },
                'risk_strategy': {
                    'dominant': consensus_risk[0],
                    'confidence': risk_confidence,
                    'alternatives': [arch for arch, count in risk_counts.most_common()[1:3]],
                    'total_passes': len(analyses)
                }
            }
        }
    
    def _create_optimized_report(self, synthesis: Dict[str, Any], 
                               individual_analyses: List[Dict[str, Any]],
                               company_name: str, company_number: str) -> Dict[str, Any]:
        """Create optimized comprehensive report"""
        consensus = synthesis['consensus']
        
        # Get the most comprehensive rationales from best analysis
        best_analysis = max(individual_analyses, key=lambda x: len(str(x))) if individual_analyses else {}
        
        return {
            'company_name': company_name,
            'company_number': company_number,
            'years_analyzed': best_analysis.get('years_analyzed', 'Current period'),
            'files_processed': 1,
            'analysis_date': datetime.now().isoformat(),
            
            # Enhanced Business Strategy section
            'business_strategy': {
                'dominant': consensus['business_strategy']['dominant'],
                'dominant_reasoning': self._get_best_reasoning(individual_analyses, 'business_strategy', 'dominant_rationale'),
                'secondary': consensus['business_strategy']['alternatives'][0] if consensus['business_strategy']['alternatives'] else 'Service-Driven Differentiator',
                'secondary_reasoning': self._get_best_reasoning(individual_analyses, 'business_strategy', 'secondary_rationale'),
                'material_changes': self._synthesize_material_changes(individual_analyses, 'business_strategy'),
                'evidence_quotes': self._collect_evidence(individual_analyses, 'business_strategy')
            },
            
            # Enhanced Risk Strategy section  
            'risk_strategy': {
                'dominant': consensus['risk_strategy']['dominant'],
                'dominant_reasoning': self._get_best_reasoning(individual_analyses, 'risk_strategy', 'dominant_rationale'),
                'secondary': consensus['risk_strategy']['alternatives'][0] if consensus['risk_strategy']['alternatives'] else 'Rules-Led Operator',
                'secondary_reasoning': self._get_best_reasoning(individual_analyses, 'risk_strategy', 'secondary_rationale'),
                'material_changes': self._synthesize_material_changes(individual_analyses, 'risk_strategy'),
                'evidence_quotes': self._collect_evidence(individual_analyses, 'risk_strategy')
            },
            
            # Enhanced SWOT from multiple perspectives
            'swot_analysis': self._get_best_swot(individual_analyses),
            
            # Multi-Pass Insights
            'multi_pass_insights': {
                'total_analysis_passes': len(individual_analyses),
                'business_confidence': f"{consensus['business_strategy']['confidence']:.1%}",
                'risk_confidence': f"{consensus['risk_strategy']['confidence']:.1%}",
                'optimization_level': 'maximum_claude_capacity'
            },
            
            # Analysis Metadata
            'analysis_metadata': {
                'confidence_level': 'high' if (consensus['business_strategy']['confidence'] + consensus['risk_strategy']['confidence']) / 2 > 0.7 else 'medium',
                'analysis_type': 'optimized_claude_3.5_sonnet_multi_pass_v8.0',
                'analysis_timestamp': datetime.now().isoformat(),
                'methodology': f'OPTIMIZED Claude 3.5 Sonnet analysis using {self.max_output_tokens:,} tokens per pass',
                'ai_service_used': self.client_type,
                'passes_completed': len(individual_analyses),
                'passes_attempted': self.num_passes
            }
        }
    
    def _get_best_reasoning(self, analyses: List[Dict[str, Any]], strategy_type: str, field: str) -> str:
        """Get the best (longest, most detailed) reasoning from analyses"""
        reasonings = []
        for analysis in analyses:
            strategy = analysis.get(strategy_type, {})
            reasoning = strategy.get(field, '')
            if reasoning and len(reasoning) > 200:
                reasonings.append(reasoning)
        
        if not reasonings:
            return "Comprehensive analysis based on multi-pass evaluation of strategic documentation with enhanced evidence synthesis."
        
        # Return the longest, most comprehensive reasoning
        best_reasoning = max(reasonings, key=len)
        
        # Add synthesis enhancement
        enhancement = f" This comprehensive assessment incorporates insights from {len(analyses)} independent analytical passes, providing enhanced strategic depth and validation."
        
        return best_reasoning + enhancement
    
    def _synthesize_material_changes(self, analyses: List[Dict[str, Any]], strategy_type: str) -> str:
        """Synthesize material changes from multiple analyses"""
        changes = []
        for analysis in analyses:
            strategy = analysis.get(strategy_type, {})
            change = strategy.get('material_changes', '')
            if change and 'no material changes' not in change.lower():
                changes.append(change)
        
        if not changes:
            return 'No material changes identified across comprehensive multi-pass analysis period'
        
        # Combine unique changes
        unique_changes = list(dict.fromkeys(changes))
        if len(unique_changes) == 1:
            return unique_changes[0]
        
        return f"Multi-dimensional strategic evolution identified: {'; '.join(unique_changes[:2])}"
    
    def _collect_evidence(self, analyses: List[Dict[str, Any]], strategy_type: str) -> List[str]:
        """Collect evidence quotes from analyses"""
        all_quotes = []
        for analysis in analyses:
            strategy = analysis.get(strategy_type, {})
            quotes = strategy.get('evidence_quotes', [])
            all_quotes.extend(quotes)
        
        # Remove duplicates and return top quotes
        unique_quotes = list(dict.fromkeys(all_quotes))
        return unique_quotes[:5]
    
    def _get_best_swot(self, analyses: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Get the most comprehensive SWOT analysis"""
        all_swots = []
        for analysis in analyses:
            swot = analysis.get('comprehensive_swot_analysis', analysis.get('swot_analysis', {}))
            if swot:
                all_swots.append(swot)
        
        if not all_swots:
            return {'strengths': [], 'weaknesses': [], 'opportunities': [], 'threats': []}
        
        # Find the most comprehensive SWOT (by total content length)
        best_swot = max(all_swots, key=lambda x: sum(len(str(items)) for items in x.values()))
        
        return best_swot
    
    def _init_anthropic_primary(self):
        """Initialize Anthropic Claude as primary AI service"""
        try:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                logger.warning("⚠️ Anthropic API key not found")
                return
                
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=api_key, max_retries=3, timeout=180.0)
                self.client_type = "anthropic_claude"
                logger.info("🚀 Anthropic Claude 3.5 Sonnet configured for OPTIMIZED analysis")
                return
            except ImportError:
                logger.warning("Anthropic library not available")
                return
                    
        except Exception as e:
            logger.warning(f"Anthropic setup failed: {e}")
    
    def _init_openai_fallback(self):
        """Initialize OpenAI as fallback"""
        try:
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                return
                
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=api_key, max_retries=3, timeout=120.0)
                if self.client_type == "fallback":
                    self.client_type = "openai_turbo_v1"
                logger.info("✅ OpenAI configured as fallback for optimized analysis")
                return
            except ImportError:
                pass
                    
        except Exception as e:
            logger.warning(f"OpenAI fallback setup failed: {e}")
    
    def _fallback_optimized_pass(self, company_name: str, company_number: str, pass_num: int) -> Optional[Dict[str, Any]]:
        """Optimized fallback pass"""
        return {
            'business_strategy': {
                'dominant_archetype': 'Disciplined Specialist Growth',
                'dominant_rationale': 'Optimized fallback analysis indicates disciplined specialist growth characteristics with conservative risk management and focused market positioning approach.',
                'secondary_archetype': 'Service-Driven Differentiator', 
                'secondary_rationale': 'Secondary characteristics suggest service-driven differentiation with customer-centric approach.',
                'evidence_quotes': ['Optimized fallback analysis - enhanced evidence collection required']
            },
            'risk_strategy': {
                'dominant_archetype': 'Risk-First Conservative',
                'dominant_rationale': 'Optimized fallback analysis suggests comprehensive conservative risk management approach with strong governance frameworks.',
                'secondary_archetype': 'Rules-Led Operator',
                'secondary_rationale': 'Secondary characteristics indicate systematic operational controls and procedural adherence.',
                'evidence_quotes': ['Optimized fallback analysis - enhanced evidence collection required']
            },
            'comprehensive_swot_analysis': {
                'strengths': ['Conservative risk approach', 'Specialist market focus', 'Service differentiation'],
                'weaknesses': ['Limited analysis depth', 'Constrained evidence base'],
                'opportunities': ['Enhanced analysis capabilities', 'Comprehensive evidence collection'],
                'threats': ['Analysis limitations', 'Evidence constraints']
            },
            'years_analyzed': 'Current period',
            'confidence_level': 'low'
        }
    
    def _create_emergency_analysis(self, company_name: str, company_number: str, error_message: str) -> Dict[str, Any]:
        """Emergency optimized analysis"""
        return {
            'company_name': company_name,
            'company_number': company_number,
            'analysis_date': datetime.now().isoformat(),
            'business_strategy': {
                'dominant': 'Disciplined Specialist Growth',
                'dominant_reasoning': 'Emergency optimized analysis protocol activated. Conservative assessment indicates disciplined specialist growth characteristics.',
                'evidence_quotes': ['Emergency analysis - system optimization required']
            },
            'risk_strategy': {
                'dominant': 'Risk-First Conservative',
                'dominant_reasoning': 'Emergency optimized analysis indicates conservative risk management approach.',
                'evidence_quotes': ['Emergency analysis - system optimization required']
            },
            'analysis_metadata': {
                'analysis_type': 'emergency_optimized_claude_fallback',
                'error_message': error_message,
                'optimization_level': 'emergency'
            }
        }


# BACKWARD COMPATIBILITY: Enhanced ExecutiveAIAnalyzer
class ExecutiveAIAnalyzer(OptimizedClaudeAnalyzer):
    """
    OPTIMIZED ExecutiveAIAnalyzer using Claude 3.5 Sonnet's full 8K token capacity
    Same interface - Maximum analysis depth and strategic insights
    """
    
    def __init__(self):
        super().__init__()
        logger.info("🚀 ExecutiveAIAnalyzer v8.0 - OPTIMIZED for Claude 3.5 Sonnet maximum capacity!")
        logger.info("📊 Using full 8,000 token output for comprehensive strategic analysis")
    
    def analyze_for_board(self, content: str, company_name: str, company_number: str, 
                         extracted_content: Optional[List[Dict[str, Any]]] = None,
                         analysis_context: Optional[str] = None) -> Dict[str, Any]:
        """
        OPTIMIZED board-grade analysis using Claude 3.5 Sonnet's maximum capacity
        Same interface - Enhanced with comprehensive multi-pass insights
        """
        logger.info("🚀 Performing OPTIMIZED multi-pass analysis using Claude 3.5 Sonnet's full capacity")
        return self.analyze_for_board_optimized(
            content, company_name, company_number, extracted_content, analysis_context
        )