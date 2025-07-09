#!/usr/bin/env python3
"""
AI Analyzer - Version 2.0 - Complete Enhanced Multi-File Analysis
DEPLOYMENT READY - Replace your existing ai_analyzer.py with this file
"""

import logging
import json
import os
import re
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List

# Safe config import with fallbacks
try:
    from config import DEFAULT_OPENAI_MODEL, AI_MAX_TOKENS, AI_TEMPERATURE
except ImportError:
    DEFAULT_OPENAI_MODEL = "gpt-4"
    AI_MAX_TOKENS = 2000
    AI_TEMPERATURE = 0.3

logger = logging.getLogger(__name__)

class AIArchetypeAnalyzer:
    """Version 2.0 - Enhanced AI-powered analyzer with complete multi-file support"""
    
    def __init__(self):
        """Initialize the AI analyzer"""
        logger.info("🚀 AIArchetypeAnalyzer v2.0 starting...")
        self.client = None
        self.client_type = "fallback"
        
        # Business Strategy Archetypes
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
        
        # Risk Strategy Archetypes
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
        
        self._setup_client()
        logger.info(f"✅ AIArchetypeAnalyzer v2.0 completed. Client type: {self.client_type}")

    def _setup_client(self):
        """Setup the AI client"""
        try:
            openai_key = os.getenv('OPENAI_API_KEY')
            
            if openai_key and openai_key.strip() and not openai_key.startswith('your_'):
                try:
                    import openai
                    
                    # Clear proxy vars
                    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
                    for var in proxy_vars:
                        if var in os.environ:
                            del os.environ[var]
                    
                    self.client = openai.OpenAI(api_key=openai_key.strip())
                    self.client_type = "openai"
                    
                    # Test connection
                    test_response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Hi"}],
                        max_tokens=1,
                        temperature=0
                    )
                    logger.info("✅ OpenAI client initialized successfully")
                    return
                    
                except Exception as e:
                    logger.warning(f"OpenAI setup failed: {e}")
            
            # Fallback
            self.client = None
            self.client_type = "fallback"
            logger.warning("Using fallback pattern analysis")
            
        except Exception as e:
            logger.error(f"Critical error in client setup: {e}")
            self.client = None
            self.client_type = "fallback"

    def analyze_archetypes(self, content: str, company_name: str, company_number: str, 
                         extracted_content: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhanced archetype analysis with multi-file support"""
        try:
            logger.info(f"🏛️ Starting archetype analysis for {company_name}")
            
            if self.client and self.client_type == "openai":
                try:
                    # Multi-file or single analysis
                    if extracted_content and len(extracted_content) > 1:
                        logger.info("🎯 Using multi-file analysis")
                        business_analysis = self._analyze_multiple_files(
                            extracted_content, self.business_archetypes, "Business Strategy"
                        )
                        risk_analysis = self._analyze_multiple_files(
                            extracted_content, self.risk_archetypes, "Risk Strategy"
                        )
                    else:
                        logger.info("🎯 Using single content analysis")
                        business_analysis = self._classify_archetypes(
                            content, self.business_archetypes, "Business Strategy"
                        )
                        risk_analysis = self._classify_archetypes(
                            content, self.risk_archetypes, "Risk Strategy"
                        )
                    
                    return {
                        "success": True,
                        "analysis_type": "ai_multi_file_classification" if extracted_content and len(extracted_content) > 1 else "ai_archetype_classification",
                        "company_name": company_name,
                        "company_number": company_number,
                        "business_strategy_archetypes": business_analysis,
                        "risk_strategy_archetypes": risk_analysis,
                        "timestamp": datetime.now().isoformat(),
                        "model_used": f"openai_{DEFAULT_OPENAI_MODEL}",
                        "analysis_metadata": {
                            "files_analyzed": len(extracted_content) if extracted_content else 1,
                            "total_content_chars": len(content),
                            "confidence_level": "high" if extracted_content and len(extracted_content) > 1 else "medium"
                        }
                    }
                    
                except Exception as ai_error:
                    logger.error(f"AI analysis failed: {ai_error}")
                    return self._fallback_analysis(content, company_name, company_number, extracted_content)
            else:
                return self._fallback_analysis(content, company_name, company_number, extracted_content)
                
        except Exception as e:
            logger.error(f"Critical error in analyze_archetypes: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis_type": "error",
                "timestamp": datetime.now().isoformat()
            }

    def _analyze_multiple_files(self, extracted_content: List[Dict[str, Any]], 
                               archetype_dict: Dict[str, str], label: str) -> Dict[str, Any]:
        """Analyze multiple files individually and synthesize"""
        logger.info(f"📊 Multi-file analysis: {label} across {len(extracted_content)} files")
        
        individual_analyses = []
        for i, file_data in enumerate(extracted_content):
            file_content = file_data.get('content', '')[:12000]  # 12K per file
            
            try:
                analysis = self._classify_archetypes(file_content, archetype_dict, f"{label} - File {i+1}")
                analysis['source_file'] = file_data.get('filename', f'File {i+1}')
                individual_analyses.append(analysis)
            except Exception as e:
                logger.warning(f"Failed to analyze file {i+1}: {e}")
                continue
        
        if individual_analyses:
            return self._synthesize_analyses(individual_analyses, label)
        else:
            # Fallback to combined analysis
            combined = "\n\n".join([f"=== File {i+1} ===\n{f.get('content', '')[:8000]}" 
                                  for i, f in enumerate(extracted_content)])
            return self._classify_archetypes(combined, archetype_dict, label)

    def _synthesize_analyses(self, individual_analyses: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
        """Synthesize multiple analyses - KEY FIX: Include ALL reasoning"""
        dominant_counts = {}
        secondary_counts = {}
        all_reasoning = []
        
        for analysis in individual_analyses:
            dominant = analysis.get('dominant', '')
            secondary = analysis.get('secondary', '')
            
            if dominant:
                dominant_counts[dominant] = dominant_counts.get(dominant, 0) + 1
            if secondary:
                secondary_counts[secondary] = secondary_counts.get(secondary, 0) + 1
            
            # Collect FULL reasoning
            file_name = analysis.get('source_file', 'Unknown')
            reasoning = analysis.get('reasoning', '')
            if reasoning:
                all_reasoning.append(f"[{file_name}] {reasoning}")
        
        # Determine final archetypes
        final_dominant = max(dominant_counts.items(), key=lambda x: x[1])[0] if dominant_counts else "Balance-Sheet Steward"
        final_secondary = ""
        
        if secondary_counts:
            for archetype, count in sorted(secondary_counts.items(), key=lambda x: x[1], reverse=True):
                if archetype != final_dominant:
                    final_secondary = archetype
                    break
        
        # Create comprehensive reasoning
        total_files = len(individual_analyses)
        confidence = dominant_counts.get(final_dominant, 0) / total_files
        
        synthesized_reasoning = f"Multi-file analysis across {total_files} documents shows {final_dominant} as the dominant archetype "
        synthesized_reasoning += f"(appears in {dominant_counts.get(final_dominant, 0)}/{total_files} files, {confidence:.0%} confidence). "
        
        if final_secondary:
            synthesized_reasoning += f"Secondary archetype {final_secondary} identified in {secondary_counts.get(final_secondary, 0)} files. "
        
        # KEY FIX: Include ALL reasoning, not truncated
        if all_reasoning:
            synthesized_reasoning += "Key evidence: " + "; ".join(all_reasoning)
        
        return {
            "dominant": final_dominant,
            "secondary": final_secondary,
            "reasoning": synthesized_reasoning,
            "confidence_score": confidence
        }

    def _classify_archetypes(self, content: str, archetype_dict: Dict[str, str], label: str) -> Dict[str, Any]:
        """Classify archetypes using OpenAI"""
        content_sample = content[:15000]  # Use 15K chars
        
        archetypes_text = "\n".join([f"- {name}: {definition}" for name, definition in archetype_dict.items()])
        
        prompt = f"""Analyze this UK financial services firm and identify the dominant and secondary {label} archetypes.

Available {label} Archetypes:
{archetypes_text}

Provide detailed evidence-based reasoning.

Format:
**Dominant:** <archetype_name>
**Secondary:** <archetype_name or "None">
**Reasoning:** <detailed explanation>

Content:
{content_sample}"""

        try:
            response = self.client.chat.completions.create(
                model=DEFAULT_OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": f"You are an expert {label.lower()} analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=AI_TEMPERATURE
            )
            
            return self._parse_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._fallback_single_analysis(content, archetype_dict, label)

    def _parse_response(self, response: str) -> Dict[str, str]:
        """Parse AI response"""
        result = {"dominant": "", "secondary": "", "reasoning": ""}
        
        lines = response.strip().split('\n')
        reasoning_lines = []
        reasoning_started = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("**Dominant:**") or line.startswith("Dominant:"):
                value = re.sub(r'\*+', '', line.replace("Dominant:", "")).strip()
                result["dominant"] = value
                
            elif line.startswith("**Secondary:**") or line.startswith("Secondary:"):
                value = re.sub(r'\*+', '', line.replace("Secondary:", "")).strip()
                result["secondary"] = value if value.lower() != "none" else ""
                
            elif line.startswith("**Reasoning:**") or line.startswith("Reasoning:"):
                value = re.sub(r'\*+', '', line.replace("Reasoning:", "")).strip()
                if value:
                    result["reasoning"] = value
                else:
                    reasoning_started = True
                    
            elif reasoning_started and line:
                reasoning_lines.append(line)
        
        if reasoning_lines:
            result["reasoning"] = ' '.join(reasoning_lines)
        
        return result

    def _fallback_analysis(self, content: str, company_name: str, company_number: str, 
                          extracted_content: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Pattern-based fallback analysis"""
        business_analysis = self._fallback_single_analysis(content, self.business_archetypes, "Business Strategy")
        risk_analysis = self._fallback_single_analysis(content, self.risk_archetypes, "Risk Strategy")
        
        return {
            "success": True,
            "analysis_type": "pattern_fallback",
            "company_name": company_name,
            "company_number": company_number,
            "business_strategy_archetypes": business_analysis,
            "risk_strategy_archetypes": risk_analysis,
            "timestamp": datetime.now().isoformat()
        }

    def _fallback_single_analysis(self, content: str, archetype_dict: Dict[str, str], label: str) -> Dict[str, str]:
        """Pattern-based analysis"""
        content_lower = content.lower()
        scores = {}
        
        # Simple keyword patterns
        patterns = {
            'Disciplined Specialist Growth': ['specialist', 'niche', 'underwriting'],
            'Balance-Sheet Steward': ['capital', 'prudent', 'conservative'],
            'Resilience-Focused Architect': ['stress.*testing', 'resilience', 'continuity'],
            'Risk-First Conservative': ['capital.*preservation', 'compliance']
        }
        
        for archetype in archetype_dict.keys():
            if archetype in patterns:
                score = sum(len(re.findall(pattern, content_lower)) for pattern in patterns[archetype])
                scores[archetype] = score
        
        # Default to common archetypes if no matches
        if not scores or max(scores.values()) == 0:
            if label == "Business Strategy":
                dominant = "Disciplined Specialist Growth"
            else:
                dominant = "Resilience-Focused Architect"
        else:
            dominant = max(scores.items(), key=lambda x: x[1])[0]
        
        return {
            "dominant": dominant,
            "secondary": "",
            "reasoning": f"Pattern-based analysis identified {dominant} based on keyword frequency analysis."
        }

# Legacy compatibility
def analyze_company_archetypes(content: str, company_name: str, company_number: str) -> Dict[str, Any]:
    """Legacy compatibility function"""
    analyzer = AIArchetypeAnalyzer()
    return analyzer.analyze_archetypes(content, company_name, company_number)

# Enhanced multi-file function
def analyze_company_archetypes_multi_file(content: str, company_name: str, company_number: str, 
                                        extracted_content: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Enhanced multi-file analysis function"""
    analyzer = AIArchetypeAnalyzer()
    return analyzer.analyze_archetypes(content, company_name, company_number, extracted_content)