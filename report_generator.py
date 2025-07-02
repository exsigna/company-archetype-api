#!/usr/bin/env python3
"""
Report Generator for Strategic Analysis Tool
Generates comprehensive reports from archetype analysis results
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from config import (
    BUSINESS_STRATEGY_ARCHETYPES, RISK_STRATEGY_ARCHETYPES
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates various types of reports from analysis results"""
    
    def __init__(self):
        """Initialize the report generator"""
        self.business_archetypes = BUSINESS_STRATEGY_ARCHETYPES
        self.risk_archetypes = RISK_STRATEGY_ARCHETYPES
        logger.info("Report generator initialized")
    
    def generate_analysis_report(self, portfolio_analysis: Dict[str, Any]) -> str:
        """
        Generate comprehensive analysis report
        
        Args:
            portfolio_analysis: Analysis results from the main analysis
            
        Returns:
            Formatted report as string
        """
        try:
            company_name = portfolio_analysis.get('company_name', 'Unknown Company')
            company_number = portfolio_analysis.get('company_number', 'Unknown')
            
            archetype_analysis = portfolio_analysis.get('archetype_analysis', {})
            business_archetypes = archetype_analysis.get('business_strategy_archetypes', {})
            risk_archetypes = archetype_analysis.get('risk_strategy_archetypes', {})
            
            report_lines = []
            
            # Header
            report_lines.extend([
                "="*80,
                "STRATEGIC ARCHETYPE CLASSIFICATION REPORT",
                "="*80,
                "",
                f"Company: {company_name}",
                f"Registration Number: {company_number}",
                f"Analysis Date: {datetime.now().strftime('%d %B %Y')}",
                f"Report Generated: {datetime.now().strftime('%d %B %Y at %H:%M')}"
            ])
            
            # Analysis methodology
            analysis_method = archetype_analysis.get('analysis_type', 'pattern_based')
            model_used = archetype_analysis.get('model_used', 'N/A')
            
            report_lines.extend([
                "",
                "ANALYSIS METHODOLOGY",
                "-" * 40,
                f"Analysis Method: {analysis_method.replace('_', ' ').title()}",
                f"AI Model Used: {model_used}" if model_used != 'N/A' else "Analysis Type: Pattern-based classification",
                f"Documents Processed: {portfolio_analysis.get('files_analyzed', 0)}",
                f"Analysis Timestamp: {archetype_analysis.get('timestamp', 'Unknown')}"
            ])
            
            # Business Strategy Archetypes
            report_lines.extend([
                "",
                "",
                "BUSINESS STRATEGY ARCHETYPE CLASSIFICATION",
                "="*60,
                ""
            ])
            
            dominant_business = business_archetypes.get('dominant', 'Unknown')
            secondary_business = business_archetypes.get('secondary', '')
            
            report_lines.extend([
                "PRIMARY CLASSIFICATION",
                "-" * 30,
                f"Dominant Archetype: {dominant_business}",
                ""
            ])
            
            # Add definition for dominant business archetype
            if dominant_business in self.business_archetypes:
                report_lines.extend([
                    "Definition:",
                    self._wrap_text(self.business_archetypes[dominant_business], 70),
                    ""
                ])
            
            if secondary_business:
                report_lines.extend([
                    "SECONDARY CLASSIFICATION", 
                    "-" * 30,
                    f"Secondary Archetype: {secondary_business}",
                    ""
                ])
                
                if secondary_business in self.business_archetypes:
                    report_lines.extend([
                        "Definition:",
                        self._wrap_text(self.business_archetypes[secondary_business], 70),
                        ""
                    ])
            
            # Business strategy reasoning
            business_reasoning = business_archetypes.get('reasoning', 'No detailed analysis available')
            report_lines.extend([
                "ANALYSIS RATIONALE",
                "-" * 30,
                self._wrap_text(business_reasoning, 70),
                ""
            ])
            
            # Risk Strategy Archetypes
            report_lines.extend([
                "",
                "RISK STRATEGY ARCHETYPE CLASSIFICATION", 
                "="*60,
                ""
            ])
            
            dominant_risk = risk_archetypes.get('dominant', 'Unknown')
            secondary_risk = risk_archetypes.get('secondary', '')
            
            report_lines.extend([
                "PRIMARY CLASSIFICATION",
                "-" * 30,
                f"Dominant Archetype: {dominant_risk}",
                ""
            ])
            
            # Add definition for dominant risk archetype
            if dominant_risk in self.risk_archetypes:
                report_lines.extend([
                    "Definition:",
                    self._wrap_text(self.risk_archetypes[dominant_risk], 70),
                    ""
                ])
            
            if secondary_risk:
                report_lines.extend([
                    "SECONDARY CLASSIFICATION",
                    "-" * 30, 
                    f"Secondary Archetype: {secondary_risk}",
                    ""
                ])
                
                if secondary_risk in self.risk_archetypes:
                    report_lines.extend([
                        "Definition:",
                        self._wrap_text(self.risk_archetypes[secondary_risk], 70),
                        ""
                    ])
            
            # Risk strategy reasoning
            risk_reasoning = risk_archetypes.get('reasoning', 'No detailed analysis available')
            report_lines.extend([
                "ANALYSIS RATIONALE",
                "-" * 30,
                self._wrap_text(risk_reasoning, 70),
                ""
            ])
            
            # Strategic Profile Summary
            report_lines.extend([
                "",
                "STRATEGIC PROFILE SUMMARY",
                "="*60,
                "",
                f"{company_name} demonstrates characteristics primarily aligned with:",
                f"• Business Strategy: {dominant_business}",
                f"• Risk Strategy: {dominant_risk}",
                ""
            ])
            
            # Generate strategic insights
            insights = self._generate_strategic_insights(dominant_business, dominant_risk)
            if insights:
                report_lines.extend([
                    "STRATEGIC INSIGHTS",
                    "-" * 30,
                    self._wrap_text(insights, 70),
                    ""
                ])
            
            # Analysis Quality Metrics
            files_analyzed = portfolio_analysis.get('files_analyzed', 0)
            files_successful = portfolio_analysis.get('files_successful', 0)
            
            report_lines.extend([
                "",
                "ANALYSIS QUALITY METRICS",
                "="*40,
                f"Documents Analyzed: {files_analyzed}",
                f"Successful Extractions: {files_successful}",
                f"Success Rate: {(files_successful/max(files_analyzed,1)*100):.1f}%",
                f"Analysis Confidence: {'High' if files_successful >= 3 else 'Medium' if files_successful >= 2 else 'Low'}",
                ""
            ])
            
            # File Analysis Details
            file_analyses = portfolio_analysis.get('file_analyses', [])
            if file_analyses:
                report_lines.extend([
                    "DOCUMENT PROCESSING DETAILS",
                    "="*40
                ])
                
                for i, file_analysis in enumerate(file_analyses, 1):
                    filename = file_analysis.get('filename', f'Document {i}')
                    extraction_method = file_analysis.get('extraction_method', 'Unknown')
                    extraction_status = file_analysis.get('extraction_status', 'Unknown')
                    
                    report_lines.extend([
                        f"",
                        f"Document {i}: {filename}",
                        f"   Extraction Method: {extraction_method}",
                        f"   Status: {extraction_status}"
                    ])
                
                report_lines.append("")
            
            # Recommendations
            recommendations = self._generate_recommendations(dominant_business, dominant_risk, analysis_method)
            if recommendations:
                report_lines.extend([
                    "RECOMMENDATIONS",
                    "="*40,
                    self._wrap_text(recommendations, 70),
                    ""
                ])
            
            # Footer
            report_lines.extend([
                "",
                "="*80,
                "END OF REPORT",
                f"Generated by Strategic Archetype Analysis Tool on {datetime.now().strftime('%d %B %Y at %H:%M')}",
                "="*80
            ])
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Error generating analysis report: {e}")
            return f"Error generating report: {str(e)}"
    
    def generate_executive_summary(self, portfolio_analysis: Dict[str, Any]) -> str:
        """
        Generate executive summary
        
        Args:
            portfolio_analysis: Analysis results
            
        Returns:
            Executive summary as string
        """
        try:
            company_name = portfolio_analysis.get('company_name', 'Unknown Company')
            company_number = portfolio_analysis.get('company_number', 'Unknown')
            
            archetype_analysis = portfolio_analysis.get('archetype_analysis', {})
            business_archetypes = archetype_analysis.get('business_strategy_archetypes', {})
            risk_archetypes = archetype_analysis.get('risk_strategy_archetypes', {})
            
            dominant_business = business_archetypes.get('dominant', 'Unknown')
            dominant_risk = risk_archetypes.get('dominant', 'Unknown')
            
            summary_lines = [
                "EXECUTIVE SUMMARY",
                "="*50,
                "",
                f"Company: {company_name} ({company_number})",
                f"Analysis Date: {datetime.now().strftime('%d %B %Y')}",
                "",
                "KEY FINDINGS",
                "-"*20,
                f"• Business Strategy Archetype: {dominant_business}",
                f"• Risk Strategy Archetype: {dominant_risk}",
                "",
                "STRATEGIC PROFILE",
                "-"*20
            ]
            
            # Add brief archetype descriptions
            if dominant_business in self.business_archetypes:
                summary_lines.extend([
                    f"Business Strategy: {self.business_archetypes[dominant_business][:100]}...",
                    ""
                ])
            
            if dominant_risk in self.risk_archetypes:
                summary_lines.extend([
                    f"Risk Strategy: {self.risk_archetypes[dominant_risk][:100]}...",
                    ""
                ])
            
            # Key insights
            insights = self._generate_strategic_insights(dominant_business, dominant_risk)
            if insights:
                summary_lines.extend([
                    "KEY INSIGHTS",
                    "-"*20,
                    self._wrap_text(insights[:200] + "...", 60),
                    ""
                ])
            
            # Analysis quality
            files_analyzed = portfolio_analysis.get('files_analyzed', 0)
            analysis_method = archetype_analysis.get('analysis_type', 'pattern_based')
            
            summary_lines.extend([
                "ANALYSIS OVERVIEW",
                "-"*20,
                f"Documents Analyzed: {files_analyzed}",
                f"Analysis Method: {analysis_method.replace('_', ' ').title()}",
                f"Confidence Level: {'High' if files_analyzed >= 3 else 'Medium' if files_analyzed >= 2 else 'Low'}",
                "",
                "="*50,
                f"Report generated: {datetime.now().strftime('%d %B %Y at %H:%M')}"
            ])
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return f"Error generating summary: {str(e)}"
    
    def generate_json_summary(self, portfolio_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate JSON summary of analysis
        
        Args:
            portfolio_analysis: Analysis results
            
        Returns:
            Summary data as dictionary
        """
        try:
            archetype_analysis = portfolio_analysis.get('archetype_analysis', {})
            business_archetypes = archetype_analysis.get('business_strategy_archetypes', {})
            risk_archetypes = archetype_analysis.get('risk_strategy_archetypes', {})
            
            summary = {
                "company_info": {
                    "name": portfolio_analysis.get('company_name', 'Unknown'),
                    "number": portfolio_analysis.get('company_number', 'Unknown'),
                    "analysis_date": datetime.now().isoformat()
                },
                "archetype_classification": {
                    "business_strategy": {
                        "dominant": business_archetypes.get('dominant', 'Unknown'),
                        "secondary": business_archetypes.get('secondary', ''),
                        "definition": self.business_archetypes.get(business_archetypes.get('dominant', ''), ''),
                        "reasoning": business_archetypes.get('reasoning', '')
                    },
                    "risk_strategy": {
                        "dominant": risk_archetypes.get('dominant', 'Unknown'),
                        "secondary": risk_archetypes.get('secondary', ''),
                        "definition": self.risk_archetypes.get(risk_archetypes.get('dominant', ''), ''),
                        "reasoning": risk_archetypes.get('reasoning', '')
                    }
                },
                "analysis_metadata": {
                    "method": archetype_analysis.get('analysis_type', 'pattern_based'),
                    "model_used": archetype_analysis.get('model_used', 'N/A'),
                    "documents_analyzed": portfolio_analysis.get('files_analyzed', 0),
                    "success_rate": portfolio_analysis.get('files_successful', 0) / max(portfolio_analysis.get('files_analyzed', 1), 1),
                    "confidence": 'high' if portfolio_analysis.get('files_analyzed', 0) >= 3 else 'medium' if portfolio_analysis.get('files_analyzed', 0) >= 2 else 'low'
                },
                "strategic_insights": self._generate_strategic_insights(
                    business_archetypes.get('dominant', ''),
                    risk_archetypes.get('dominant', '')
                ),
                "recommendations": self._generate_recommendations(
                    business_archetypes.get('dominant', ''),
                    risk_archetypes.get('dominant', ''),
                    archetype_analysis.get('analysis_type', 'pattern_based')
                ),
                "generated_timestamp": datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating JSON summary: {e}")
            return {"error": str(e)}
    
    def _wrap_text(self, text: str, width: int) -> str:
        """
        Wrap text to specified width
        
        Args:
            text: Text to wrap
            width: Maximum line width
            
        Returns:
            Wrapped text
        """
        if not text:
            return ""
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= width:
                current_line = current_line + " " + word if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return "\n".join(lines)
    
    def _generate_strategic_insights(self, business_archetype: str, risk_archetype: str) -> str:
        """
        Generate strategic insights based on archetype combination
        
        Args:
            business_archetype: Dominant business strategy archetype
            risk_archetype: Dominant risk strategy archetype
            
        Returns:
            Strategic insights text
        """
        try:
            insights = []
            
            # Business archetype insights
            if business_archetype == "Disciplined Specialist Growth":
                insights.append("The company focuses on sustainable growth within its specialist niche, leveraging deep underwriting expertise.")
            elif business_archetype == "Balance-Sheet Steward":
                insights.append("Strong emphasis on capital preservation and financial stability, typical of mutual organisations or building societies.")
            elif business_archetype == "Service-Driven Differentiator":
                insights.append("Competitive advantage derived from superior customer experience rather than pricing strategies.")
            elif business_archetype == "Tech-Productivity Accelerator":
                insights.append("Heavy investment in technology and automation to drive operational efficiency and reduce unit costs.")
            
            # Risk archetype insights
            if risk_archetype == "Risk-First Conservative":
                insights.append("Risk management prioritises capital preservation and regulatory compliance over aggressive growth.")
            elif risk_archetype == "Rules-Led Operator":
                insights.append("Strong emphasis on procedural consistency and regulatory adherence in risk management approach.")
            elif risk_archetype == "Embedded Risk Partner":
                insights.append("Risk management is integrated into business decision-making rather than operating as a separate function.")
            elif risk_archetype == "Resilience-Focused Architect":
                insights.append("Risk strategy emphasises operational continuity and crisis preparedness.")
            
            # Combination insights
            if business_archetype == "Disciplined Specialist Growth" and risk_archetype == "Risk-First Conservative":
                insights.append("This combination suggests a mature, specialist lender with strong risk discipline - well-positioned for regulatory scrutiny.")
            elif "Service-Driven" in business_archetype and "Embedded Risk" in risk_archetype:
                insights.append("Customer-centric approach balanced with integrated risk management supports sustainable differentiation.")
            
            return " ".join(insights) if insights else "Strategic archetype combination indicates a balanced approach to growth and risk management."
            
        except Exception as e:
            logger.error(f"Error generating strategic insights: {e}")
            return "Analysis completed but detailed insights not available."
    
    def _generate_recommendations(self, business_archetype: str, risk_archetype: str, analysis_method: str) -> str:
        """
        Generate recommendations based on analysis
        
        Args:
            business_archetype: Dominant business strategy archetype
            risk_archetype: Dominant risk strategy archetype
            analysis_method: Method used for analysis
            
        Returns:
            Recommendations text
        """
        try:
            recommendations = []
            
            # General recommendations
            recommendations.append("1. Review archetype classification against current strategic plans to identify alignment or gaps.")
            
            # Business strategy recommendations
            if "Growth" in business_archetype:
                recommendations.append("2. Ensure risk management frameworks can support growth ambitions without compromising control standards.")
            elif "Steward" in business_archetype:
                recommendations.append("2. Consider opportunities for measured growth that align with stewardship principles.")
            
            # Risk strategy recommendations
            if "Conservative" in risk_archetype:
                recommendations.append("3. Balance conservative risk approach with market opportunities to optimise returns.")
            elif "Partner" in risk_archetype:
                recommendations.append("3. Leverage embedded risk capabilities to support business innovation and expansion.")
            
            # Analysis quality recommendations
            if analysis_method == "pattern_archetype_classification":
                recommendations.append("4. Consider upgrading to AI-powered analysis for more detailed archetype insights and reasoning.")
            
            recommendations.append("5. Conduct peer benchmarking to compare archetype patterns with industry competitors.")
            recommendations.append("6. Use archetype insights to inform board discussions on strategic direction and risk appetite.")
            
            return " ".join(recommendations)
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return "Please review analysis results and consider strategic implications for business planning."


# Convenience functions
def generate_company_report(portfolio_analysis: Dict[str, Any]) -> str:
    """
    Quick function to generate a company report
    
    Args:
        portfolio_analysis: Analysis results
        
    Returns:
        Formatted report
    """
    generator = ReportGenerator()
    return generator.generate_analysis_report(portfolio_analysis)


def generate_summary(portfolio_analysis: Dict[str, Any]) -> str:
    """
    Quick function to generate an executive summary
    
    Args:
        portfolio_analysis: Analysis results
        
    Returns:
        Executive summary
    """
    generator = ReportGenerator()
    return generator.generate_executive_summary(portfolio_analysis)


if __name__ == "__main__":
    # Test the report generator
    test_analysis = {
        "company_name": "Test Company Ltd",
        "company_number": "12345678",
        "files_analyzed": 3,
        "files_successful": 3,
        "archetype_analysis": {
            "analysis_type": "ai_archetype_classification",
            "model_used": "gpt-4",
            "business_strategy_archetypes": {
                "dominant": "Disciplined Specialist Growth",
                "secondary": "",
                "reasoning": "The company demonstrates focused growth within its specialist lending niche."
            },
            "risk_strategy_archetypes": {
                "dominant": "Risk-First Conservative",
                "secondary": "",
                "reasoning": "Strong emphasis on capital preservation and regulatory compliance."
            }
        }
    }
    
    generator = ReportGenerator()
    
    print("Testing Report Generator:")
    print("=" * 40)
    
    # Test analysis report
    report = generator.generate_analysis_report(test_analysis)
    print(f"Analysis report generated: {len(report)} characters")
    
    # Test executive summary
    summary = generator.generate_executive_summary(test_analysis)
    print(f"Executive summary generated: {len(summary)} characters")
    
    # Test JSON summary
    json_summary = generator.generate_json_summary(test_analysis)
    print(f"JSON summary generated with {len(json_summary)} fields")
    
    print("Report generator test completed successfully")