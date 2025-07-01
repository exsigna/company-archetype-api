#!/usr/bin/env python3
"""
Report generation module
Creates various output formats for archetype analysis results
"""

from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path


class ReportGenerator:
    """Generates archetype analysis reports in various formats"""
    
    def __init__(self):
        pass
    
    def generate_analysis_report(self, portfolio_analysis: Dict, 
                               company_profile: Optional[Dict] = None) -> str:
        """
        Generate a comprehensive archetype analysis report in plain text format
        
        Args:
            portfolio_analysis: Results from portfolio analysis containing archetype data
            company_profile: Optional company profile from Companies House
            
        Returns:
            Formatted text report
        """
        if not portfolio_analysis:
            return "Error: No portfolio analysis data provided"
        
        company_number = portfolio_analysis.get("company_number", "Unknown")
        files_successful = portfolio_analysis.get("files_successful", 0)
        total_sections = portfolio_analysis.get("total_content_sections", 0)
        
        # Get company name
        if company_profile:
            company_name = company_profile.get("company_name", "Unknown Company")
        else:
            company_name = portfolio_analysis.get("company_name", "Unknown Company")
        
        # Format analysis date
        analysis_timestamp = portfolio_analysis.get('analysis_timestamp')
        formatted_date = self._format_timestamp(analysis_timestamp)
        
        report = f"""EXSIGNA STRATEGY & GOVERNANCE ARCHETYPE REPORT
=============================================
Company Name: {company_name}
Company Number: {company_number}
Analysis Date: {formatted_date}

EXECUTIVE SUMMARY
================
Files Analyzed: {portfolio_analysis.get('files_analyzed', 0)}
Successful Extractions: {files_successful}
Total Content Sections: {total_sections}

"""
        
        # Add archetype analysis results if available
        archetype_section = self._generate_archetype_section(portfolio_analysis.get("archetype_analysis"))
        report += archetype_section
        
        # Add technical details
        report += self._generate_technical_section(portfolio_analysis.get("archetype_analysis"))
        
        return report
    
    def generate_executive_summary(self, portfolio_analysis: Dict) -> str:
        """
        Generate a concise executive summary for archetype analysis
        
        Args:
            portfolio_analysis: Results from portfolio analysis
            
        Returns:
            Executive summary text
        """
        if not portfolio_analysis:
            return "Error: No portfolio analysis data provided"
        
        archetype_analysis = portfolio_analysis.get("archetype_analysis", {})
        
        if not archetype_analysis.get("success", False):
            return f"""EXECUTIVE SUMMARY
================
Analysis Status: Failed
Error: {archetype_analysis.get('error', 'Unknown error')}
Files Processed: {portfolio_analysis.get('files_successful', 0)}
"""
        
        business_archetypes = archetype_analysis.get("business_strategy_archetypes", {})
        risk_archetypes = archetype_analysis.get("risk_strategy_archetypes", {})
        
        summary = f"""EXECUTIVE SUMMARY
================
Primary Business Strategy: {business_archetypes.get('dominant', 'Not determined')}
Secondary Business Strategy: {business_archetypes.get('secondary', 'None')}
Primary Risk Strategy: {risk_archetypes.get('dominant', 'Not determined')}
Secondary Risk Strategy: {risk_archetypes.get('secondary', 'None')}
Analysis Method: {archetype_analysis.get('analysis_type', 'Unknown')}
Files Processed: {portfolio_analysis.get('files_successful', 0)}
Content Sections Analyzed: {portfolio_analysis.get('total_content_sections', 0)}

BUSINESS STRATEGY INSIGHTS:
{business_archetypes.get('reasoning', 'No insights available')}

RISK STRATEGY INSIGHTS:
{risk_archetypes.get('reasoning', 'No insights available')}
"""
        return summary
    
    def generate_json_summary(self, portfolio_analysis: Dict, 
                            company_profile: Optional[Dict] = None) -> Dict:
        """
        Generate a structured JSON summary for API/programmatic use
        
        Args:
            portfolio_analysis: Results from portfolio analysis
            company_profile: Optional company profile
            
        Returns:
            Structured summary dict
        """
        if not portfolio_analysis:
            return {"error": "No portfolio analysis data provided"}
        
        archetype_analysis = portfolio_analysis.get("archetype_analysis", {})
        
        summary = {
            "company_info": {
                "company_number": portfolio_analysis.get("company_number", "Unknown"),
                "company_name": (company_profile.get("company_name") if company_profile 
                               else archetype_analysis.get("company_name", "Unknown")),
                "analysis_date": portfolio_analysis.get("analysis_timestamp")
            },
            "analysis_results": {
                "files_analyzed": portfolio_analysis.get("files_analyzed", 0),
                "files_successful": portfolio_analysis.get("files_successful", 0),
                "total_content_sections": portfolio_analysis.get("total_content_sections", 0),
                "analysis_status": "success" if archetype_analysis.get("success", False) else "failed"
            },
            "archetype_classification": {},
            "metadata": {
                "analysis_method": archetype_analysis.get("analysis_type", "unknown"),
                "model_used": archetype_analysis.get("model_used", "unknown"),
                "timestamp": archetype_analysis.get("timestamp")
            }
        }
        
        if archetype_analysis.get("success", False):
            business_archetypes = archetype_analysis.get("business_strategy_archetypes", {})
            risk_archetypes = archetype_analysis.get("risk_strategy_archetypes", {})
            
            summary["archetype_classification"] = {
                "business_strategy": {
                    "dominant": business_archetypes.get("dominant"),
                    "secondary": business_archetypes.get("secondary"),
                    "reasoning": business_archetypes.get("reasoning")
                },
                "risk_strategy": {
                    "dominant": risk_archetypes.get("dominant"),
                    "secondary": risk_archetypes.get("secondary"),
                    "reasoning": risk_archetypes.get("reasoning")
                }
            }
        else:
            summary["error"] = archetype_analysis.get("error")
        
        return summary
    
    def generate_detailed_breakdown(self, portfolio_analysis: Dict) -> str:
        """
        Generate detailed breakdown of file-by-file analysis
        
        Args:
            portfolio_analysis: Results from portfolio analysis
            
        Returns:
            Detailed breakdown text
        """
        if not portfolio_analysis:
            return "Error: No portfolio analysis data provided"
        
        report = """DETAILED FILE ANALYSIS
=====================

"""
        
        file_analyses = portfolio_analysis.get("file_analyses", [])
        
        for i, analysis in enumerate(file_analyses, 1):
            filename = analysis.get("filename", f"File {i}")
            status = analysis.get("extraction_status", "unknown")
            
            report += f"""File {i}: {filename}
{'='*50}
Extraction Status: {status}
"""
            
            if status == "success":
                content_summary = analysis.get("content_summary", {})
                report += f"""Content Found:
  Strategy: {'✓' if content_summary.get('strategy_found') else '✗'}
  Governance: {'✓' if content_summary.get('governance_found') else '✗'}
  Risk: {'✓' if content_summary.get('risk_found') else '✗'}
  Audit: {'✓' if content_summary.get('audit_found') else '✗'}
  Total Sections: {content_summary.get('total_content_sections', 0)}

Extraction Method: {analysis.get('extraction_method', 'Unknown')}
"""
                
                debug_info = analysis.get("debug_info", {})
                if debug_info:
                    report += f"""Technical Details:
  Pages: {debug_info.get('total_pages', 'Unknown')}
  Text Length: {debug_info.get('text_length', 0):,} characters
  Tables Found: {debug_info.get('tables_found', 0)}
"""
            else:
                error = analysis.get("error", "Unknown error")
                report += f"Error: {error}\n"
            
            report += "\n"
        
        return report
    
    def _generate_archetype_section(self, archetype_analysis: Optional[Dict]) -> str:
        """Generate the archetype analysis section of the report"""
        if not archetype_analysis:
            return """ARCHETYPE CLASSIFICATION ANALYSIS
=================================
No archetype analysis available - insufficient content for classification.

"""
        
        if not archetype_analysis.get("success", False):
            return f"""ARCHETYPE CLASSIFICATION ANALYSIS
=================================
Error in analysis: {archetype_analysis.get('error', 'Unknown error')}

"""
        
        business_archetypes = archetype_analysis.get("business_strategy_archetypes", {})
        risk_archetypes = archetype_analysis.get("risk_strategy_archetypes", {})
        
        section = f"""ARCHETYPE CLASSIFICATION ANALYSIS
=================================

BUSINESS STRATEGY ARCHETYPES
----------------------------
Dominant Archetype: {business_archetypes.get('dominant', 'Unknown')}
Secondary Archetype: {business_archetypes.get('secondary', 'None')}

Business Strategy Analysis:
{business_archetypes.get('reasoning', 'No analysis provided')}

RISK STRATEGY ARCHETYPES
------------------------
Dominant Archetype: {risk_archetypes.get('dominant', 'Unknown')}
Secondary Archetype: {risk_archetypes.get('secondary', 'None')}

Risk Strategy Analysis:
{risk_archetypes.get('reasoning', 'No analysis provided')}

STRATEGIC PROFILE SUMMARY
------------------------
This company demonstrates characteristics primarily aligned with:
• Business Strategy: {business_archetypes.get('dominant', 'Unknown')}
• Risk Strategy: {risk_archetypes.get('dominant', 'Unknown')}

This archetype combination suggests a strategic approach that balances growth 
objectives with risk management priorities, consistent with UK financial 
services regulatory expectations.

"""
        
        return section
    
    def _generate_technical_section(self, archetype_analysis: Optional[Dict] = None) -> str:
        """Generate the technical details section"""
        base_section = """TECHNICAL DETAILS
================
Analysis generated using: Strategy & Governance Archetype Classification Tool
PDF extraction methods: pdfplumber, PyPDF2, OCR (as needed)
"""
        
        if archetype_analysis:
            analysis_type = archetype_analysis.get("analysis_type", "unknown")
            model_used = archetype_analysis.get("model_used", "unknown")
            
            if analysis_type == "ai_archetype_classification":
                base_section += f"AI archetype classification: {model_used}\n"
            elif analysis_type == "pattern_archetype_classification":
                base_section += "Pattern-based archetype classification: Keyword frequency analysis\n"
            else:
                base_section += f"Archetype classification method: {analysis_type}\n"
        else:
            base_section += "Archetype classification: Pattern-based fallback method\n"
        
        return base_section
    
    def _format_timestamp(self, timestamp: Optional[str]) -> str:
        """Format timestamp for display"""
        if not timestamp:
            return datetime.now().strftime("%d/%m/%Y")
        
        try:
            # Handle different timestamp formats
            if isinstance(timestamp, str):
                if 'T' in timestamp:
                    # ISO format
                    analysis_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00').split('+')[0])
                else:
                    # YYYYMMDD_HHMMSS format
                    analysis_date = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            else:
                analysis_date = timestamp
            
            return analysis_date.strftime("%d/%m/%Y")
        except Exception:
            return datetime.now().strftime("%d/%m/%Y")
    
    def generate_comparison_report(self, analyses: List[Dict]) -> str:
        """
        Generate a comparison report for multiple companies' archetype classifications
        
        Args:
            analyses: List of analysis results for different companies
            
        Returns:
            Comparison report text
        """
        if not analyses:
            return "No analyses provided for comparison"
        
        report = f"""COMPARATIVE ARCHETYPE ANALYSIS REPORT
====================================
Companies Analyzed: {len(analyses)}
Generated: {datetime.now().strftime("%d/%m/%Y %H:%M")}

"""
        
        # Summary table
        report += "ARCHETYPE OVERVIEW\n"
        report += "==================\n"
        report += f"{'Company':<15} {'Business Archetype':<25} {'Risk Archetype':<25} {'Files':<6}\n"
        report += "-" * 80 + "\n"
        
        for analysis in analyses:
            company_num = analysis.get("company_number", "Unknown")[:12]
            archetype_analysis = analysis.get("archetype_analysis", {})
            
            if archetype_analysis.get("success", False):
                business = archetype_analysis.get("business_strategy_archetypes", {}).get("dominant", "Unknown")[:23]
                risk = archetype_analysis.get("risk_strategy_archetypes", {}).get("dominant", "Unknown")[:23]
            else:
                business = "Failed"
                risk = "Failed"
            
            files = analysis.get("files_successful", 0)
            
            report += f"{company_num:<15} {business:<25} {risk:<25} {files:<6}\n"
        
        report += "\n"
        
        # Detailed breakdown for each company
        for i, analysis in enumerate(analyses, 1):
            company_name = analysis.get("firm_name", f"Company {i}")
            company_number = analysis.get("company_number", "Unknown")
            
            report += f"COMPANY {i}: {company_name} ({company_number})\n"
            report += "=" * 60 + "\n"
            
            archetype_analysis = analysis.get("archetype_analysis", {})
            if archetype_analysis.get("success", False):
                business_archetypes = archetype_analysis.get("business_strategy_archetypes", {})
                risk_archetypes = archetype_analysis.get("risk_strategy_archetypes", {})
                
                report += f"Business Strategy: {business_archetypes.get('dominant', 'Unknown')}\n"
                if business_archetypes.get('secondary'):
                    report += f"  Secondary: {business_archetypes.get('secondary')}\n"
                
                report += f"Risk Strategy: {risk_archetypes.get('dominant', 'Unknown')}\n"
                if risk_archetypes.get('secondary'):
                    report += f"  Secondary: {risk_archetypes.get('secondary')}\n"
                
                # Add brief reasoning
                business_reasoning = business_archetypes.get('reasoning', 'No reasoning provided')
                if len(business_reasoning) > 150:
                    business_reasoning = business_reasoning[:150] + "..."
                report += f"Business Rationale: {business_reasoning}\n"
                
                risk_reasoning = risk_archetypes.get('reasoning', 'No reasoning provided')
                if len(risk_reasoning) > 150:
                    risk_reasoning = risk_reasoning[:150] + "..."
                report += f"Risk Rationale: {risk_reasoning}\n"
                
            else:
                report += f"Analysis Error: {archetype_analysis.get('error', 'Unknown error')}\n"
            
            report += "\n"
        
        return report
    
    def generate_archetype_matrix(self, analyses: List[Dict]) -> str:
        """
        Generate an archetype classification matrix showing patterns across companies
        
        Args:
            analyses: List of analysis results for different companies
            
        Returns:
            Matrix report text
        """
        if not analyses:
            return "No analyses provided for matrix generation"
        
        # Collect archetype data
        business_archetypes = {}
        risk_archetypes = {}
        
        for analysis in analyses:
            archetype_analysis = analysis.get("archetype_analysis", {})
            if archetype_analysis.get("success", False):
                business = archetype_analysis.get("business_strategy_archetypes", {}).get("dominant")
                risk = archetype_analysis.get("risk_strategy_archetypes", {}).get("dominant")
                
                if business:
                    business_archetypes[business] = business_archetypes.get(business, 0) + 1
                if risk:
                    risk_archetypes[risk] = risk_archetypes.get(risk, 0) + 1
        
        report = f"""ARCHETYPE CLASSIFICATION MATRIX
==============================
Total Companies Analyzed: {len(analyses)}
Successful Classifications: {sum(1 for a in analyses if a.get('archetype_analysis', {}).get('success', False))}

BUSINESS STRATEGY ARCHETYPE DISTRIBUTION
---------------------------------------
"""
        
        for archetype, count in sorted(business_archetypes.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(analyses)) * 100
            report += f"{archetype:<30} {count:>3} ({percentage:>5.1f}%)\n"
        
        report += f"""
RISK STRATEGY ARCHETYPE DISTRIBUTION
-----------------------------------
"""
        
        for archetype, count in sorted(risk_archetypes.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(analyses)) * 100
            report += f"{archetype:<30} {count:>3} ({percentage:>5.1f}%)\n"
        
        return report