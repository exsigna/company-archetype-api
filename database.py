#!/usr/bin/env python3
"""
ENHANCED: Database module for Strategic Analysis API
FIXED: Complete preservation of full reasoning text from AI analyzer
ENHANCED: Improved field mapping and text extraction methods
"""

import os
import json
import logging
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
import threading

logger = logging.getLogger(__name__)

class AnalysisDatabase:
    """Database handler for analysis results using SQLite"""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection"""
        # Always use SQLite to avoid MySQL dependency issues
        self.db_path = db_path or os.environ.get('DATABASE_URL', 'analysis_database.db')
        
        # Clean up database URL if it contains SQLite prefix
        if self.db_path.startswith('sqlite:///'):
            self.db_path = self.db_path.replace('sqlite:///', '')
        elif self.db_path.startswith('mysql://') or self.db_path.startswith('postgresql://'):
            # Fallback to SQLite if other database URLs are provided but not available
            logger.warning("Non-SQLite database URL detected, falling back to SQLite")
            self.db_path = 'analysis_database.db'
        
        self.lock = threading.Lock()
        
        try:
            self._create_tables()
            self._detect_data_location()
            logger.info(f"✅ Database initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    def _get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def _detect_data_location(self):
        """Detect which table contains the analysis data"""
        try:
            with self._get_connection() as conn:
                tables_to_check = ['analysis_history', 'analysis_results', 'analyses']
                
                for table_name in tables_to_check:
                    try:
                        # Check if table exists
                        cursor = conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
                            (table_name,)
                        )
                        table_exists = cursor.fetchone()
                        
                        if table_exists:
                            # Get record count
                            count_result = conn.execute(f'SELECT COUNT(*) FROM {table_name}')
                            count = count_result.fetchone()[0]
                            logger.info(f"🔍 Table '{table_name}' exists with {count} records")
                            
                            if count > 0:
                                # Get sample data
                                sample = conn.execute(f'SELECT company_number FROM {table_name} LIMIT 3').fetchall()
                                logger.info(f"🔍 Sample company numbers in '{table_name}': {[row[0] for row in sample]}")
                        else:
                            logger.info(f"🔍 Table '{table_name}' does not exist")
                            
                    except Exception as e:
                        logger.warning(f"🔍 Error checking table '{table_name}': {e}")
                        
        except Exception as e:
            logger.error(f"Error detecting data location: {e}")
    
    def _create_tables(self):
        """Create necessary database tables with ENHANCED TEXT fields"""
        try:
            with self._get_connection() as conn:
                # Create analysis_history table with ENHANCED TEXT storage
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        company_number TEXT NOT NULL,
                        company_name TEXT,
                        analysis_date TEXT NOT NULL,
                        years_analyzed TEXT,  -- JSON array
                        files_processed INTEGER DEFAULT 0,
                        business_strategy TEXT,  -- Full JSON object
                        risk_strategy TEXT,      -- Full JSON object
                        business_strategy_dominant TEXT,
                        business_strategy_secondary TEXT,
                        business_strategy_reasoning TEXT,  -- ENHANCED: Now stores full text
                        business_strategy_definition TEXT, -- NEW: Store definitions
                        risk_strategy_dominant TEXT,
                        risk_strategy_secondary TEXT,
                        risk_strategy_reasoning TEXT,      -- ENHANCED: Now stores full text
                        risk_strategy_definition TEXT,     -- NEW: Store definitions
                        swot_analysis TEXT,                -- NEW: Store SWOT data
                        analysis_type TEXT DEFAULT 'unknown',
                        confidence_level TEXT DEFAULT 'medium',
                        status TEXT DEFAULT 'completed',
                        raw_response TEXT,  -- Full JSON response - ENHANCED storage
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create analysis_results table with ENHANCED TEXT storage
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        company_number TEXT NOT NULL,
                        company_name TEXT,
                        analysis_date TEXT NOT NULL,
                        years_analyzed TEXT,  -- JSON array
                        files_processed INTEGER DEFAULT 0,
                        business_strategy TEXT,  -- Full JSON object
                        risk_strategy TEXT,      -- Full JSON object
                        business_strategy_dominant TEXT,
                        business_strategy_secondary TEXT,
                        business_strategy_reasoning TEXT,  -- ENHANCED: Now stores full text
                        business_strategy_definition TEXT, -- NEW: Store definitions
                        risk_strategy_dominant TEXT,
                        risk_strategy_secondary TEXT,
                        risk_strategy_reasoning TEXT,      -- ENHANCED: Now stores full text
                        risk_strategy_definition TEXT,     -- NEW: Store definitions
                        swot_analysis TEXT,                -- NEW: Store SWOT data
                        analysis_type TEXT DEFAULT 'unknown',
                        confidence_level TEXT DEFAULT 'medium',
                        status TEXT DEFAULT 'completed',
                        raw_response TEXT,  -- Full JSON response - ENHANCED storage
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better performance
                for table in ['analysis_history', 'analysis_results']:
                    conn.execute(f'CREATE INDEX IF NOT EXISTS idx_company_number_{table} ON {table}(company_number)')
                    conn.execute(f'CREATE INDEX IF NOT EXISTS idx_analysis_date_{table} ON {table}(analysis_date)')
                    conn.execute(f'CREATE INDEX IF NOT EXISTS idx_company_name_{table} ON {table}(company_name)')
                
                conn.commit()
                logger.info("Database tables created successfully with enhanced text storage")
                
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self._get_connection() as conn:
                conn.execute('SELECT 1')
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def extract_business_strategy(self, analysis_data: Dict[str, Any]) -> tuple:
        """
        ENHANCED: Extract business strategy information with FULL TEXT preservation
        
        Returns:
            tuple: (dominant, secondary, reasoning, definition)
        """
        logger.info(f"🔍 EXTRACTING BUSINESS STRATEGY FROM: {list(analysis_data.keys())}")
        
        # Initialize with defaults
        dominant = 'Disciplined Specialist Growth'
        secondary = 'Service-Driven Differentiator'
        reasoning = ''
        definition = ''
        
        # NEW: Try structured report format first (from AI analyzer)
        if 'business_strategy' in analysis_data:
            strategy = analysis_data['business_strategy']
            logger.info(f"🔍 Found business_strategy field, type: {type(strategy)}")
            
            if isinstance(strategy, dict):
                # Extract all available fields
                dominant = strategy.get('dominant', '').strip() or dominant
                secondary = strategy.get('secondary', '').strip() or secondary
                
                # ENHANCED: Extract FULL reasoning with all possible field names
                reasoning_sources = [
                    strategy.get('dominant_reasoning'),    # Primary AI output
                    strategy.get('dominant_rationale'),    # Alternative AI output
                    strategy.get('reasoning'),             # Fallback
                    strategy.get('rationale'),             # Additional fallback
                    strategy.get('analysis'),              # Detailed analysis
                    strategy.get('evidence')               # Evidence field
                ]
                
                # Get the longest/most comprehensive reasoning
                reasoning_candidates = [r for r in reasoning_sources if r and str(r).strip()]
                if reasoning_candidates:
                    reasoning = max(reasoning_candidates, key=len)
                    logger.info(f"🔍 Selected business reasoning from {len(reasoning_candidates)} candidates, length: {len(reasoning)}")
                
                # Extract definition
                definition = strategy.get('dominant_definition', '').strip()
                
                logger.info(f"✅ Business strategy extracted - Dominant: {dominant}, Length: {len(reasoning)}")
                
            elif isinstance(strategy, str):
                dominant = strategy
                logger.info(f"✅ Business strategy as string: {dominant}")
        
        # Legacy format support with ENHANCED extraction
        elif 'business_strategy_analysis' in analysis_data:
            strategy = analysis_data['business_strategy_analysis']
            dominant = strategy.get('dominant_archetype') or strategy.get('dominant') or dominant
            secondary = strategy.get('secondary_archetype') or strategy.get('secondary') or secondary
            
            # Extract reasoning from multiple possible sources
            reasoning_sources = [
                strategy.get('strategic_rationale'),
                strategy.get('reasoning'),
                strategy.get('analysis'),
                strategy.get('rationale'),
                strategy.get('evidence')
            ]
            reasoning_candidates = [r for r in reasoning_sources if r and str(r).strip()]
            if reasoning_candidates:
                reasoning = max(reasoning_candidates, key=len)
                
            definition = strategy.get('definition', '').strip()
            logger.info(f"✅ Business strategy from legacy format - Length: {len(reasoning)}")
        
        # Check direct fields from database/API response
        else:
            # Try direct database fields
            if analysis_data.get('business_strategy_dominant'):
                dominant = analysis_data['business_strategy_dominant']
            if analysis_data.get('business_strategy_secondary'):
                secondary = analysis_data['business_strategy_secondary']
            if analysis_data.get('business_strategy_reasoning'):
                reasoning = analysis_data['business_strategy_reasoning']
            if analysis_data.get('business_strategy_definition'):
                definition = analysis_data['business_strategy_definition']
                
            logger.info(f"✅ Business strategy from direct fields - Length: {len(reasoning)}")
        
        # ENHANCED: Final validation and logging
        if not reasoning or len(reasoning.strip()) < 50:
            logger.warning(f"❌ Business reasoning is too short ({len(reasoning)} chars), keeping what we have")
        
        logger.info(f"🔍 FINAL BUSINESS EXTRACTION:")
        logger.info(f"   Dominant: {dominant}")
        logger.info(f"   Secondary: {secondary}")
        logger.info(f"   Reasoning length: {len(reasoning)} characters")
        logger.info(f"   Definition length: {len(definition)} characters")
        logger.info(f"   Reasoning preview: {reasoning[:100]}..." if reasoning else "   No reasoning")
        
        return dominant, secondary, reasoning, definition
    
    def extract_risk_strategy(self, analysis_data: Dict[str, Any]) -> tuple:
        """
        ENHANCED: Extract risk strategy information with FULL TEXT preservation
        
        Returns:
            tuple: (dominant, secondary, reasoning, definition)
        """
        logger.info(f"🔍 EXTRACTING RISK STRATEGY FROM: {list(analysis_data.keys())}")
        
        # Initialize with defaults
        dominant = 'Risk-First Conservative'
        secondary = 'Rules-Led Operator'
        reasoning = ''
        definition = ''
        
        # NEW: Try structured report format first (from AI analyzer)
        if 'risk_strategy' in analysis_data:
            strategy = analysis_data['risk_strategy']
            logger.info(f"🔍 Found risk_strategy field, type: {type(strategy)}")
            
            if isinstance(strategy, dict):
                # Extract all available fields
                dominant = strategy.get('dominant', '').strip() or dominant
                secondary = strategy.get('secondary', '').strip() or secondary
                
                # ENHANCED: Extract FULL reasoning with all possible field names
                reasoning_sources = [
                    strategy.get('dominant_reasoning'),    # Primary AI output
                    strategy.get('dominant_rationale'),    # Alternative AI output
                    strategy.get('reasoning'),             # Fallback
                    strategy.get('rationale'),             # Additional fallback
                    strategy.get('analysis'),              # Detailed analysis
                    strategy.get('evidence')               # Evidence field
                ]
                
                # Get the longest/most comprehensive reasoning
                reasoning_candidates = [r for r in reasoning_sources if r and str(r).strip()]
                if reasoning_candidates:
                    reasoning = max(reasoning_candidates, key=len)
                    logger.info(f"🔍 Selected risk reasoning from {len(reasoning_candidates)} candidates, length: {len(reasoning)}")
                
                # Extract definition
                definition = strategy.get('dominant_definition', '').strip()
                
                logger.info(f"✅ Risk strategy extracted - Dominant: {dominant}, Length: {len(reasoning)}")
                
            elif isinstance(strategy, str):
                dominant = strategy
                logger.info(f"✅ Risk strategy as string: {dominant}")
        
        # Legacy format support with ENHANCED extraction
        elif 'risk_strategy_analysis' in analysis_data:
            strategy = analysis_data['risk_strategy_analysis']
            dominant = strategy.get('dominant_archetype') or strategy.get('dominant') or dominant
            secondary = strategy.get('secondary_archetype') or strategy.get('secondary') or secondary
            
            # Extract reasoning from multiple possible sources
            reasoning_sources = [
                strategy.get('risk_rationale'),
                strategy.get('reasoning'),
                strategy.get('analysis'),
                strategy.get('rationale'),
                strategy.get('evidence')
            ]
            reasoning_candidates = [r for r in reasoning_sources if r and str(r).strip()]
            if reasoning_candidates:
                reasoning = max(reasoning_candidates, key=len)
                
            definition = strategy.get('definition', '').strip()
            logger.info(f"✅ Risk strategy from legacy format - Length: {len(reasoning)}")
        
        # Check direct fields from database/API response
        else:
            # Try direct database fields
            if analysis_data.get('risk_strategy_dominant'):
                dominant = analysis_data['risk_strategy_dominant']
            if analysis_data.get('risk_strategy_secondary'):
                secondary = analysis_data['risk_strategy_secondary']
            if analysis_data.get('risk_strategy_reasoning'):
                reasoning = analysis_data['risk_strategy_reasoning']
            if analysis_data.get('risk_strategy_definition'):
                definition = analysis_data['risk_strategy_definition']
                
            logger.info(f"✅ Risk strategy from direct fields - Length: {len(reasoning)}")
        
        # ENHANCED: Final validation and logging
        if not reasoning or len(reasoning.strip()) < 50:
            logger.warning(f"❌ Risk reasoning is too short ({len(reasoning)} chars), keeping what we have")
        
        logger.info(f"🔍 FINAL RISK EXTRACTION:")
        logger.info(f"   Dominant: {dominant}")
        logger.info(f"   Secondary: {secondary}")
        logger.info(f"   Reasoning length: {len(reasoning)} characters")
        logger.info(f"   Definition length: {len(definition)} characters")
        logger.info(f"   Reasoning preview: {reasoning[:100]}..." if reasoning else "   No reasoning")
        
        return dominant, secondary, reasoning, definition
    
    def extract_swot_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """Extract SWOT analysis data as JSON string"""
        swot_sources = [
            analysis_data.get('swot_analysis'),
            analysis_data.get('swot'),
            analysis_data.get('analysis_metadata', {}).get('swot_analysis')
        ]
        
        for swot in swot_sources:
            if swot:
                if isinstance(swot, dict):
                    return json.dumps(swot)
                elif isinstance(swot, str):
                    return swot
        
        return ''
    
    def store_analysis_result(self, analysis_data: Dict[str, Any]) -> int:
        """
        ENHANCED: Store analysis result with complete text preservation
        
        Args:
            analysis_data: Analysis result dictionary
            
        Returns:
            int: ID of stored record
        """
        with self.lock:
            try:
                # DEBUG: Log the complete analysis data structure
                logger.info(f"💾 STORING ANALYSIS DATA:")
                logger.info(f"   Top-level keys: {list(analysis_data.keys())}")
                
                # Extract basic data
                company_number = analysis_data.get('company_number', '')
                company_name = analysis_data.get('company_name', '')
                analysis_date = analysis_data.get('analysis_date', datetime.now().isoformat())
                years_analyzed = json.dumps(analysis_data.get('years_analyzed', []))
                files_processed = analysis_data.get('files_processed', 0)
                
                # ENHANCED: Extract business strategy with FULL preservation
                business_dominant, business_secondary, business_reasoning, business_definition = self.extract_business_strategy(analysis_data)
                
                # ENHANCED: Extract risk strategy with FULL preservation
                risk_dominant, risk_secondary, risk_reasoning, risk_definition = self.extract_risk_strategy(analysis_data)
                
                # Extract SWOT analysis
                swot_analysis = self.extract_swot_analysis(analysis_data)
                
                # CRITICAL: Ensure we preserve the FULL reasoning text
                logger.info(f"💾 FINAL STORAGE PREPARATION:")
                logger.info(f"   Business reasoning chars: {len(business_reasoning)}")
                logger.info(f"   Risk reasoning chars: {len(risk_reasoning)}")
                logger.info(f"   Business preview: {business_reasoning[:100]}..." if business_reasoning else "   Empty")
                logger.info(f"   Risk preview: {risk_reasoning[:100]}..." if risk_reasoning else "   Empty")
                
                analysis_type = analysis_data.get('analysis_type', 'strategic_archetype')
                confidence_level = analysis_data.get('confidence_level', 'medium')
                status = analysis_data.get('status', 'completed')
                raw_response = json.dumps(analysis_data, ensure_ascii=False)  # ENHANCED: Preserve unicode
                
                with self._get_connection() as conn:
                    # ENHANCED: Store with complete text preservation
                    cursor = conn.execute('''
                        INSERT INTO analysis_results (
                            company_number, company_name, analysis_date, years_analyzed,
                            files_processed, business_strategy, risk_strategy,
                            business_strategy_dominant, business_strategy_secondary,
                            business_strategy_reasoning, business_strategy_definition,
                            risk_strategy_dominant, risk_strategy_secondary,
                            risk_strategy_reasoning, risk_strategy_definition,
                            swot_analysis, analysis_type, confidence_level, status, raw_response
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        company_number, 
                        company_name, 
                        analysis_date, 
                        years_analyzed,
                        files_processed, 
                        json.dumps(analysis_data.get('business_strategy', {}), ensure_ascii=False),  # Full business strategy object
                        json.dumps(analysis_data.get('risk_strategy', {}), ensure_ascii=False),      # Full risk strategy object
                        business_dominant,        # Dominant archetype name
                        business_secondary,       # Secondary archetype name
                        business_reasoning,       # FULL reasoning text - preserved completely
                        business_definition,      # Definition text
                        risk_dominant,            # Dominant risk archetype name
                        risk_secondary,           # Secondary risk archetype name
                        risk_reasoning,           # FULL risk reasoning text - preserved completely
                        risk_definition,          # Risk definition text
                        swot_analysis,            # SWOT analysis data
                        analysis_type, 
                        confidence_level, 
                        status, 
                        raw_response              # Complete raw response
                    ))
                    
                    record_id = cursor.lastrowid
                    conn.commit()
                    
                    # VERIFICATION: Check what was actually stored
                    verification = conn.execute('''
                        SELECT business_strategy_reasoning, risk_strategy_reasoning 
                        FROM analysis_results WHERE id = ?
                    ''', (record_id,)).fetchone()
                    
                    if verification:
                        stored_business_len = len(verification[0]) if verification[0] else 0
                        stored_risk_len = len(verification[1]) if verification[1] else 0
                        logger.info(f"✅ VERIFICATION - Stored lengths: Business={stored_business_len}, Risk={stored_risk_len}")
                    
                    logger.info(f"✅ Analysis stored successfully with ID: {record_id}")
                    logger.info(f"   Final business_strategy_dominant: {business_dominant}")
                    logger.info(f"   Final risk_strategy_dominant: {risk_dominant}")
                    logger.info(f"   Final business reasoning length: {len(business_reasoning)} chars")
                    logger.info(f"   Final risk reasoning length: {len(risk_reasoning)} chars")
                    
                    return record_id
                    
            except Exception as e:
                logger.error(f"❌ Error storing analysis result: {e}")
                logger.error(f"   Analysis data keys: {list(analysis_data.keys())}")
                raise
    
    def get_analysis_by_company(self, company_number: str) -> List[Dict[str, Any]]:
        """
        ENHANCED: Get all analyses for a company with FULL text retrieval
        """
        try:
            logger.info(f"🔍 ENHANCED: Searching for company_number: '{company_number}'")
            
            with self._get_connection() as conn:
                # Check multiple table names in order of preference
                tables_to_check = ['analysis_results', 'analysis_history', 'analyses']
                
                for table_name in tables_to_check:
                    try:
                        # Check if table exists
                        table_check = conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
                            (table_name,)
                        ).fetchone()
                        
                        if table_check:
                            # Get total count in table
                            total_count = conn.execute(f'SELECT COUNT(*) FROM {table_name}').fetchone()[0]
                            logger.info(f"🔍 Table '{table_name}' exists with {total_count} total records")
                            
                            if total_count > 0:
                                # Try exact match with ENHANCED field selection
                                exact_match = conn.execute(f'SELECT COUNT(*) FROM {table_name} WHERE company_number = ?', (company_number,)).fetchone()[0]
                                logger.info(f"🔍 Exact matches in '{table_name}' for '{company_number}': {exact_match}")
                                
                                if exact_match > 0:
                                    # ENHANCED: Get ALL fields including the new definition fields
                                    rows = conn.execute(f'''
                                        SELECT * FROM {table_name} 
                                        WHERE company_number = ? 
                                        ORDER BY analysis_date DESC
                                    ''', (company_number,)).fetchall()
                                    
                                    result = []
                                    for row in rows:
                                        row_dict = dict(row)
                                        
                                        # ENHANCED: Log the retrieved text lengths for verification
                                        business_reasoning = row_dict.get('business_strategy_reasoning', '')
                                        risk_reasoning = row_dict.get('risk_strategy_reasoning', '')
                                        logger.info(f"📄 Retrieved analysis {row_dict.get('id')}:")
                                        logger.info(f"   Business reasoning: {len(business_reasoning)} chars")
                                        logger.info(f"   Risk reasoning: {len(risk_reasoning)} chars")
                                        logger.info(f"   Business preview: {business_reasoning[:100]}..." if business_reasoning else "   Empty")
                                        logger.info(f"   Risk preview: {risk_reasoning[:100]}..." if risk_reasoning else "   Empty")
                                        
                                        result.append(row_dict)
                                    
                                    logger.info(f"🔍 ENHANCED: Returning {len(result)} results from '{table_name}' with full text")
                                    return result
                                
                                # Try without leading zero
                                company_number_no_zero = company_number.lstrip('0')
                                if company_number_no_zero != company_number:
                                    no_zero_match = conn.execute(f'SELECT COUNT(*) FROM {table_name} WHERE company_number = ?', (company_number_no_zero,)).fetchone()[0]
                                    logger.info(f"🔍 Matches in '{table_name}' for '{company_number_no_zero}' (no leading zero): {no_zero_match}")
                                    
                                    if no_zero_match > 0:
                                        rows = conn.execute(f'''
                                            SELECT * FROM {table_name} 
                                            WHERE company_number = ? 
                                            ORDER BY analysis_date DESC
                                        ''', (company_number_no_zero,)).fetchall()
                                        
                                        result = [dict(row) for row in rows]
                                        logger.info(f"🔍 ENHANCED: Returning {len(result)} results from '{table_name}' (no leading zero)")
                                        return result
                        else:
                            logger.info(f"🔍 Table '{table_name}' does not exist")
                    
                    except Exception as table_error:
                        logger.warning(f"🔍 Error checking table '{table_name}': {table_error}")
                
                logger.info(f"🔍 ENHANCED: No matches found in any table for company number '{company_number}'")
                return []
                
        except Exception as e:
            logger.error(f"Error getting analyses for company {company_number}: {e}")
            return []
    
    # ... (rest of the methods remain the same but with enhanced logging)
    
    def get_recent_analyses(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent analyses from any available table"""
        try:
            with self._get_connection() as conn:
                # Try tables in order of preference
                for table_name in ['analysis_results', 'analysis_history', 'analyses']:
                    try:
                        table_check = conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
                            (table_name,)
                        ).fetchone()
                        
                        if table_check:
                            rows = conn.execute(f'''
                                SELECT * FROM {table_name} 
                                ORDER BY created_at DESC 
                                LIMIT ?
                            ''', (limit,)).fetchall()
                            
                            if rows:
                                logger.info(f"🔍 Returning {len(rows)} recent analyses from '{table_name}'")
                                return [dict(row) for row in rows]
                    except Exception as e:
                        logger.warning(f"Error querying table {table_name}: {e}")
                        continue
                
                return []
                
        except Exception as e:
            logger.error(f"Error getting recent analyses: {e}")
            return []
    
    def search_companies(self, search_term: str) -> List[Dict[str, Any]]:
        """Search for companies by name or number"""
        try:
            with self._get_connection() as conn:
                # Try tables in order of preference
                for table_name in ['analysis_results', 'analysis_history', 'analyses']:
                    try:
                        table_check = conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
                            (table_name,)
                        ).fetchone()
                        
                        if table_check:
                            # Search by company name or number
                            rows = conn.execute(f'''
                                SELECT DISTINCT company_number, company_name, 
                                       MAX(analysis_date) as latest_analysis
                                FROM {table_name} 
                                WHERE company_name LIKE ? OR company_number LIKE ?
                                GROUP BY company_number, company_name
                                ORDER BY latest_analysis DESC
                                LIMIT 20
                            ''', (f'%{search_term}%', f'%{search_term}%')).fetchall()
                            
                            if rows:
                                logger.info(f"🔍 Found {len(rows)} companies matching '{search_term}'")
                                return [dict(row) for row in rows]
                    except Exception as e:
                        logger.warning(f"Error searching in table {table_name}: {e}")
                        continue
                
                return []
                
        except Exception as e:
            logger.error(f"Error searching companies: {e}")
            return []
    
    def delete_analysis_by_id(self, analysis_id: int, company_number: str = None) -> bool:
        """Delete a specific analysis by ID"""
        try:
            with self._get_connection() as conn:
                # Try deleting from all possible tables
                for table_name in ['analysis_results', 'analysis_history', 'analyses']:
                    try:
                        if company_number:
                            cursor = conn.execute(f'DELETE FROM {table_name} WHERE id = ? AND company_number = ?', 
                                                (analysis_id, company_number))
                        else:
                            cursor = conn.execute(f'DELETE FROM {table_name} WHERE id = ?', (analysis_id,))
                        
                        if cursor.rowcount > 0:
                            conn.commit()
                            logger.info(f"✅ Deleted analysis {analysis_id} from {table_name}")
                            return True
                    except Exception as e:
                        logger.warning(f"Could not delete from {table_name}: {e}")
                        continue
                
                logger.warning(f"❌ Analysis {analysis_id} not found in any table")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting analysis {analysis_id}: {e}")
            return False
    
    def cleanup_invalid_analyses(self, company_number: str) -> int:
        """Remove invalid analysis entries for a company"""
        try:
            deleted_count = 0
            
            with self._get_connection() as conn:
                # Get analyses to check
                analyses = self.get_analysis_by_company(company_number)
                
                for analysis in analyses:
                    should_delete = False
                    
                    # Check for invalid patterns
                    if analysis.get('company_name', '').upper().find('HSBC') != -1 and company_number == '02613335':
                        should_delete = True
                    
                    # Check for generic reasoning
                    business_reasoning = analysis.get('business_strategy_reasoning', '')
                    if business_reasoning == 'The company demonstrates strong growth-oriented strategies with focus on market expansion and innovation.':
                        should_delete = True
                    
                    if should_delete:
                        if self.delete_analysis_by_id(analysis.get('id'), company_number):
                            deleted_count += 1
                
                logger.info(f"🧹 Cleaned up {deleted_count} invalid analyses for company {company_number}")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error cleaning up analyses for {company_number}: {e}")
            return 0
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            with self._get_connection() as conn:
                stats = {
                    'total_analyses': 0,
                    'unique_companies': 0,
                    'latest_analysis': None,
                    'table_counts': {},
                    'text_statistics': {}
                }
                
                for table_name in ['analysis_results', 'analysis_history', 'analyses']:
                    try:
                        table_check = conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
                            (table_name,)
                        ).fetchone()
                        
                        if table_check:
                            # Get count
                            count = conn.execute(f'SELECT COUNT(*) FROM {table_name}').fetchone()[0]
                            stats['table_counts'][table_name] = count
                            stats['total_analyses'] += count
                            
                            if count > 0:
                                # Get unique companies
                                unique = conn.execute(f'SELECT COUNT(DISTINCT company_number) FROM {table_name}').fetchone()[0]
                                stats['unique_companies'] = max(stats['unique_companies'], unique)
                                
                                # Get latest analysis
                                latest = conn.execute(f'SELECT MAX(analysis_date) FROM {table_name}').fetchone()[0]
                                if latest and (not stats['latest_analysis'] or latest > stats['latest_analysis']):
                                    stats['latest_analysis'] = latest
                                
                                # ENHANCED: Get text statistics
                                try:
                                    text_stats = conn.execute(f'''
                                        SELECT 
                                            AVG(LENGTH(business_strategy_reasoning)) as avg_business_length,
                                            AVG(LENGTH(risk_strategy_reasoning)) as avg_risk_length,
                                            COUNT(CASE WHEN LENGTH(business_strategy_reasoning) > 100 THEN 1 END) as business_with_text,
                                            COUNT(CASE WHEN LENGTH(risk_strategy_reasoning) > 100 THEN 1 END) as risk_with_text
                                        FROM {table_name}
                                        WHERE business_strategy_reasoning IS NOT NULL 
                                           OR risk_strategy_reasoning IS NOT NULL
                                    ''').fetchone()
                                    
                                    if text_stats:
                                        stats['text_statistics'][table_name] = {
                                            'avg_business_reasoning_length': round(text_stats[0] or 0),
                                            'avg_risk_reasoning_length': round(text_stats[1] or 0),
                                            'analyses_with_business_text': text_stats[2] or 0,
                                            'analyses_with_risk_text': text_stats[3] or 0
                                        }
                                except Exception as text_error:
                                    logger.warning(f"Could not get text stats from {table_name}: {text_error}")
                    
                    except Exception as e:
                        logger.warning(f"Error getting stats from {table_name}: {e}")
                        stats['table_counts'][table_name] = 0
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'error': str(e)}