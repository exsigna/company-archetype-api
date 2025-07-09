#!/usr/bin/env python3
"""
FIXED: Database module for Strategic Analysis API
Handles storage and retrieval of analysis results
FIXED: Proper field mapping between AI analyzer output and database storage
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
        """Create necessary database tables"""
        try:
            with self._get_connection() as conn:
                # Create analysis_history table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        company_number TEXT NOT NULL,
                        company_name TEXT,
                        analysis_date TEXT NOT NULL,
                        years_analyzed TEXT,  -- JSON array
                        files_processed INTEGER DEFAULT 0,
                        business_strategy TEXT,
                        risk_strategy TEXT,
                        business_strategy_dominant TEXT,
                        business_strategy_secondary TEXT,
                        business_strategy_reasoning TEXT,
                        risk_strategy_dominant TEXT,
                        risk_strategy_secondary TEXT,
                        risk_strategy_reasoning TEXT,
                        analysis_type TEXT DEFAULT 'unknown',
                        confidence_level TEXT DEFAULT 'medium',
                        status TEXT DEFAULT 'completed',
                        raw_response TEXT,  -- Full JSON response
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create analysis_results table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        company_number TEXT NOT NULL,
                        company_name TEXT,
                        analysis_date TEXT NOT NULL,
                        years_analyzed TEXT,  -- JSON array
                        files_processed INTEGER DEFAULT 0,
                        business_strategy TEXT,
                        risk_strategy TEXT,
                        business_strategy_dominant TEXT,
                        business_strategy_secondary TEXT,
                        business_strategy_reasoning TEXT,
                        risk_strategy_dominant TEXT,
                        risk_strategy_secondary TEXT,
                        risk_strategy_reasoning TEXT,
                        analysis_type TEXT DEFAULT 'unknown',
                        confidence_level TEXT DEFAULT 'medium',
                        status TEXT DEFAULT 'completed',
                        raw_response TEXT,  -- Full JSON response
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
                logger.info("Database tables created successfully")
                
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
        FIXED: Extract business strategy information from analysis data
        
        Returns:
            tuple: (dominant, secondary, reasoning)
        """
        # NEW: Try structured report format first
        if 'business_strategy' in analysis_data:
            strategy = analysis_data['business_strategy']
            if isinstance(strategy, dict):
                # Structured report format - FIXED FIELD MAPPING
                dominant = strategy.get('dominant')
                secondary = strategy.get('secondary') 
                # FIXED: Try all possible reasoning field names
                reasoning = (strategy.get('dominant_reasoning') or  # AI analyzer output
                           strategy.get('dominant_rationale') or   # Alternative AI output
                           strategy.get('reasoning') or            # Fallback
                           strategy.get('rationale'))              # Additional fallback
                
                if dominant:  # If we found the new format, return it
                    logger.info(f"🔍 Found business strategy in structured format: {dominant}")
                    logger.info(f"🔍 Business reasoning field used: {reasoning[:50] if reasoning else 'None'}...")
                    return dominant, secondary, reasoning
                
                # Fallback to old format
                dominant = strategy.get('dominant_archetype') or strategy.get('dominant')
                secondary = strategy.get('secondary_archetype') or strategy.get('secondary')
                reasoning = strategy.get('strategic_rationale') or strategy.get('reasoning')
                return dominant, secondary, reasoning
            elif isinstance(strategy, str):
                return strategy, None, None
        
        # Legacy format support
        elif 'business_strategy_analysis' in analysis_data:
            strategy = analysis_data['business_strategy_analysis']
            dominant = strategy.get('dominant_archetype') or strategy.get('dominant')
            secondary = strategy.get('secondary_archetype') or strategy.get('secondary')
            reasoning = strategy.get('strategic_rationale') or strategy.get('reasoning')
            return dominant, secondary, reasoning
        
        logger.warning("❌ No business strategy found in analysis data")
        return None, None, None
    
    def extract_risk_strategy(self, analysis_data: Dict[str, Any]) -> tuple:
        """
        FIXED: Extract risk strategy information from analysis data
        
        Returns:
            tuple: (dominant, secondary, reasoning)
        """
        # NEW: Try structured report format first
        if 'risk_strategy' in analysis_data:
            strategy = analysis_data['risk_strategy']
            if isinstance(strategy, dict):
                # Structured report format - FIXED FIELD MAPPING
                dominant = strategy.get('dominant')
                secondary = strategy.get('secondary')
                # FIXED: Try all possible reasoning field names
                reasoning = (strategy.get('dominant_reasoning') or  # AI analyzer output
                           strategy.get('dominant_rationale') or   # Alternative AI output
                           strategy.get('reasoning') or            # Fallback
                           strategy.get('rationale'))              # Additional fallback
                
                if dominant:  # If we found the new format, return it
                    logger.info(f"🔍 Found risk strategy in structured format: {dominant}")
                    logger.info(f"🔍 Risk reasoning field used: {reasoning[:50] if reasoning else 'None'}...")
                    return dominant, secondary, reasoning
                
                # Fallback to old format
                dominant = strategy.get('dominant_archetype') or strategy.get('dominant')
                secondary = strategy.get('secondary_archetype') or strategy.get('secondary')
                reasoning = strategy.get('risk_rationale') or strategy.get('reasoning')
                return dominant, secondary, reasoning
            elif isinstance(strategy, str):
                return strategy, None, None
        
        # Legacy format support
        elif 'risk_strategy_analysis' in analysis_data:
            strategy = analysis_data['risk_strategy_analysis']
            dominant = strategy.get('dominant_archetype') or strategy.get('dominant')
            secondary = strategy.get('secondary_archetype') or strategy.get('secondary')
            reasoning = strategy.get('risk_rationale') or strategy.get('reasoning')
            return dominant, secondary, reasoning
        
        logger.warning("❌ No risk strategy found in analysis data")
        return None, None, None
    
    def store_analysis_result(self, analysis_data: Dict[str, Any]) -> int:
        """
        FIXED: Store analysis result in database with proper field extraction
        
        Args:
            analysis_data: Analysis result dictionary
            
        Returns:
            int: ID of stored record
        """
        with self.lock:
            try:
                # DEBUG: Log the complete analysis data structure
                logger.info(f"🔍 FULL ANALYSIS DATA STRUCTURE:")
                logger.info(f"   Top-level keys: {list(analysis_data.keys())}")
                
                # Extract basic data
                company_number = analysis_data.get('company_number', '')
                company_name = analysis_data.get('company_name', '')
                analysis_date = analysis_data.get('analysis_date', datetime.now().isoformat())
                years_analyzed = json.dumps(analysis_data.get('years_analyzed', []))
                files_processed = analysis_data.get('files_processed', 0)
                
                # FIXED: Extract business strategy using new method
                business_dominant, business_secondary, business_reasoning = self.extract_business_strategy(analysis_data)
                
                # FIXED: Extract risk strategy using new method
                risk_dominant, risk_secondary, risk_reasoning = self.extract_risk_strategy(analysis_data)
                
                # VALIDATION: Ensure we have meaningful data
                if not business_dominant:
                    business_dominant = "Analysis Pending"
                    business_reasoning = "Business strategy analysis requires additional processing"
                
                if not risk_dominant:
                    risk_dominant = "Analysis Pending"
                    risk_reasoning = "Risk strategy analysis requires additional processing"
                
                # Log what we extracted for debugging
                logger.info(f"💾 Storing analysis for {company_number}:")
                logger.info(f"   Business Strategy: {business_dominant}")
                logger.info(f"   Risk Strategy: {risk_dominant}")
                logger.info(f"   Business Reasoning: {business_reasoning[:100] if business_reasoning else 'None'}...")
                logger.info(f"   Risk Reasoning: {risk_reasoning[:100] if risk_reasoning else 'None'}...")
                
                analysis_type = analysis_data.get('analysis_type', 'strategic_archetype')
                confidence_level = analysis_data.get('confidence_level', 'medium')
                status = analysis_data.get('status', 'completed')
                raw_response = json.dumps(analysis_data)
                
                with self._get_connection() as conn:
                    # Store in analysis_results table (primary table) with FIXED SQL
                    cursor = conn.execute('''
                        INSERT INTO analysis_results (
                            company_number, company_name, analysis_date, years_analyzed,
                            files_processed, business_strategy, risk_strategy,
                            business_strategy_dominant, business_strategy_secondary,
                            business_strategy_reasoning, risk_strategy_dominant, risk_strategy_secondary,
                            risk_strategy_reasoning, analysis_type, confidence_level, status, raw_response
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        company_number, 
                        company_name, 
                        analysis_date, 
                        years_analyzed,
                        files_processed, 
                        json.dumps(analysis_data.get('business_strategy', {})),  # Full business strategy object
                        json.dumps(analysis_data.get('risk_strategy', {})),      # Full risk strategy object
                        business_dominant,    # FIXED: Proper business dominant
                        business_secondary,   # FIXED: Proper business secondary
                        business_reasoning,   # FIXED: Proper business reasoning
                        risk_dominant,        # FIXED: Proper risk dominant
                        risk_secondary,       # FIXED: Proper risk secondary
                        risk_reasoning,       # FIXED: Proper risk reasoning
                        analysis_type, 
                        confidence_level, 
                        status, 
                        raw_response
                    ))
                    
                    record_id = cursor.lastrowid
                    conn.commit()
                    
                    logger.info(f"✅ Analysis stored successfully with ID: {record_id}")
                    logger.info(f"   Saved business_strategy_dominant: {business_dominant}")
                    logger.info(f"   Saved risk_strategy_dominant: {risk_dominant}")
                    logger.info(f"   Saved business_strategy_reasoning: {business_reasoning[:50] if business_reasoning else 'None'}...")
                    logger.info(f"   Saved risk_strategy_reasoning: {risk_reasoning[:50] if risk_reasoning else 'None'}...")
                    return record_id
                    
            except Exception as e:
                logger.error(f"❌ Error storing analysis result: {e}")
                logger.error(f"   Analysis data keys: {list(analysis_data.keys())}")
                raise
    
    def update_existing_null_records(self) -> int:
        """
        FIXED: Update existing records that have null business_strategy_dominant and risk_strategy_dominant
        
        Returns:
            int: Number of records updated
        """
        with self.lock:
            try:
                with self._get_connection() as conn:
                    # Get records with null summary fields
                    rows = conn.execute('''
                        SELECT id, raw_response, company_number 
                        FROM analysis_results 
                        WHERE business_strategy_dominant IS NULL 
                           OR risk_strategy_dominant IS NULL
                           OR business_strategy_reasoning IS NULL
                           OR risk_strategy_reasoning IS NULL
                    ''').fetchall()
                    
                    updated_count = 0
                    
                    for row in rows:
                        record_id, raw_response, company_number = row
                        
                        try:
                            # Parse raw_response
                            if isinstance(raw_response, str):
                                analysis_data = json.loads(raw_response)
                            else:
                                analysis_data = raw_response
                            
                            # FIXED: Extract strategies using improved methods
                            business_dominant, business_secondary, business_reasoning = self.extract_business_strategy(analysis_data)
                            risk_dominant, risk_secondary, risk_reasoning = self.extract_risk_strategy(analysis_data)
                            
                            logger.info(f"🔧 Updating record {record_id} for company {company_number}:")
                            logger.info(f"   Business: {business_dominant}")
                            logger.info(f"   Risk: {risk_dominant}")
                            logger.info(f"   Business Reasoning: {business_reasoning[:50] if business_reasoning else 'None'}...")
                            logger.info(f"   Risk Reasoning: {risk_reasoning[:50] if risk_reasoning else 'None'}...")
                            
                            # Update the record
                            conn.execute('''
                                UPDATE analysis_results 
                                SET business_strategy_dominant = ?, 
                                    business_strategy_secondary = ?,
                                    business_strategy_reasoning = ?,
                                    risk_strategy_dominant = ?,
                                    risk_strategy_secondary = ?,
                                    risk_strategy_reasoning = ?
                                WHERE id = ?
                            ''', (
                                business_dominant, business_secondary, business_reasoning,
                                risk_dominant, risk_secondary, risk_reasoning,
                                record_id
                            ))
                            
                            updated_count += 1
                            
                        except Exception as e:
                            logger.error(f"❌ Error updating record {record_id}: {e}")
                    
                    conn.commit()
                    logger.info(f"✅ Successfully updated {updated_count} records")
                    return updated_count
                    
            except Exception as e:
                logger.error(f"❌ Error updating existing records: {e}")
                return 0
    
    # ... [Rest of the methods remain the same as they're working correctly] ...
    
    def get_analysis_by_company(self, company_number: str) -> List[Dict[str, Any]]:
        """Get all analyses for a company - CHECKS MULTIPLE TABLES"""
        try:
            logger.info(f"🔍 DEBUG: Searching for company_number: '{company_number}'")
            
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
                            logger.info(f"🔍 DEBUG: Table '{table_name}' exists with {total_count} total records")
                            
                            if total_count > 0:
                                # Try exact match
                                exact_match = conn.execute(f'SELECT COUNT(*) FROM {table_name} WHERE company_number = ?', (company_number,)).fetchone()[0]
                                logger.info(f"🔍 DEBUG: Exact matches in '{table_name}' for '{company_number}': {exact_match}")
                                
                                if exact_match > 0:
                                    rows = conn.execute(f'''
                                        SELECT * FROM {table_name} 
                                        WHERE company_number = ? 
                                        ORDER BY analysis_date DESC
                                    ''', (company_number,)).fetchall()
                                    
                                    result = [dict(row) for row in rows]
                                    logger.info(f"🔍 DEBUG: Returning {len(result)} results from '{table_name}'")
                                    return result
                                
                                # Try without leading zero
                                company_number_no_zero = company_number.lstrip('0')
                                if company_number_no_zero != company_number:
                                    no_zero_match = conn.execute(f'SELECT COUNT(*) FROM {table_name} WHERE company_number = ?', (company_number_no_zero,)).fetchone()[0]
                                    logger.info(f"🔍 DEBUG: Matches in '{table_name}' for '{company_number_no_zero}' (no leading zero): {no_zero_match}")
                                    
                                    if no_zero_match > 0:
                                        rows = conn.execute(f'''
                                            SELECT * FROM {table_name} 
                                            WHERE company_number = ? 
                                            ORDER BY analysis_date DESC
                                        ''', (company_number_no_zero,)).fetchall()
                                        
                                        result = [dict(row) for row in rows]
                                        logger.info(f"🔍 DEBUG: Returning {len(result)} results from '{table_name}' (no leading zero)")
                                        return result
                        else:
                            logger.info(f"🔍 DEBUG: Table '{table_name}' does not exist")
                    
                    except Exception as table_error:
                        logger.warning(f"🔍 DEBUG: Error checking table '{table_name}': {table_error}")
                
                logger.info(f"🔍 DEBUG: No matches found in any table for company number '{company_number}'")
                return []
                
        except Exception as e:
            logger.error(f"Error getting analyses for company {company_number}: {e}")
            return []
    
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
                                logger.info(f"🔍 DEBUG: Returning {len(rows)} recent analyses from '{table_name}'")
                                return [dict(row) for row in rows]
                    except Exception as e:
                        logger.warning(f"Error querying table {table_name}: {e}")
                        continue
                
                return []
                
        except Exception as e:
            logger.error(f"Error getting recent analyses: {e}")
            return []