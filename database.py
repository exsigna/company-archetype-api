#!/usr/bin/env python3
"""
Database module for Strategic Analysis API
Handles storage and retrieval of analysis results
ENHANCED: Checks multiple tables (analysis_history, analysis_results)
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
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
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
    
    def store_analysis_result(self, analysis_data: Dict[str, Any]) -> int:
        """
        Store analysis result in database
        
        Args:
            analysis_data: Analysis result dictionary
            
        Returns:
            int: ID of stored record
        """
        with self.lock:
            try:
                # Extract data from analysis_data
                company_number = analysis_data.get('company_number', '')
                company_name = analysis_data.get('company_name', '')
                analysis_date = analysis_data.get('analysis_date', datetime.now().isoformat())
                years_analyzed = json.dumps(analysis_data.get('years_analyzed', []))
                files_processed = analysis_data.get('files_processed', 0)
                
                # Extract business strategy
                business_strategy = analysis_data.get('business_strategy', {})
                business_strategy_dominant = business_strategy.get('dominant', '')
                business_strategy_secondary = business_strategy.get('secondary', '')
                business_strategy_reasoning = business_strategy.get('reasoning', '')
                
                # Extract risk strategy
                risk_strategy = analysis_data.get('risk_strategy', {})
                risk_strategy_dominant = risk_strategy.get('dominant', '')
                risk_strategy_secondary = risk_strategy.get('secondary', '')
                risk_strategy_reasoning = risk_strategy.get('reasoning', '')
                
                analysis_type = analysis_data.get('analysis_type', 'unknown')
                confidence_level = analysis_data.get('confidence_level', 'medium')
                status = analysis_data.get('status', 'completed')
                raw_response = json.dumps(analysis_data)
                
                with self._get_connection() as conn:
                    # Store in analysis_results table (primary table)
                    cursor = conn.execute('''
                        INSERT INTO analysis_results (
                            company_number, company_name, analysis_date, years_analyzed,
                            files_processed, business_strategy, risk_strategy,
                            business_strategy_dominant, business_strategy_secondary,
                            business_strategy_reasoning, risk_strategy_dominant, risk_strategy_secondary,
                            risk_strategy_reasoning, analysis_type, confidence_level, status, raw_response
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        company_number, company_name, analysis_date, years_analyzed,
                        files_processed, business_strategy_dominant, risk_strategy_dominant,
                        business_strategy_dominant, business_strategy_secondary,
                        business_strategy_reasoning, risk_strategy_dominant, risk_strategy_secondary,
                        risk_strategy_reasoning, analysis_type, confidence_level, status, raw_response
                    ))
                    
                    record_id = cursor.lastrowid
                    conn.commit()
                    
                    logger.info(f"Analysis stored successfully with ID: {record_id}")
                    return record_id
                    
            except Exception as e:
                logger.error(f"Error storing analysis result: {e}")
                raise
    
    def get_analysis_by_company(self, company_number: str) -> List[Dict[str, Any]]:
        """
        Get all analyses for a company - CHECKS MULTIPLE TABLES
        
        Args:
            company_number: Company number
            
        Returns:
            List of analysis records
        """
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
                                # Get sample company numbers
                                sample_numbers = conn.execute(f'SELECT DISTINCT company_number FROM {table_name} LIMIT 5').fetchall()
                                logger.info(f"🔍 DEBUG: Sample company numbers in '{table_name}': {[row[0] for row in sample_numbers]}")
                                
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
    
    def search_companies(self, search_term: str) -> List[Dict[str, Any]]:
        """Search companies in any available table"""
        try:
            search_pattern = f"%{search_term}%"
            
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
                                SELECT DISTINCT company_number, company_name, 
                                       COUNT(*) as analysis_count,
                                       MAX(analysis_date) as latest_analysis
                                FROM {table_name} 
                                WHERE company_name LIKE ? OR company_number LIKE ?
                                GROUP BY company_number, company_name
                                ORDER BY latest_analysis DESC
                            ''', (search_pattern, search_pattern)).fetchall()
                            
                            if rows:
                                logger.info(f"🔍 DEBUG: Found {len(rows)} company matches in '{table_name}'")
                                return [dict(row) for row in rows]
                    except Exception as e:
                        logger.warning(f"Error searching in table {table_name}: {e}")
                        continue
                
                return []
                
        except Exception as e:
            logger.error(f"Error searching companies: {e}")
            return []
    
    def delete_analysis_by_id(self, analysis_id: int, company_number: str) -> bool:
        """Delete a specific analysis from any table"""
        with self.lock:
            try:
                with self._get_connection() as conn:
                    deleted = False
                    
                    # Try both tables
                    for table_name in ['analysis_results', 'analysis_history']:
                        try:
                            cursor = conn.execute(f'''
                                DELETE FROM {table_name} 
                                WHERE id = ? AND company_number = ?
                            ''', (analysis_id, company_number))
                            
                            if cursor.rowcount > 0:
                                deleted = True
                                logger.info(f"Deleted analysis {analysis_id} from {table_name}")
                        except Exception as e:
                            logger.warning(f"Error deleting from {table_name}: {e}")
                    
                    conn.commit()
                    return deleted
                        
            except Exception as e:
                logger.error(f"Error deleting analysis {analysis_id}: {e}")
                return False
    
    def cleanup_invalid_analyses(self, company_number: str) -> int:
        """Clean up invalid analyses for a company"""
        with self.lock:
            try:
                # Get all analyses for the company
                analyses = self.get_analysis_by_company(company_number)
                
                ids_to_delete = []
                
                for analysis in analyses:
                    issues = []
                    
                    # Check for specific issues
                    company_name = analysis.get('company_name', '')
                    if 'HSBC' in company_name and company_number == '02613335':
                        issues.append("wrong_company")
                    
                    business_reasoning = analysis.get('business_strategy_reasoning', '')
                    if business_reasoning == 'The company demonstrates strong growth-oriented strategies with focus on market expansion and innovation.':
                        issues.append("generic_business_reasoning")
                    
                    risk_reasoning = analysis.get('risk_strategy_reasoning', '')
                    if risk_reasoning == 'Conservative risk management approach with emphasis on regulatory compliance and stable operations.':
                        issues.append("generic_risk_reasoning")
                    
                    # Only delete if multiple issues (VERY CONSERVATIVE)
                    if len(issues) >= 2:
                        ids_to_delete.append(analysis['id'])
                
                # Delete identified analyses
                deleted_count = 0
                if ids_to_delete:
                    with self._get_connection() as conn:
                        for analysis_id in ids_to_delete:
                            # Try both tables
                            for table_name in ['analysis_results', 'analysis_history']:
                                try:
                                    cursor = conn.execute(f'''
                                        DELETE FROM {table_name} 
                                        WHERE id = ? AND company_number = ?
                                    ''', (analysis_id, company_number))
                                    
                                    if cursor.rowcount > 0:
                                        deleted_count += 1
                                except Exception as e:
                                    logger.warning(f"Error deleting from {table_name}: {e}")
                        
                        conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} invalid analyses for company {company_number}")
                return deleted_count
                
            except Exception as e:
                logger.error(f"Error cleaning up analyses for company {company_number}: {e}")
                return 0
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get database statistics from all tables"""
        try:
            with self._get_connection() as conn:
                stats = {}
                
                # Check all tables
                for table_name in ['analysis_results', 'analysis_history', 'analyses']:
                    try:
                        table_check = conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
                            (table_name,)
                        ).fetchone()
                        
                        if table_check:
                            # Total analyses
                            total_count = conn.execute(f'SELECT COUNT(*) FROM {table_name}').fetchone()[0]
                            
                            # Unique companies
                            company_count = conn.execute(f'SELECT COUNT(DISTINCT company_number) FROM {table_name}').fetchone()[0]
                            
                            # Recent activity (last 30 days)
                            recent_count = conn.execute(f'''
                                SELECT COUNT(*) FROM {table_name} 
                                WHERE created_at >= datetime('now', '-30 days')
                            ''').fetchone()[0]
                            
                            # Analysis types
                            type_stats = conn.execute(f'''
                                SELECT analysis_type, COUNT(*) 
                                FROM {table_name} 
                                GROUP BY analysis_type
                            ''').fetchall()
                            
                            stats[table_name] = {
                                'total_analyses': total_count,
                                'unique_companies': company_count,
                                'recent_analyses_30_days': recent_count,
                                'analysis_types': dict(type_stats)
                            }
                    except Exception as e:
                        stats[table_name] = {'error': str(e)}
                
                stats['database_path'] = self.db_path
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            return {'error': str(e)}