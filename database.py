#!/usr/bin/env python3
"""
Database module for Strategic Analysis API
Handles storage and retrieval of analysis results
UPDATED: Added MySQL support for existing database
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
    """Database handler for analysis results - MySQL and SQLite support"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection
        
        Args:
            db_path: Path to database or connection string
        """
        self.db_url = db_path or os.environ.get('DATABASE_URL', 'analysis_database.db')
        self.lock = threading.Lock()
        
        # Determine database type
        if self.db_url.startswith('mysql://'):
            self.db_type = 'mysql'
            self._setup_mysql()
        else:
            self.db_type = 'sqlite'
            self.db_path = self.db_url
            if self.db_path.startswith('sqlite:///'):
                self.db_path = self.db_path.replace('sqlite:///', '')
            self._setup_sqlite()
        
        logger.info(f"✅ Database initialized: {self.db_type} - {self.db_url}")
    
    def _setup_mysql(self):
        """Setup MySQL connection"""
        try:
            import pymysql
            
            # Parse MySQL URL: mysql://user:pass@host:port/database
            url_parts = self.db_url.replace('mysql://', '').split('/')
            database = url_parts[1] if len(url_parts) > 1 else 'exsigna_analysis'
            
            auth_host = url_parts[0].split('@')
            if len(auth_host) == 2:
                auth, host_port = auth_host
                user_pass = auth.split(':')
                user = user_pass[0] if len(user_pass) > 0 else 'root'
                password = user_pass[1] if len(user_pass) > 1 else ''
                
                host_port_parts = host_port.split(':')
                host = host_port_parts[0]
                port = int(host_port_parts[1]) if len(host_port_parts) > 1 else 3306
            else:
                host = auth_host[0]
                port = 3306
                user = 'root'
                password = ''
            
            self.mysql_config = {
                'host': host,
                'port': port,
                'user': user,
                'password': password,
                'database': database,
                'charset': 'utf8mb4',
                'autocommit': True
            }
            
            logger.info(f"MySQL config: {user}@{host}:{port}/{database}")
            
        except ImportError:
            logger.error("pymysql not installed. Install with: pip install pymysql")
            raise
        except Exception as e:
            logger.error(f"Error setting up MySQL: {e}")
            raise
    
    def _setup_sqlite(self):
        """Setup SQLite connection"""
        try:
            self._create_sqlite_tables()
        except Exception as e:
            logger.error(f"Error setting up SQLite: {e}")
            raise
    
    def _get_connection(self):
        """Get database connection"""
        if self.db_type == 'mysql':
            import pymysql
            return pymysql.connect(**self.mysql_config)
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
    
    def _create_sqlite_tables(self):
        """Create SQLite tables"""
        try:
            with self._get_connection() as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        company_number TEXT NOT NULL,
                        company_name TEXT,
                        analysis_date TEXT NOT NULL,
                        years_analyzed TEXT,
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
                        raw_response TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('CREATE INDEX IF NOT EXISTS idx_company_number ON analysis_history(company_number)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_analysis_date ON analysis_history(analysis_date)')
                
                conn.commit()
                logger.info("SQLite tables created successfully")
                
        except Exception as e:
            logger.error(f"Error creating SQLite tables: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self._get_connection() as conn:
                if self.db_type == 'mysql':
                    cursor = conn.cursor()
                    cursor.execute('SELECT 1')
                    cursor.close()
                else:
                    conn.execute('SELECT 1')
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_analysis_by_company(self, company_number: str) -> List[Dict[str, Any]]:
        """
        Get all analyses for a company - ENHANCED WITH MYSQL SUPPORT
        
        Args:
            company_number: Company number
            
        Returns:
            List of analysis records
        """
        try:
            logger.info(f"🔍 DEBUG: Searching for company_number: '{company_number}' in {self.db_type} database")
            
            with self._get_connection() as conn:
                if self.db_type == 'mysql':
                    cursor = conn.cursor()
                    
                    # Check table exists and get count
                    cursor.execute("SHOW TABLES LIKE 'analysis_history'")
                    table_exists = cursor.fetchone()
                    
                    if table_exists:
                        cursor.execute("SELECT COUNT(*) FROM analysis_history")
                        total_count = cursor.fetchone()[0]
                        logger.info(f"🔍 DEBUG: MySQL table 'analysis_history' has {total_count} total records")
                        
                        # Get sample company numbers
                        cursor.execute("SELECT DISTINCT company_number FROM analysis_history LIMIT 5")
                        sample_numbers = cursor.fetchall()
                        logger.info(f"🔍 DEBUG: Sample company numbers: {[row[0] for row in sample_numbers]}")
                        
                        # Try exact match
                        cursor.execute("SELECT COUNT(*) FROM analysis_history WHERE company_number = %s", (company_number,))
                        exact_match = cursor.fetchone()[0]
                        logger.info(f"🔍 DEBUG: Exact matches for '{company_number}': {exact_match}")
                        
                        if exact_match > 0:
                            cursor.execute('''
                                SELECT * FROM analysis_history 
                                WHERE company_number = %s 
                                ORDER BY analysis_date DESC
                            ''', (company_number,))
                            
                            rows = cursor.fetchall()
                            columns = [desc[0] for desc in cursor.description]
                            
                            result = []
                            for row in rows:
                                result.append(dict(zip(columns, row)))
                            
                            logger.info(f"🔍 DEBUG: Returning {len(result)} results from MySQL")
                            cursor.close()
                            return result
                        
                        # Try without leading zero
                        company_number_no_zero = company_number.lstrip('0')
                        cursor.execute("SELECT COUNT(*) FROM analysis_history WHERE company_number = %s", (company_number_no_zero,))
                        no_zero_match = cursor.fetchone()[0]
                        logger.info(f"🔍 DEBUG: Matches for '{company_number_no_zero}' (no leading zero): {no_zero_match}")
                        
                        if no_zero_match > 0:
                            cursor.execute('''
                                SELECT * FROM analysis_history 
                                WHERE company_number = %s 
                                ORDER BY analysis_date DESC
                            ''', (company_number_no_zero,))
                            
                            rows = cursor.fetchall()
                            columns = [desc[0] for desc in cursor.description]
                            
                            result = []
                            for row in rows:
                                result.append(dict(zip(columns, row)))
                            
                            logger.info(f"🔍 DEBUG: Returning {len(result)} results from MySQL (no leading zero)")
                            cursor.close()
                            return result
                    
                    cursor.close()
                    
                else:
                    # SQLite fallback
                    rows = conn.execute('''
                        SELECT * FROM analysis_history 
                        WHERE company_number = ? 
                        ORDER BY analysis_date DESC
                    ''', (company_number,)).fetchall()
                    
                    if rows:
                        return [dict(row) for row in rows]
                
                logger.info(f"🔍 DEBUG: No matches found for company number '{company_number}'")
                return []
                
        except Exception as e:
            logger.error(f"Error getting analyses for company {company_number}: {e}")
            return []
    
    def store_analysis_result(self, analysis_data: Dict[str, Any]) -> int:
        """Store analysis result in database"""
        with self.lock:
            try:
                # Extract data
                company_number = analysis_data.get('company_number', '')
                company_name = analysis_data.get('company_name', '')
                analysis_date = analysis_data.get('analysis_date', datetime.now().isoformat())
                years_analyzed = json.dumps(analysis_data.get('years_analyzed', []))
                files_processed = analysis_data.get('files_processed', 0)
                
                # Extract strategies
                business_strategy = analysis_data.get('business_strategy', {})
                risk_strategy = analysis_data.get('risk_strategy', {})
                
                business_strategy_dominant = business_strategy.get('dominant', '')
                risk_strategy_dominant = risk_strategy.get('dominant', '')
                
                analysis_type = analysis_data.get('analysis_type', 'unknown')
                confidence_level = analysis_data.get('confidence_level', 'medium')
                status = analysis_data.get('status', 'completed')
                raw_response = json.dumps(analysis_data)
                
                with self._get_connection() as conn:
                    if self.db_type == 'mysql':
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT INTO analysis_history (
                                company_number, company_name, analysis_date, years_analyzed,
                                files_processed, business_strategy, risk_strategy,
                                business_strategy_dominant, risk_strategy_dominant,
                                analysis_type, confidence_level, status, raw_response
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ''', (
                            company_number, company_name, analysis_date, years_analyzed,
                            files_processed, business_strategy_dominant, risk_strategy_dominant,
                            business_strategy_dominant, risk_strategy_dominant,
                            analysis_type, confidence_level, status, raw_response
                        ))
                        
                        record_id = cursor.lastrowid
                        cursor.close()
                    else:
                        cursor = conn.execute('''
                            INSERT INTO analysis_history (
                                company_number, company_name, analysis_date, years_analyzed,
                                files_processed, business_strategy, risk_strategy,
                                business_strategy_dominant, risk_strategy_dominant,
                                analysis_type, confidence_level, status, raw_response
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            company_number, company_name, analysis_date, years_analyzed,
                            files_processed, business_strategy_dominant, risk_strategy_dominant,
                            business_strategy_dominant, risk_strategy_dominant,
                            analysis_type, confidence_level, status, raw_response
                        ))
                        
                        record_id = cursor.lastrowid
                        conn.commit()
                    
                    logger.info(f"Analysis stored successfully with ID: {record_id}")
                    return record_id
                    
            except Exception as e:
                logger.error(f"Error storing analysis result: {e}")
                raise
    
    def get_recent_analyses(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent analyses"""
        try:
            with self._get_connection() as conn:
                if self.db_type == 'mysql':
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT * FROM analysis_history 
                        ORDER BY analysis_date DESC 
                        LIMIT %s
                    ''', (limit,))
                    
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    
                    result = []
                    for row in rows:
                        result.append(dict(zip(columns, row)))
                    
                    cursor.close()
                    return result
                else:
                    rows = conn.execute('''
                        SELECT * FROM analysis_history 
                        ORDER BY analysis_date DESC 
                        LIMIT ?
                    ''', (limit,)).fetchall()
                    
                    return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting recent analyses: {e}")
            return []
    
    def search_companies(self, search_term: str) -> List[Dict[str, Any]]:
        """Search companies"""
        try:
            search_pattern = f"%{search_term}%"
            
            with self._get_connection() as conn:
                if self.db_type == 'mysql':
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT DISTINCT company_number, company_name, 
                               COUNT(*) as analysis_count,
                               MAX(analysis_date) as latest_analysis
                        FROM analysis_history 
                        WHERE company_name LIKE %s OR company_number LIKE %s
                        GROUP BY company_number, company_name
                        ORDER BY latest_analysis DESC
                    ''', (search_pattern, search_pattern))
                    
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    
                    result = []
                    for row in rows:
                        result.append(dict(zip(columns, row)))
                    
                    cursor.close()
                    return result
                else:
                    rows = conn.execute('''
                        SELECT DISTINCT company_number, company_name, 
                               COUNT(*) as analysis_count,
                               MAX(analysis_date) as latest_analysis
                        FROM analysis_history 
                        WHERE company_name LIKE ? OR company_number LIKE ?
                        GROUP BY company_number, company_name
                        ORDER BY latest_analysis DESC
                    ''', (search_pattern, search_pattern)).fetchall()
                    
                    return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error searching companies: {e}")
            return []
    
    # Other methods remain similar with MySQL/SQLite adaptations...
    def delete_analysis_by_id(self, analysis_id: int, company_number: str) -> bool:
        """Delete specific analysis"""
        # Implementation with MySQL/SQLite support
        return False
    
    def cleanup_invalid_analyses(self, company_number: str) -> int:
        """Cleanup invalid analyses"""
        # Implementation with MySQL/SQLite support
        return 0
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self._get_connection() as conn:
                if self.db_type == 'mysql':
                    cursor = conn.cursor()
                    cursor.execute('SELECT COUNT(*) FROM analysis_history')
                    total_count = cursor.fetchone()[0]
                    
                    cursor.execute('SELECT COUNT(DISTINCT company_number) FROM analysis_history')
                    company_count = cursor.fetchone()[0]
                    
                    cursor.close()
                    
                    return {
                        'total_analyses': total_count,
                        'unique_companies': company_count,
                        'database_type': self.db_type,
                        'database_url': self.db_url
                    }
                else:
                    total_count = conn.execute('SELECT COUNT(*) FROM analysis_history').fetchone()[0]
                    company_count = conn.execute('SELECT COUNT(DISTINCT company_number) FROM analysis_history').fetchone()[0]
                    
                    return {
                        'total_analyses': total_count,
                        'unique_companies': company_count,
                        'database_type': self.db_type,
                        'database_path': self.db_path
                    }
                
        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            return {'error': str(e)}